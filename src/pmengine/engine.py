"""The event loop: feeds -> fair value -> signal -> risk -> execution -> store."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

from .config import MarketConfig, Settings
from .execution import Executor
from .feeds.binance import BinanceFeed, Oracle
from .feeds.polymarket import PolymarketFeed
from .gamma import fetch_meta
from .models import Fill, MarketMeta
from .portfolio import Portfolio
from .risk import RiskManager
from .signals import generate
from .store import Store
from .strategies import BaseRateStrategy, CryptoDriftStrategy, Strategy

log = logging.getLogger(__name__)


@dataclass
class Market:
    cfg: MarketConfig
    meta: MarketMeta
    strategy: Strategy
    min_edge: float


class Engine:
    def __init__(self, settings: Settings, executor: Executor, store: Store) -> None:
        self.s = settings
        self.executor = executor
        self.store = store
        self.mode = executor.mode
        self.oracle = Oracle(settings.vol)
        self.portfolio = Portfolio(settings.bankroll)
        self.markets: dict[str, Market] = {}
        self.pm_feed: PolymarketFeed | None = None
        self.binance: BinanceFeed | None = None
        self.risk: RiskManager | None = None
        self._by_token: dict[str, str] = {}
        self._by_symbol: dict[str, list[str]] = {}
        self._dirty: dict[str, float] = {}  # market name -> time of first pending event
        self._wake = asyncio.Event()
        self._halt_logged = False

    # ---- setup -----------------------------------------------------------

    async def setup(self) -> None:
        self.portfolio.replay(self.store.fills(self.mode))
        async with httpx.AsyncClient(timeout=20) as client:
            metas = await asyncio.gather(*(fetch_meta(client, self.s.gamma_url, m.slug) for m in self.s.markets))
        for cfg, meta in zip(self.s.markets, metas):
            if meta.closed:
                self._settle(cfg, meta)
                continue
            if cfg.strategy == "crypto_drift":
                expiry = cfg.expiry.timestamp() if cfg.expiry else meta.end_ts
                if expiry is None:
                    raise ValueError(f"{cfg.name}: market has no end date; set expiry in the config")
                strat: Strategy = CryptoDriftStrategy(cfg, expiry, self.oracle, self.s.oracle)
                self._by_symbol.setdefault(cfg.symbol, []).append(cfg.name)
            else:
                strat = BaseRateStrategy(cfg)
            self.markets[cfg.name] = Market(cfg, meta, strat, cfg.min_edge or self.s.strategy.min_edge)
            self._by_token[meta.yes_token] = cfg.name
            self._by_token[meta.no_token] = cfg.name
            log.info("market %s: %s | fee rate %.3f | min size %g",
                     cfg.name, meta.question, meta.fee_rate, meta.min_size)
        if not self.markets:
            raise RuntimeError("no open markets in the config")
        self.pm_feed = PolymarketFeed(self.s.clob_ws, list(self._by_token), self._on_books)
        if self._by_symbol:
            self.binance = BinanceFeed(self.s.oracle.binance_ws, list(self._by_symbol), self.oracle, self._on_price)
        self.risk = RiskManager(self.s.risk, self.portfolio.equity(self._mark))

    def _settle(self, cfg: MarketConfig, meta: MarketMeta) -> None:
        """Book the payout of a resolved market against any position still held."""
        prices = meta.outcome_prices
        if prices is None or sorted(prices) != [0.0, 1.0]:
            log.warning("market %s is closed but not resolved yet; skipping", cfg.name)
            return
        for token, outcome, payout in ((meta.yes_token, "YES", prices[0]), (meta.no_token, "NO", prices[1])):
            held = self.portfolio.held(token)
            if held > 0:
                fill = Fill(time.time(), self.mode, cfg.name, token, outcome, "SETTLE", held, payout, 0.0, "settlement")
                self.portfolio.apply(fill)
                self.store.record_fill(fill)
                log.info("settled %s %s: %.2f shares at %.0f", cfg.name, outcome, held, payout)
        log.info("market %s is resolved; not trading it", cfg.name)

    # ---- event plumbing --------------------------------------------------

    def _mark_dirty(self, name: str) -> None:
        self._dirty.setdefault(name, time.time())
        self._wake.set()

    def _on_books(self, tokens: set[str]) -> None:
        for t in tokens:
            self._mark_dirty(self._by_token[t])

    def _on_price(self, symbol: str) -> None:
        for name in self._by_symbol.get(symbol, ()):
            self._mark_dirty(name)

    def _mark(self, token: str) -> float | None:
        book = self.pm_feed.books.get(token) if self.pm_feed else None
        return book.best_bid if book and book.ready else None

    # ---- core loop -------------------------------------------------------

    async def _worker(self) -> None:
        """Bursts of events for one market collapse into a single evaluation."""
        while True:
            await self._wake.wait()
            self._wake.clear()
            pending, self._dirty = self._dirty, {}
            for name, t_event in pending.items():
                await self._evaluate(self.markets[name], t_event)

    async def _evaluate(self, m: Market, t_event: float) -> None:
        now = time.time()
        if self.risk.halted or self.pm_feed.silence(now) > self.s.risk.max_feed_silence_s:
            return
        fair = m.strategy.fair_yes(now)
        if fair is None:
            return
        books = self.pm_feed.books
        signals = generate(
            m.cfg.name, fair, books[m.meta.yes_token], books[m.meta.no_token], m.meta,
            self.s.strategy, m.min_edge, self.portfolio.equity(self._mark),
            self.portfolio.held(m.meta.yes_token), self.portfolio.held(m.meta.no_token),
        )
        for sig in signals:
            decision = self.risk.review(sig, m.meta, self.portfolio, time.time())
            if not decision.ok:
                log.debug("%s %s %s rejected: %s", sig.market, sig.side, sig.outcome, decision.reason)
                continue
            sig = decision.signal
            self.risk.record_attempt(sig.token_id, time.time())
            decision_ms = (time.time() - t_event) * 1000.0
            res = await self.executor.execute(sig, books[sig.token_id])
            self.store.record_order(self.mode, sig, res.status, res.detail, decision_ms)
            if res.fill:
                f = res.fill
                self.portfolio.apply(f)
                self.store.record_fill(f)
                log.info("FILL %s %s %s %.2f @ %.4f (fair %.4f, edge %.4f, fee %.3f, decided in %.1f ms)",
                         f.market, f.side, f.outcome, f.shares, f.price, sig.fair, sig.edge, f.fee, decision_ms)
            else:
                log.info("order %s %s %s %.2f @ <=%.4f %s: %s",
                         sig.market, sig.side, sig.outcome, sig.shares, sig.limit, res.status, res.detail)
            if res.fatal:
                self.risk.halt(res.detail)

    async def _watchdog(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            self.risk.check_global(self.portfolio.equity(self._mark))
            if self.risk.halted and not self._halt_logged:
                self._halt_logged = True
                log.critical("TRADING HALTED: %s. Feeds stay up; restart the engine to resume.", self.risk.halted)

    def status(self) -> None:
        pf = self.portfolio
        equity = pf.equity(self._mark)
        self.store.record_equity(self.mode, equity, pf.cash, pf.exposure(), pf.realized, pf.fees)
        log.info("[%s] equity %.2f | cash %.2f | exposure %.2f | realized %+.2f | fees %.2f%s",
                 self.mode, equity, pf.cash, pf.exposure(), pf.realized, pf.fees,
                 " | HALTED" if self.risk.halted else "")
        now = time.time()
        for m in self.markets.values():
            book = self.pm_feed.books[m.meta.yes_token]
            fair = m.strategy.fair_yes(now)
            extra = ""
            if isinstance(m.strategy, CryptoDriftStrategy):
                tick = self.oracle.price(m.cfg.symbol)
                if tick:
                    extra = f" | {m.cfg.symbol} {tick[0]:.2f} vol {self.oracle.vol(m.cfg.symbol):.2f}"
            log.info("  %-24s fair %s | YES bid %s ask %s | held YES %.2f NO %.2f%s",
                     m.cfg.name, f"{fair:.4f}" if fair is not None else "n/a",
                     book.best_bid, book.best_ask,
                     pf.held(m.meta.yes_token), pf.held(m.meta.no_token), extra)

    async def _status_loop(self) -> None:
        while True:
            await asyncio.sleep(self.s.status_interval_s)
            self.status()

    async def run(self, duration: float | None = None) -> None:
        await self.setup()
        await self.executor.start()
        coros = [self.pm_feed.run(), self._worker(), self._watchdog(), self._status_loop()]
        if self.binance:
            coros.append(self.binance.run())
        tasks = [asyncio.create_task(c) for c in coros]
        try:
            done, _ = await asyncio.wait(tasks, timeout=duration, return_when=asyncio.FIRST_EXCEPTION)
            for t in done:
                t.result()  # surface a crashed task
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.executor.close()
            self.status()
