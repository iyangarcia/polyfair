"""The event loop: feeds -> fair value -> signal -> risk -> execution -> store."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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


async def discover(client: httpx.AsyncClient, s: Settings, now: datetime | None = None) -> list[MarketConfig]:
    """MarketConfigs for every open strike of each `discover` rule's daily ladder.

    Each market is named by its slug, so fills can be settled after the market
    has dropped out of a later day's discovery.
    """
    from .recorder import open_markets

    now = now or datetime.now(timezone.utc)
    out: list[MarketConfig] = []
    for rule in s.discover:
        for m in await open_markets(client, s.gamma_url, rule.asset, rule.days_ahead, now):
            if m["strike"] is None:
                continue
            out.append(MarketConfig(name=m["slug"], slug=m["slug"], strategy="crypto_drift", symbol=rule.symbol,
                                    kind="above", strike=m["strike"], min_edge=rule.min_edge))
        log.info("discovered %d open %s strikes", len(out), rule.asset)
    return out


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
            cfgs = list(self.s.markets)
            names = {c.name for c in cfgs}
            cfgs += [c for c in await discover(client, self.s) if c.name not in names]
            metas = await asyncio.gather(*(fetch_meta(client, self.s.gamma_url, m.slug) for m in cfgs))
            await self._settle_untracked(client, {c.name for c in cfgs})
        for cfg, meta in zip(cfgs, metas):
            if meta.closed:
                self._settle(cfg.name, meta)
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
        if self._by_symbol and self.s.vol.model == "seasonal":
            await self._warm_seasonal()
        self.pm_feed = PolymarketFeed(self.s.clob_ws, list(self._by_token), self._on_books)
        if self._by_symbol:
            self.binance = BinanceFeed(self.s.oracle.binance_ws, list(self._by_symbol), self.oracle, self._on_price)
        self.risk = RiskManager(self.s.risk, self.portfolio.equity(self._mark))

    async def _warm_seasonal(self) -> None:
        """Feed the last seasonal_days of Binance 1-minute candles to the oracle,
        so the time-of-day vol profile is ready at startup instead of after a week."""
        from .backtest import load_klines  # local import: backtest pulls in more than the engine needs

        now = time.time()
        start = now - (self.s.vol.seasonal_days + 1) * 86400
        cache = Path(self.s.db_path).parent / "cache"
        for symbol in self._by_symbol:
            candles = await asyncio.to_thread(load_klines, symbol, start, now, cache, self.s.oracle.binance_rest)
            for ts, px in candles:
                if ts <= now:
                    self.oracle.update(symbol, px, ts)
            log.info("seasonal vol for %s warmed with %d candles", symbol, len(candles))

    async def _settle_untracked(self, client: httpx.AsyncClient, tracked: set[str]) -> None:
        """Settle positions in markets that are no longer configured, e.g. yesterday's
        discovered ladder. Their fills carry the market slug as the market name."""
        stale = {p.market for p in self.portfolio.positions.values()} - tracked
        for name in sorted(stale):
            try:
                meta = await fetch_meta(client, self.s.gamma_url, name)
            except LookupError:
                log.warning("open position in %s, which is not a market slug and not configured; "
                            "it will not settle until it is added back", name)
                continue
            if meta.closed:
                self._settle(name, meta)
            else:
                log.warning("open position in %s, which is still open but not configured", name)

    def _settle(self, name: str, meta: MarketMeta) -> None:
        """Book the payout of a resolved market against any position still held."""
        prices = meta.outcome_prices
        if prices is None or sorted(prices) != [0.0, 1.0]:
            log.warning("market %s is closed but not resolved yet; skipping", name)
            return
        for token, outcome, payout in ((meta.yes_token, "YES", prices[0]), (meta.no_token, "NO", prices[1])):
            held = self.portfolio.held(token)
            if held > 0:
                fill = Fill(time.time(), self.mode, name, token, outcome, "SETTLE", held, payout, 0.0, "settlement")
                self.portfolio.apply(fill)
                self.store.record_fill(fill)
                log.info("settled %s %s: %.2f shares at %.0f", name, outcome, held, payout)
        log.info("market %s is resolved; not trading it", name)

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
                    vol = self.oracle.vol_to(m.cfg.symbol, now, m.strategy.expiry_ts)  # what fair value uses
                    extra = f" | {m.cfg.symbol} {tick[0]:.2f} vol to expiry {vol:.2f}"
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
