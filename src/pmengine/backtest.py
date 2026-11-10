"""Historical backtest of the crypto_drift strategy on resolved markets.

Replays Polymarket's 1-minute price history for the daily "<asset> above $X"
strike markets against Binance 1-minute candles, through the same signal,
risk and portfolio code the live engine uses, and settles every position at
the market's real outcome.

What it can and cannot tell you:
  * It tests whether the model disagrees with the market profitably at a
    one-minute horizon. It cannot test the sub-second latency edge.
  * Polymarket publishes historical prices, not historical order books. The
    quote is rebuilt as  price +/- half_spread  with `depth_usd` on each side,
    so results depend on that assumption; the report shows several spreads.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from .config import MarketConfig, Settings
from .feeds.binance import Oracle
from .gamma import parse_meta
from .models import Fill, MarketMeta
from .orderbook import OrderBook
from .portfolio import Portfolio
from .pricing import taker_fee
from .risk import RiskManager
from .signals import generate
from .strategies import CryptoDriftStrategy

log = logging.getLogger(__name__)

CLOB_URL = "https://clob.polymarket.com"
BINANCE_URL = "https://api.binance.com"
SYMBOLS = {"bitcoin": "BTCUSDT", "ethereum": "ETHUSDT"}


@dataclass
class BtMarket:
    name: str
    meta: MarketMeta
    strike: float
    start_ts: float  # when the market was listed
    end_ts: float
    yes_won: bool
    history: list[tuple[float, float]]  # (ts, YES price)


@dataclass
class BtResult:
    half_spread: float
    bankroll: float
    final_equity: float
    fees: float
    max_drawdown: float
    trades: list[dict] = field(default_factory=list)
    pnl_by_market: dict[str, float] = field(default_factory=dict)
    brier_model: float = float("nan")
    brier_market: float = float("nan")
    brier_n: int = 0

    @property
    def pnl(self) -> float:
        return self.final_equity - self.bankroll


# ---- data loading ------------------------------------------------------------

async def _cached(path: Path, fetch):
    if path.exists():
        return json.loads(path.read_text())
    data = await fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def _iso_ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


async def load_markets(settings: Settings, asset: str, days: int, window_days: float,
                       cache: Path, end_day: date | None = None) -> list[BtMarket]:
    """Resolved strike markets for the `days` daily expiries up to end_day."""
    end_day = end_day or datetime.now(timezone.utc).date()
    sem = asyncio.Semaphore(4)
    async with httpx.AsyncClient(timeout=30) as client:

        async def event(d: date) -> list[dict]:
            slug = f"{asset}-above-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"

            async def fetch():
                async with sem:
                    r = await client.get(f"{settings.gamma_url}/events", params={"slug": slug})
                r.raise_for_status()
                rows = r.json()
                mk = rows[0]["markets"] if rows else []
                if mk and not all(m.get("closed") for m in mk):
                    raise LookupError("not resolved yet")  # do not cache
                return mk

            try:
                return await _cached(cache / f"event_{slug}.json", fetch)
            except LookupError:
                return []

        async def history(raw: dict) -> BtMarket | None:
            meta = parse_meta(raw)
            if meta.outcome_prices is None or sorted(meta.outcome_prices) != [0.0, 1.0]:
                return None
            end = meta.end_ts
            listed = _iso_ts(raw["startDate"])
            start = max(listed, end - window_days * 86400)

            async def fetch():
                async with sem:
                    r = await client.get(f"{CLOB_URL}/prices-history", params={
                        "market": meta.yes_token, "startTs": int(start), "endTs": int(end), "fidelity": 1})
                r.raise_for_status()
                return r.json().get("history", [])

            h = await _cached(cache / f"hist_{meta.slug}_{window_days:g}d.json", fetch)
            strike = float(str(raw["groupItemTitle"]).replace(",", "").replace("$", ""))
            return BtMarket(meta.slug, meta, strike, listed, end, meta.outcome_prices[0] == 1.0,
                            [(float(x["t"]), float(x["p"])) for x in h])

        days_list = [end_day - timedelta(days=i) for i in range(days)]
        events = await asyncio.gather(*(event(d) for d in days_list))
        raws = [m for ev in events for m in ev]
        markets = await asyncio.gather(*(history(m) for m in raws))
    return [m for m in markets if m and m.history]


async def load_klines(symbol: str, start: float, end: float, cache: Path) -> list[tuple[float, float]]:
    """Binance 1-minute candles as (close time, close price)."""
    out: list[tuple[float, float]] = []
    async with httpx.AsyncClient(timeout=30) as client:
        day = int(start // 86400)
        while day * 86400 < end:

            async def fetch(day=day):
                rows = []
                for half in (0, 1):  # 1440 candles per day, 1000 per request
                    t0 = (day * 86400 + half * 43200) * 1000
                    r = await client.get(f"{BINANCE_URL}/api/v3/klines", params={
                        "symbol": symbol, "interval": "1m", "startTime": t0,
                        "endTime": t0 + 43200_000 - 1, "limit": 1000})
                    r.raise_for_status()
                    rows += [[(k[6] + 1) / 1000.0, float(k[4])] for k in r.json()]
                if (day + 1) * 86400 > datetime.now(timezone.utc).timestamp():
                    raise LookupError(rows)  # today is incomplete: use it, do not cache it
                return rows

            try:
                rows = await _cached(cache / f"klines_{symbol}_{day}.json", fetch)
            except LookupError as e:
                rows = e.args[0]
            out += [(r[0], r[1]) for r in rows]
            day += 1
    return out


# ---- simulation --------------------------------------------------------------

def _quote(book: OrderBook, mid: float, hs: float, depth_usd: float, ts: float) -> None:
    bid, ask = round(mid - hs, 4), round(mid + hs, 4)
    bids = [{"price": bid, "size": depth_usd / max(bid, 0.05)}] if bid > 0 else []
    asks = [{"price": ask, "size": depth_usd / max(ask, 0.05)}] if ask < 1 else []
    book.apply_snapshot(bids, asks, ts)


def simulate(markets: list[BtMarket], klines: list[tuple[float, float]], settings: Settings,
             symbol: str, half_spread: float, depth_usd: float, skip_open_h: float = 12.0) -> BtResult:
    oracle = Oracle(settings.vol)
    # Candles are a minute apart, so "fresh" means under three minutes here.
    ocfg = settings.oracle.model_copy(update={"max_age_s": 180.0})
    pf = Portfolio(settings.bankroll)
    risk = RiskManager(settings.risk, settings.bankroll)
    books: dict[str, OrderBook] = {}
    strats: list[CryptoDriftStrategy] = []
    for m in markets:
        cfg = MarketConfig(name=m.name, slug=m.name, strategy="crypto_drift",
                           symbol=symbol, kind="above", strike=m.strike)
        strats.append(CryptoDriftStrategy(cfg, m.end_ts, oracle, ocfg))
        books[m.meta.yes_token] = OrderBook()
        books[m.meta.no_token] = OrderBook()

    timeline = [(t, 0, i, p) for i, m in enumerate(markets) for t, p in m.history]
    timeline += [(m.end_ts, 1, i, 0.0) for i, m in enumerate(markets)]
    timeline.sort()

    res = BtResult(half_spread, settings.bankroll, settings.bankroll, 0.0, 0.0)
    pnl = res.pnl_by_market

    def mark(token: str) -> float | None:
        b = books.get(token)
        if b is None or not b.ready:
            return None
        return b.best_bid or 0.0  # quoted but no bid: worth nothing if sold now

    def book_fill(f: Fill, fair: float = float("nan"), edge: float = float("nan")) -> None:
        pf.apply(f)
        flow = f.shares * f.price
        pnl[f.market] = pnl.get(f.market, 0.0) + (-flow if f.side == "BUY" else flow) - f.fee
        res.trades.append({"ts": f.ts, "market": f.market, "side": f.side, "outcome": f.outcome,
                           "shares": round(f.shares, 2), "price": round(f.price, 4),
                           "fee": round(f.fee, 4), "fair": round(fair, 4), "edge": round(edge, 4)})

    k = 0
    peak = settings.bankroll
    next_sample = 0.0
    last_brier: dict[int, float] = {}
    b_model = b_market = 0.0
    for t, kind, i, p in timeline:
        while k < len(klines) and klines[k][0] <= t:  # only candles already closed: no lookahead
            oracle.update(symbol, klines[k][1], klines[k][0])
            k += 1
        m = markets[i]
        yes, no = m.meta.yes_token, m.meta.no_token
        if kind == 1:  # settlement at the real outcome
            for token, outcome, payout in ((yes, "YES", float(m.yes_won)), (no, "NO", float(not m.yes_won))):
                held = pf.held(token)
                if held > 0:
                    book_fill(Fill(t, "backtest", m.name, token, outcome, "SETTLE", held, payout, 0.0))
                books[token].clear()
            continue

        if t < m.start_ts + skip_open_h * 3600:
            continue  # a new listing shows a placeholder price (0.50) until real quotes arrive
        _quote(books[yes], p, half_spread, depth_usd, t)
        _quote(books[no], 1.0 - p, half_spread, depth_usd, t)
        fair = strats[i].fair_yes(t)
        if fair is None:
            continue
        if t - last_brier.get(i, 0.0) >= 3600:  # hourly accuracy sample: model vs market
            last_brier[i] = t
            y = float(m.yes_won)
            b_model += (fair - y) ** 2
            b_market += (p - y) ** 2
            res.brier_n += 1

        signals = generate(m.name, fair, books[yes], books[no], m.meta, settings.strategy,
                           settings.strategy.min_edge, pf.equity(mark), pf.held(yes), pf.held(no))
        for sig in signals:
            d = risk.review(sig, m.meta, pf, t)
            if not d.ok:
                continue
            sig = d.signal
            risk.record_attempt(sig.token_id, t)
            filled, notional = books[sig.token_id].sweep(sig.side, sig.shares, sig.limit)
            if filled <= 0:
                continue
            price = notional / filled
            fee = filled * taker_fee(price, sig.fee_rate, sig.fee_exponent)
            book_fill(Fill(t, "backtest", m.name, sig.token_id, sig.outcome, sig.side, filled, price, fee),
                      sig.fair, sig.edge)

        if t >= next_sample:  # drawdown on a mark-to-bid equity curve, sampled every 10 minutes
            next_sample = t + 600
            eq = pf.equity(mark)
            peak = max(peak, eq)
            res.max_drawdown = max(res.max_drawdown, peak - eq)

    res.final_equity = pf.equity(mark)
    res.fees = pf.fees
    if res.brier_n:
        res.brier_model, res.brier_market = b_model / res.brier_n, b_market / res.brier_n
    return res


# ---- reporting ---------------------------------------------------------------

def summary_row(r: BtResult) -> str:
    entries = [t for t in r.trades if t["side"] == "BUY"]
    traded = [v for v in r.pnl_by_market.values()]
    wins = sum(1 for v in traded if v > 0)
    return (f"{r.half_spread:>11.3f} {len(entries):>7} {len(traded):>8} "
            f"{(100 * wins / len(traded) if traded else 0):>6.0f}% {r.fees:>8.2f} "
            f"{r.pnl:>+9.2f} {100 * r.pnl / r.bankroll:>+7.2f}% {r.max_drawdown:>9.2f}")


def print_report(results: list[BtResult], markets: list[BtMarket], main: BtResult) -> None:
    first = min(m.end_ts for m in markets)
    last = max(m.end_ts for m in markets)
    day = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    print(f"\nBacktest: {len(markets)} resolved markets, expiries {day(first)} to {day(last)}, "
          f"bankroll {main.bankroll:.0f}")
    print(f"\nForecast accuracy (Brier score, lower is better, {main.brier_n} hourly samples):")
    print(f"  model {main.brier_model:.4f}   market {main.brier_market:.4f}   "
          f"-> the {'model' if main.brier_model < main.brier_market else 'market'} was more accurate")
    print("\nhalf_spread  entries  markets   wins     fees       pnl   return   max_dd")
    for r in results:
        print(summary_row(r))

    print(f"\nBy expiry day (half_spread {main.half_spread:g}):")
    by_day: dict[str, list[float]] = {}
    ends = {m.name: m.end_ts for m in markets}
    for name, v in main.pnl_by_market.items():
        by_day.setdefault(day(ends[name]), []).append(v)
    for d in sorted(by_day):
        v = by_day[d]
        print(f"  {d}  markets traded {len(v):>2}  pnl {sum(v):>+8.2f}")
    ranked = sorted(main.pnl_by_market.items(), key=lambda kv: kv[1])
    if ranked:
        print("\nWorst and best markets:")
        for name, v in ranked[:3] + ranked[-3:][::-1] if len(ranked) > 6 else ranked:
            print(f"  {v:>+8.2f}  {name}")


def write_trades(path: Path, r: BtResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "market", "side", "outcome", "shares", "price", "fee", "fair", "edge"])
        w.writeheader()
        for t in r.trades:
            row = dict(t)
            row["time"] = datetime.fromtimestamp(row.pop("ts"), timezone.utc).isoformat(timespec="seconds")
            w.writerow(row)


async def run_backtest(settings: Settings, asset: str, days: int, window_days: float,
                       spreads: list[float], depth_usd: float, out: Path,
                       skip_open_h: float = 12.0) -> list[BtResult]:
    cache = Path(settings.db_path).parent / "cache"
    symbol = SYMBOLS[asset]
    markets = await load_markets(settings, asset, days, window_days, cache)
    if not markets:
        raise RuntimeError("no resolved markets found for that range")
    start = min(m.history[0][0] for m in markets) - 6 * 3600  # warm the vol estimate up
    klines = await load_klines(symbol, start, max(m.end_ts for m in markets), cache)
    log.info("loaded %d markets, %d price points, %d candles", len(markets),
             sum(len(m.history) for m in markets), len(klines))
    results = [simulate(markets, klines, settings, symbol, hs, depth_usd, skip_open_h) for hs in spreads]
    main = results[len(results) // 2]
    print_report(results, markets, main)
    write_trades(out, main)
    print(f"\nTrades for half_spread {main.half_spread:g} written to {out}")
    return results
