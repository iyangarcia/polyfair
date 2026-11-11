"""Walk-forward backtest on real Polymarket trades.

Data comes from `pmengine backfill`: the full taker trade tape of each
resolved daily strike market, plus Binance 1-minute candles for the
crypto_drift model. Every `step_s` seconds of each market's trading window:

  1. Quotes are read from the tape *before* t: the ask of a token is what the
     latest taker paid for it, the bid what the latest taker seller received
     (or 1 - the other token's ask). No quote within `lookback_s`: no trade.
  2. The strategy gives a fair P(YES); the engine's own signal and risk code
     turn it into orders.
  3. An order fills only against real taker trades on the same side that
     happened after t + latency, at or better than its limit, up to
     `participation` of their size. Positions settle at the real outcome.

The `calibrated` strategy is refitted as markets resolve, using only markets
settled at least an hour before the decision, and nothing is traded during
the first `warmup_days` expiry days, which only feed the fit.
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .backfill import read_market
from .calibration import Calibration
from .config import MarketConfig, Settings
from .feeds.binance import Oracle
from .models import Fill, MarketMeta
from .orderbook import OrderBook
from .portfolio import Portfolio
from .pricing import taker_fee
from .risk import RiskManager
from .signals import generate
from .strategies import CryptoDriftStrategy
from .tape import Tape, Trade

log = logging.getLogger(__name__)

BINANCE_URL = "https://api.binance.com"
SYMBOLS = {"bitcoin": "BTCUSDT", "ethereum": "ETHUSDT"}
SETTLE_LAG_S = 3600.0  # a resolved market feeds the calibration this long after expiry
QUOTE_SIZE = 1e9  # the tape, not the quote, limits fills


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


@dataclass
class BtMarket:
    name: str
    meta: MarketMeta
    strike: float | None
    start_ts: float
    end_ts: float
    yes_won: bool
    tape: Tape

    @property
    def day(self) -> str:
        return _day(self.end_ts)


@dataclass
class BtConfig:
    strategy: str = "calibrated"  # calibrated | crypto_drift
    step_s: float = 300.0  # decision interval per market
    lookback_s: float = 900.0  # a quote older than this is no quote
    latency_s: float = 1.0  # order reaches the book this long after the decision
    fill_window_s: float = 60.0  # how long after arrival later trades count as fillable
    participation: float = 0.5  # share of later takers' size we could have taken
    window_days: float = 3.0  # trade a market only this close to expiry
    warmup_days: int = 30  # first expiry days only feed the calibration
    skip_open_h: float = 12.0  # ignore placeholder trading right after listing
    shrink_days: float = 20.0
    symmetric: bool = True
    min_edge: float | None = None  # overrides settings.strategy.min_edge
    seed: int = 7


@dataclass
class BtResult:
    bankroll: float
    final_equity: float = 0.0
    fees: float = 0.0
    max_drawdown: float = 0.0
    orders: int = 0  # approved by risk and sent to the tape
    filled: int = 0
    trades: list[dict] = field(default_factory=list)
    pnl_by_market: dict[str, float] = field(default_factory=dict)
    pnl_by_day: dict[str, float] = field(default_factory=dict)
    test_days: list[str] = field(default_factory=list)
    brier_model: float = float("nan")
    brier_market: float = float("nan")
    brier_n: int = 0

    @property
    def pnl(self) -> float:
        return self.final_equity - self.bankroll


# ---- data ----------------------------------------------------------------------

def market_from_record(rec: dict) -> BtMarket:
    meta = MarketMeta(rec["slug"], rec.get("question", ""), rec["condition_id"], rec["yes_token"], rec["no_token"],
                      5.0, float(rec.get("fee_rate") or 0.0), 1.0, rec["end_ts"], True,
                      (1.0, 0.0) if rec["yes_won"] else (0.0, 1.0))
    trades = [Trade(float(t["ts"]), t["token_id"], int(t["outcome_index"]), t["side"], float(t["price"]),
                    float(t["size"])) for t in rec["trades"]]
    start = rec.get("start_ts")
    start_ts = datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp() if start else \
        (trades[0].ts if trades else rec["end_ts"])
    return BtMarket(rec["slug"], meta, rec.get("strike"), start_ts, float(rec["end_ts"]), bool(rec["yes_won"]),
                    Tape(trades))


def load_markets(trades_dir: Path, asset: str = "bitcoin", days: int | None = None) -> list[BtMarket]:
    """Backfilled markets of `asset`, limited to the most recent `days` expiry days."""
    paths = sorted(glob.glob(str(trades_dir / f"{asset}-above-*.json.gz")))
    markets = [market_from_record(read_market(Path(p))) for p in paths]
    markets = [m for m in markets if m.tape.trades]
    if days:
        keep = set(sorted({m.day for m in markets})[-days:])
        markets = [m for m in markets if m.day in keep]
    return sorted(markets, key=lambda m: m.end_ts)


def load_klines(symbol: str, start: float, end: float, cache: Path,
                base_url: str = BINANCE_URL) -> list[tuple[float, float]]:
    """Binance 1-minute candles as (close time, close price), cached per UTC day."""
    out: list[tuple[float, float]] = []
    with httpx.Client(timeout=30) as client:
        day = int(start // 86400)
        while day * 86400 < end:
            path = cache / f"klines_{symbol}_{day}.json"
            if path.exists():
                rows = json.loads(path.read_text())
            else:
                rows = []
                for half in (0, 1):  # 1440 candles per day, 1000 per request
                    t0 = (day * 86400 + half * 43200) * 1000
                    r = client.get(f"{base_url}/api/v3/klines", params={
                        "symbol": symbol, "interval": "1m", "startTime": t0,
                        "endTime": t0 + 43200_000 - 1, "limit": 1000})
                    r.raise_for_status()
                    rows += [[(k[6] + 1) / 1000.0, float(k[4])] for k in r.json()]
                if (day + 1) * 86400 <= datetime.now(timezone.utc).timestamp():  # only complete days
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(rows))
            out += [(r[0], r[1]) for r in rows]
            day += 1
    return out


def market_samples(m: BtMarket, step_s: float, lookback_s: float, skip_open_h: float) -> list[tuple[float, float]]:
    """(hours to expiry, YES price) on a fixed grid: what the calibration learns from."""
    out = []
    t = m.start_ts + skip_open_h * 3600
    while t < m.end_ts:
        p = m.tape.last_yes_price(t, lookback_s)
        if p is not None:
            out.append(((m.end_ts - t) / 3600.0, p))
        t += step_s
    return out


# ---- simulation ----------------------------------------------------------------

def _quote(book: OrderBook, ask: float | None, bid: float | None, ts: float) -> None:
    if ask is not None and bid is not None and bid >= ask:
        bid = None  # stale or crossed evidence: do not trust a bid above the ask
    book.apply_snapshot([{"price": bid, "size": QUOTE_SIZE}] if bid else [],
                        [{"price": ask, "size": QUOTE_SIZE}] if ask else [], ts)


def simulate(markets: list[BtMarket], settings: Settings, cfg: BtConfig,
             klines: list[tuple[float, float]] | None = None, symbol: str = "BTCUSDT") -> BtResult:
    markets = sorted(markets, key=lambda m: m.end_ts)
    days = sorted({m.day for m in markets})
    test_days = set(days[cfg.warmup_days:])
    min_edge = cfg.min_edge if cfg.min_edge is not None else settings.strategy.min_edge
    min_tte = settings.oracle.min_time_to_expiry_s

    cal = Calibration(cfg.shrink_days, cfg.symmetric)
    samples = [market_samples(m, cfg.step_s, cfg.lookback_s, cfg.skip_open_h) for m in markets]
    fitted = 0  # markets[:fitted] are in the calibration

    oracle = Oracle(settings.vol)
    ocfg = settings.oracle.model_copy(update={"max_age_s": 180.0})  # one-minute candles
    drift: dict[int, CryptoDriftStrategy] = {}
    if cfg.strategy == "crypto_drift":
        for i, m in enumerate(markets):
            if m.strike:
                mc = MarketConfig(name=m.name, slug=m.name, strategy="crypto_drift", symbol=symbol,
                                  kind="above", strike=m.strike)
                drift[i] = CryptoDriftStrategy(mc, m.end_ts, oracle, ocfg)

    pf = Portfolio(settings.bankroll)
    risk = RiskManager(settings.risk, settings.bankroll)
    books = {t: OrderBook() for m in markets for t in (m.meta.yes_token, m.meta.no_token)}
    res = BtResult(settings.bankroll, test_days=sorted(test_days))
    last_yes: dict[str, float] = {}

    def mark(token: str) -> float | None:
        return last_yes.get(token)

    def book_fill(m: BtMarket, f: Fill, fair: float = float("nan"), edge: float = float("nan")) -> None:
        pf.apply(f)
        flow = f.shares * f.price
        v = (-flow if f.side == "BUY" else flow) - f.fee
        res.pnl_by_market[m.name] = res.pnl_by_market.get(m.name, 0.0) + v
        res.pnl_by_day[m.day] = res.pnl_by_day.get(m.day, 0.0) + v
        res.trades.append({"ts": f.ts, "market": m.name, "side": f.side, "outcome": f.outcome,
                           "shares": round(f.shares, 2), "price": round(f.price, 4), "fee": round(f.fee, 4),
                           "fair": round(fair, 4), "edge": round(edge, 4),
                           "tte_h": round((m.end_ts - f.ts) / 3600.0, 3)})

    timeline: list[tuple[float, int, int]] = []
    for i, m in enumerate(markets):
        if m.day not in test_days:
            continue
        t = max(m.start_ts + cfg.skip_open_h * 3600, m.end_ts - cfg.window_days * 86400)
        t = math.ceil(t / cfg.step_s) * cfg.step_s
        while t < m.end_ts - min_tte:
            timeline.append((t, 0, i))
            t += cfg.step_s
        timeline.append((m.end_ts, 1, i))
    timeline.sort()

    k = 0
    peak = settings.bankroll
    next_sample = 0.0
    last_brier: dict[int, float] = {}
    b_model = b_market = 0.0
    for t, kind, i in timeline:
        m = markets[i]
        yes, no = m.meta.yes_token, m.meta.no_token
        if kind == 1:  # settlement at the real outcome
            for token, outcome, payout in ((yes, "YES", float(m.yes_won)), (no, "NO", float(not m.yes_won))):
                held = pf.held(token)
                if held > 0:
                    book_fill(m, Fill(t, "backtest", m.name, token, outcome, "SETTLE", held, payout, 0.0))
                last_yes.pop(token, None)
            continue

        while fitted < len(markets) and markets[fitted].end_ts + SETTLE_LAG_S <= t:
            mk = markets[fitted]
            cal.add(samples[fitted], mk.yes_won, mk.day)
            fitted += 1
        if klines is not None:
            while k < len(klines) and klines[k][0] <= t:  # only candles already closed
                oracle.update(symbol, klines[k][1], klines[k][0])
                k += 1

        p = m.tape.last_yes_price(t, cfg.lookback_s)
        if p is None:
            continue
        last_yes[yes], last_yes[no] = p, 1.0 - p
        tte_h = (m.end_ts - t) / 3600.0
        if cfg.strategy == "calibrated":
            fair = cal.fair(tte_h, p)
        elif i in drift:
            fair = drift[i].fair_yes(t)
        else:
            fair = None
        if fair is None:
            continue

        if t - last_brier.get(i, -math.inf) >= 3600:  # hourly accuracy sample: model vs market
            last_brier[i] = t
            y = float(m.yes_won)
            b_model += (fair - y) ** 2
            b_market += (p - y) ** 2
            res.brier_n += 1

        for token, other in ((yes, no), (no, yes)):
            ask = m.tape.last_paid(token, "BUY", t, cfg.lookback_s)
            bid = m.tape.last_paid(token, "SELL", t, cfg.lookback_s)
            if bid is None:
                other_ask = m.tape.last_paid(other, "BUY", t, cfg.lookback_s)
                bid = 1.0 - other_ask if other_ask is not None else None
            _quote(books[token], ask, bid, t)

        signals = generate(m.name, fair, books[yes], books[no], m.meta, settings.strategy, min_edge,
                           pf.equity(mark), pf.held(yes), pf.held(no))
        for sig in signals:
            d = risk.review(sig, m.meta, pf, t)
            if not d.ok:
                continue
            sig = d.signal
            risk.record_attempt(sig.token_id, t)
            res.orders += 1
            t0 = t + cfg.latency_s
            filled, notional = m.tape.fill(sig.token_id, sig.side, sig.shares, sig.limit, t0,
                                           t0 + cfg.fill_window_s, cfg.participation)
            if filled <= 0:
                continue
            res.filled += 1
            price = notional / filled
            fee = filled * taker_fee(price, sig.fee_rate, sig.fee_exponent)
            book_fill(m, Fill(t0, "backtest", m.name, sig.token_id, sig.outcome, sig.side, filled, price, fee),
                      sig.fair, sig.edge)

        if t >= next_sample:  # drawdown on equity marked at the last trade price, every 10 minutes
            next_sample = t + 600
            eq = pf.equity(mark)
            peak = max(peak, eq)
            res.max_drawdown = max(res.max_drawdown, peak - eq)

    res.final_equity = pf.equity(mark)
    res.fees = pf.fees
    if res.brier_n:
        res.brier_model, res.brier_market = b_model / res.brier_n, b_market / res.brier_n
    return res


# ---- statistics and reporting --------------------------------------------------

def daily_pnl(r: BtResult) -> list[float]:
    """PnL of every test day, zero on days nothing was traded."""
    return [r.pnl_by_day.get(d, 0.0) for d in r.test_days]


def bootstrap_mean(xs: list[float], n: int = 2000, seed: int = 7) -> tuple[float, float]:
    """95% interval of the mean, resampling whole days."""
    if not xs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def sharpe(xs: list[float]) -> float:
    """Annualised Sharpe ratio of daily PnL (365 trading days: crypto never closes)."""
    if len(xs) < 2:
        return float("nan")
    mu = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))
    return mu / sd * math.sqrt(365) if sd > 0 else float("nan")


def _bucket_table(r: BtResult, title: str, key) -> list[str]:
    entries = [t for t in r.trades if t["side"] == "BUY"]
    groups: dict[str, list[dict]] = {}
    for t in entries:
        groups.setdefault(key(t), []).append(t)
    lines = [f"\n{title}", f"  {'bucket':<10}{'entries':>8}{'staked':>10}{'avg px':>8}{'avg fair':>9}"]
    for g in sorted(groups):
        ts = groups[g]
        sh = sum(t["shares"] for t in ts)
        cost = sum(t["shares"] * t["price"] for t in ts)
        fair = sum(t["fair"] * t["shares"] for t in ts) / sh if sh else 0
        lines.append(f"  {g:<10}{len(ts):>8}{cost:>10.0f}{cost / sh if sh else 0:>8.3f}{fair:>9.3f}")
    return lines


def report(r: BtResult, markets: list[BtMarket]) -> str:
    xs = daily_pnl(r)
    lo, hi = bootstrap_mean(xs)
    half = len(xs) // 2
    first, second = sum(xs[:half]), sum(xs[half:])
    traded_days = sum(1 for x in xs if x != 0)
    pos_days = sum(1 for x in xs if x > 0)
    lines = [
        f"\nWalk-forward backtest on real trades: {len(markets)} markets, "
        f"test days {r.test_days[0] if r.test_days else '-'} to {r.test_days[-1] if r.test_days else '-'} "
        f"({len(r.test_days)} days), bankroll {r.bankroll:.0f}",
        f"\nForecast accuracy (Brier, lower is better, {r.brier_n} hourly samples): "
        f"model {r.brier_model:.4f}  market {r.brier_market:.4f}",
        f"\nPnL {r.pnl:+.2f} ({100 * r.pnl / r.bankroll:+.1f}%) | fees {r.fees:.2f} | max drawdown {r.max_drawdown:.2f}",
        f"orders {r.orders} | filled {r.filled} ({100 * r.filled / r.orders if r.orders else 0:.0f}%) | "
        f"entries {sum(1 for t in r.trades if t['side'] == 'BUY')}",
        f"daily PnL: mean {sum(xs) / len(xs) if xs else 0:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}] | "
        f"Sharpe {sharpe(xs):.2f} | traded {traded_days} days, positive {pos_days}",
        f"first half of test {first:+.2f} | second half {second:+.2f}",
    ]

    def tte_key(t):
        h = t["tte_h"]
        return "a <1h" if h < 1 else "b 1-6h" if h < 6 else "c 6-24h" if h < 24 else "d >1d"

    def px_key(t):
        p = t["price"]
        return "a <0.10" if p < 0.10 else "b 0.10-0.50" if p < 0.50 else "c 0.50-0.90" if p < 0.90 else "d >=0.90"

    lines += _bucket_table(r, "Entries by time to expiry", tte_key)
    lines += _bucket_table(r, "Entries by price paid", px_key)
    return "\n".join(lines)


def write_trades(path: Path, r: BtResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        cols = ["time", "market", "side", "outcome", "shares", "price", "fee", "fair", "edge", "tte_h"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in r.trades:
            row = dict(t)
            row["time"] = datetime.fromtimestamp(row.pop("ts"), timezone.utc).isoformat(timespec="seconds")
            w.writerow(row)


def run_backtest(settings: Settings, cfg: BtConfig, trades_dir: Path, asset: str, days: int | None,
                 out: Path) -> BtResult:
    markets = load_markets(trades_dir, asset, days)
    if not markets:
        raise RuntimeError(f"no backfilled {asset} markets in {trades_dir}: run `pmengine backfill` first")
    klines = None
    if cfg.strategy == "crypto_drift":
        warm = (settings.vol.seasonal_days + 1) * 86400 if settings.vol.model == "seasonal" else 6 * 3600
        start = min(m.start_ts for m in markets) - warm
        klines = load_klines(SYMBOLS[asset], start, max(m.end_ts for m in markets),
                             Path(settings.db_path).parent / "cache")
    log.info("loaded %d markets, %d trades", len(markets), sum(len(m.tape.trades) for m in markets))
    r = simulate(markets, settings, cfg, klines, SYMBOLS[asset])
    print(report(r, markets))
    write_trades(out, r)
    print(f"\nTrades written to {out}")
    return r
