"""Command line: run the engine, scan for markets, report on results."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from .backfill import backfill
from .backtest import SYMBOLS, BtConfig, run_backtest
from .config import Settings, load_settings
from .engine import Engine
from .execution import LiveExecutor, PaperExecutor
from .gamma import parse_meta, scan
from .portfolio import Portfolio
from .recorder import snapshot
from .store import Store


def _run(args: argparse.Namespace) -> None:
    settings = load_settings(args.config)
    executor = LiveExecutor() if args.live else PaperExecutor(settings.paper.latency_ms)
    if args.live:
        logging.getLogger(__name__).warning("LIVE MODE: real orders will be sent to Polymarket")
    store = Store(settings.db_path)
    try:
        asyncio.run(Engine(settings, executor, store).run(args.duration))
    except KeyboardInterrupt:
        pass
    finally:
        store.close()


def _scan(args: argparse.Namespace) -> None:
    rows = asyncio.run(scan(args.gamma_url, args.query, args.limit))
    if not rows:
        print("no matching active markets")
    for raw in rows:
        m = parse_meta(raw)
        print(f"{m.slug}\n    {m.question}\n    ends {raw.get('endDate')} | bid {raw.get('bestBid')} "
              f"ask {raw.get('bestAsk')} | fee rate {m.fee_rate:g} | 24h vol ${float(raw.get('volume24hr') or 0):,.0f}")


def _backtest(args: argparse.Namespace) -> None:
    settings = load_settings(args.config) if Path(args.config).exists() else Settings()
    if args.vol_model:
        settings = settings.model_copy(update={"vol": settings.vol.model_copy(update={"model": args.vol_model})})
    cfg = BtConfig(strategy=args.strategy, step_s=args.step_s, lookback_s=args.lookback_s,
                   latency_s=args.latency_s, fill_window_s=args.fill_window_s, participation=args.participation,
                   window_days=args.window_days, warmup_days=args.warmup_days, skip_open_h=args.skip_open_hours,
                   shrink_days=args.shrink_days, symmetric=not args.no_symmetric, min_edge=args.min_edge)
    run_backtest(settings, cfg, Path(args.trades_dir), args.asset, args.days, Path(args.out))


def _backfill(args: argparse.Namespace) -> None:
    r = asyncio.run(backfill(args.asset, args.days, Path(args.out)))
    print(f"fetched {r.fetched} markets ({r.trades} trades) | already had {r.skipped} | "
          f"not resolved yet {r.unresolved} | in {args.out}")


def _record(args: argparse.Namespace) -> None:
    path = asyncio.run(snapshot(args.asset, Path(args.out), args.days_ahead, args.max_levels))
    if path is None:
        raise SystemExit("no open markets to snapshot")
    print(path)


def _paper_day(args: argparse.Namespace) -> None:
    from .paper import run_day

    run_day(load_settings(args.config), Path(args.state_dir), args.force, max_minutes=args.max_minutes)


def _report(args: argparse.Namespace) -> None:
    settings = load_settings(args.config)
    store = Store(settings.db_path)
    mode = "live" if args.live else "paper"
    fills = store.fills(mode)
    pf = Portfolio(settings.bankroll)
    pf.replay(fills)
    print(f"mode {mode} | fills {len(fills)} | cash {pf.cash:.2f} | open cost basis {pf.exposure():.2f} "
          f"| realized {pf.realized:+.2f} | fees {pf.fees:.2f}")
    for p in pf.positions.values():
        print(f"  open  {p.market:<24} {p.outcome:<3} {p.shares:>10.2f} sh @ avg {p.avg:.4f}")
    for status, n, ms, edge in store.order_stats(mode):
        print(f"  orders {status:<9} n={n:<5} avg decision {ms:.1f} ms | avg edge at signal {edge:.4f}")
    store.close()


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(prog="pmengine", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the engine (paper mode unless --live)")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--live", action="store_true", help="send real orders (also needs PM_ENABLE_LIVE=1)")
    r.add_argument("--duration", type=float, help="stop after this many seconds")
    r.set_defaults(fn=_run)

    s = sub.add_parser("scan", help="find active markets by words in the question")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=25)
    s.add_argument("--gamma-url", default="https://gamma-api.polymarket.com")
    s.set_defaults(fn=_scan)

    t = sub.add_parser("report", help="positions and PnL from the fill log")
    t.add_argument("--config", default="config.yaml")
    t.add_argument("--live", action="store_true")
    t.set_defaults(fn=_report)

    d = BtConfig()
    b = sub.add_parser("backtest", help="walk-forward backtest on backfilled real trades; sends no orders")
    b.add_argument("--config", default="config.yaml", help="strategy/risk/vol settings (defaults if missing)")
    b.add_argument("--trades-dir", default="data/trades", help="output of `pmengine backfill`")
    b.add_argument("--asset", choices=sorted(SYMBOLS), default="bitcoin")
    b.add_argument("--strategy", choices=["calibrated", "crypto_drift"], default=d.strategy)
    b.add_argument("--vol-model", choices=["ewma", "seasonal"], help="override vol.model for crypto_drift")
    b.add_argument("--days", type=int, help="use only the most recent N expiry days (default: all)")
    b.add_argument("--warmup-days", type=int, default=d.warmup_days,
                   help="first expiry days only train the calibration; nothing is traded")
    b.add_argument("--window-days", type=float, default=d.window_days, help="trade a market only this close to expiry")
    b.add_argument("--step-s", type=float, default=d.step_s, help="decision interval per market")
    b.add_argument("--lookback-s", type=float, default=d.lookback_s, help="max age of a trade used as a quote")
    b.add_argument("--latency-s", type=float, default=d.latency_s)
    b.add_argument("--fill-window-s", type=float, default=d.fill_window_s)
    b.add_argument("--participation", type=float, default=d.participation,
                   help="share of later takers' size the order could have taken")
    b.add_argument("--shrink-days", type=float, default=d.shrink_days)
    b.add_argument("--no-symmetric", action="store_true", help="do not pool YES and NO in the calibration")
    b.add_argument("--min-edge", type=float, help="override strategy.min_edge")
    b.add_argument("--skip-open-hours", type=float, default=d.skip_open_h,
                   help="ignore each market for this long after listing (placeholder prices)")
    b.add_argument("--out", default="data/backtest_trades.csv")
    b.set_defaults(fn=_backtest)

    f = sub.add_parser("backfill", help="download the trade history of resolved daily strike markets")
    f.add_argument("--asset", choices=sorted(SYMBOLS), default="bitcoin")
    f.add_argument("--days", type=int, default=30, help="number of daily expiries, counting back from today")
    f.add_argument("--out", default="data/trades")
    f.set_defaults(fn=_backfill)

    pd = sub.add_parser("paper-day", help="one unattended day of paper trading the discovered ladder")
    pd.add_argument("--config", default="paper.yaml")
    pd.add_argument("--state-dir", default="data/paper", help="fill log, run markers and summary.csv")
    pd.add_argument("--force", action="store_true", help="run now even outside the window or if already run")
    pd.add_argument("--max-minutes", type=float, help="stop after this long at most")
    pd.set_defaults(fn=_paper_day)

    c = sub.add_parser("record", help="save one order book snapshot of the open daily strike markets")
    c.add_argument("--asset", choices=sorted(SYMBOLS), default="bitcoin")
    c.add_argument("--days-ahead", type=int, default=1, help="also snapshot this many upcoming daily events")
    c.add_argument("--max-levels", type=int, default=50, help="price levels kept per side, best first")
    c.add_argument("--out", default="data/books")
    c.set_defaults(fn=_record)

    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    args.fn(args)


if __name__ == "__main__":
    main()
