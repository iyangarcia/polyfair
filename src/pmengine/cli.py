"""Command line: run the engine, scan for markets, report on results."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from .backfill import backfill
from .backtest import SYMBOLS, run_backtest
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
    spreads = [args.half_spread] if args.half_spread is not None else [0.005, 0.01, 0.02]
    asyncio.run(run_backtest(settings, args.asset, args.days, args.window_days, spreads,
                             args.depth_usd, Path(args.out), args.skip_open_hours))


def _backfill(args: argparse.Namespace) -> None:
    r = asyncio.run(backfill(args.asset, args.days, Path(args.out)))
    print(f"fetched {r.fetched} markets ({r.trades} trades) | already had {r.skipped} | "
          f"not resolved yet {r.unresolved} | in {args.out}")


def _record(args: argparse.Namespace) -> None:
    path = asyncio.run(snapshot(args.asset, Path(args.out), args.days_ahead, args.max_levels))
    if path is None:
        raise SystemExit("no open markets to snapshot")
    print(path)


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

    b = sub.add_parser("backtest", help="replay resolved daily strike markets; sends no orders")
    b.add_argument("--config", default="config.yaml", help="strategy/risk/vol settings (defaults if missing)")
    b.add_argument("--asset", choices=sorted(SYMBOLS), default="bitcoin")
    b.add_argument("--days", type=int, default=14, help="number of daily expiries, counting back from today")
    b.add_argument("--window-days", type=float, default=7, help="how long before each expiry to trade")
    b.add_argument("--half-spread", type=float, help="assumed half spread (default: run 0.005, 0.01, 0.02)")
    b.add_argument("--depth-usd", type=float, default=200, help="assumed size resting on each side")
    b.add_argument("--skip-open-hours", type=float, default=12,
                   help="ignore each market for this long after listing (placeholder prices)")
    b.add_argument("--out", default="data/backtest_trades.csv")
    b.set_defaults(fn=_backtest)

    f = sub.add_parser("backfill", help="download the trade history of resolved daily strike markets")
    f.add_argument("--asset", choices=sorted(SYMBOLS), default="bitcoin")
    f.add_argument("--days", type=int, default=30, help="number of daily expiries, counting back from today")
    f.add_argument("--out", default="data/trades")
    f.set_defaults(fn=_backfill)

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
