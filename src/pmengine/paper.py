"""One unattended day of paper trading, for a scheduler such as GitHub Actions.

The tested setup trades only the last hours before each daily expiry, which
is noon New York time (16:00 UTC in summer, 17:00 UTC in winter). A runner
can start this twice a day, an hour apart; only the start that falls between
`min_lead_h` and `max_lead_h` before today's expiry trades, the other exits
at once. A marker file stops a second run for the same expiry day.

State lives in `state_dir`: the SQLite fill log (settings.db_path should point
inside it), runs/<day>.done markers, and summary.csv with one row per day.
Positions are held to resolution and settle at the start of the next run.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .config import Settings
from .engine import Engine
from .execution import PaperExecutor
from .portfolio import Portfolio
from .recorder import open_markets
from .store import Store

log = logging.getLogger(__name__)

STOP_BEFORE_S = 240.0  # the strategy stands down 300 s before expiry anyway
SUMMARY_COLS = ["day", "fills_today", "buys_today", "cash", "open_cost", "realized_total", "fees_total"]


async def todays_expiry(s: Settings, now: datetime) -> float | None:
    """Earliest expiry among the open discovered strikes, or None if nothing is open."""
    async with httpx.AsyncClient(timeout=20) as client:
        ends = []
        for rule in s.discover:
            ends += [m["end_ts"] for m in await open_markets(client, s.gamma_url, rule.asset, rule.days_ahead, now)
                     if m["end_ts"]]
    return min(ends) if ends else None


def should_run(expiry: float | None, now: float, min_lead_h: float, max_lead_h: float) -> tuple[bool, str]:
    if expiry is None:
        return False, "no open markets to trade today"
    lead_h = (expiry - now) / 3600.0
    if not min_lead_h < lead_h <= max_lead_h:
        return False, f"expiry is {lead_h:.2f} h away, outside ({min_lead_h:g}, {max_lead_h:g}] h"
    return True, f"expiry in {lead_h:.2f} h"


def append_summary(path: Path, day: str, s: Settings, store: Store, since: float) -> dict:
    fills = store.fills("paper")
    pf = Portfolio(s.bankroll)
    pf.replay(fills)
    today = [f for f in fills if f.ts >= since]
    row = {"day": day, "fills_today": len(today), "buys_today": sum(1 for f in today if f.side == "BUY"),
           "cash": round(pf.cash, 2), "open_cost": round(pf.exposure(), 2),
           "realized_total": round(pf.realized, 2), "fees_total": round(pf.fees, 2)}
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def run_day(s: Settings, state_dir: Path, force: bool = False, min_lead_h: float = 5.0,
            max_lead_h: float = 6.0, max_minutes: float | None = None) -> int:
    if not s.discover:
        raise SystemExit("paper-day needs a `discover` section in the config")
    now = datetime.now(timezone.utc)
    expiry = asyncio.run(todays_expiry(s, now))
    ok, why = should_run(expiry, now.timestamp(), min_lead_h, max_lead_h)
    if not ok and not force:
        print(f"skip: {why}")
        return 0
    if expiry is None:
        raise SystemExit("no open markets to trade")
    day = datetime.fromtimestamp(expiry, timezone.utc).strftime("%Y-%m-%d")
    marker = state_dir / "runs" / f"{day}.done"
    if marker.exists() and not force:
        print(f"skip: already ran for the {day} expiry")
        return 0

    duration = max(60.0, expiry - time.time() - STOP_BEFORE_S)
    if max_minutes is not None:
        duration = min(duration, max_minutes * 60.0)
    print(f"paper day {day}: {why}; running for {duration / 60:.0f} min")
    Path(s.db_path).parent.mkdir(parents=True, exist_ok=True)
    store = Store(s.db_path)
    started = time.time()
    try:
        asyncio.run(Engine(s, PaperExecutor(s.paper.latency_ms), store).run(duration))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now(timezone.utc).isoformat())
        row = append_summary(state_dir / "summary.csv", day, s, store, started)
        print("summary:", row)
    finally:
        store.close()
    return 0
