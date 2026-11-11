"""Download the full trade history of resolved daily strike markets.

Polymarket keeps every trade of a market after it resolves, so this needs no
always-on process: run it whenever you like and it fetches only the markets
it does not have yet. One gzipped JSON file per market lands in the output
directory, with the market's metadata and its trades in time order.

The `side` field is stored exactly as the Data API returns it. Whether it is
the aggressor's side or the side of the wallet named on the trade has not
been verified; check that before building a fill model on it.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from .gamma import parse_meta

log = logging.getLogger(__name__)

DATA_API = "https://data-api.polymarket.com/v2"
PAGE = 1000  # the Data API's maximum page size
MAX_PAGES = 500  # safety stop: 500k trades for one market


def event_slug(asset: str, d: date) -> str:
    """Slug of the daily strike ladder, e.g. bitcoin-above-on-november-5-2026."""
    return f"{asset}-above-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"


async def fetch_trades(client: httpx.AsyncClient, condition_id: str, base_url: str = DATA_API) -> list[dict]:
    """Every trade of one market, oldest first, de-duplicated."""
    rows: list[dict] = []
    cursor: str | None = None
    for _ in range(MAX_PAGES):
        params = {"conditionId": condition_id, "limit": PAGE}
        if cursor:
            params["cursor"] = cursor
        r = await client.get(f"{base_url}/trades", params=params)
        r.raise_for_status()
        body = r.json()
        rows += body.get("data", [])
        page = body.get("pagination") or {}
        cursor = page.get("next_cursor")
        if not page.get("has_more") or not cursor:
            break
    else:
        raise RuntimeError(f"{condition_id}: more than {MAX_PAGES} pages of trades; raise MAX_PAGES")

    out, seen = [], set()
    for t in rows:
        trade = {
            "ts": int(t["timestamp"]),
            "token_id": str(t["token_id"]),
            "outcome_index": int(t["outcome_index"]),
            "side": t["side"],
            "price": float(t["price"]),
            "size": float(t["size"]),
            "tx": t.get("transaction_hash", ""),
        }
        key = tuple(trade.values())
        if key not in seen:
            seen.add(key)
            out.append(trade)
    out.sort(key=lambda t: (t["ts"], t["tx"]))
    return out


@dataclass
class BackfillResult:
    fetched: int = 0  # markets downloaded this run
    skipped: int = 0  # already on disk
    unresolved: int = 0  # not resolved yet: try again later
    trades: int = 0


def market_path(out_dir: Path, slug: str) -> Path:
    return out_dir / f"{slug}.json.gz"


def write_market(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(record, f, separators=(",", ":"))
    tmp.replace(path)  # a crash never leaves a half-written file that looks complete


def read_market(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


async def backfill(asset: str, days: int, out_dir: Path, gamma_url: str = "https://gamma-api.polymarket.com",
                   data_url: str = DATA_API, end_day: date | None = None,
                   client: httpx.AsyncClient | None = None) -> BackfillResult:
    """Fetch trades for every resolved strike of the last `days` daily expiries."""
    end_day = end_day or datetime.now(timezone.utc).date()
    res = BackfillResult()
    sem = asyncio.Semaphore(4)
    own = client is None
    client = client or httpx.AsyncClient(timeout=30)
    try:
        async def one_market(raw: dict) -> None:
            meta = parse_meta(raw)
            path = market_path(out_dir, meta.slug)
            if path.exists():
                res.skipped += 1
                return
            if not meta.closed or meta.outcome_prices is None or sorted(meta.outcome_prices) != [0.0, 1.0]:
                res.unresolved += 1
                return
            async with sem:
                trades = await fetch_trades(client, meta.condition_id, data_url)
            strike = str(raw.get("groupItemTitle", "")).replace(",", "").replace("$", "")
            write_market(path, {
                "slug": meta.slug,
                "event": raw.get("_event_slug"),
                "question": meta.question,
                "condition_id": meta.condition_id,
                "yes_token": meta.yes_token,
                "no_token": meta.no_token,
                "strike": float(strike) if strike else None,
                "start_ts": raw.get("startDate"),
                "end_ts": meta.end_ts,
                "yes_won": meta.outcome_prices[0] == 1.0,
                "fee_rate": meta.fee_rate,
                "trades": trades,
            })
            res.fetched += 1
            res.trades += len(trades)
            log.info("%s: %d trades", meta.slug, len(trades))

        async def one_event(d: date) -> list[dict]:
            slug = event_slug(asset, d)
            async with sem:
                r = await client.get(f"{gamma_url}/events", params={"slug": slug})
            r.raise_for_status()
            rows = r.json()
            markets = rows[0].get("markets", []) if rows else []
            return [{**m, "_event_slug": slug} for m in markets]

        events = await asyncio.gather(*(one_event(end_day - timedelta(days=i)) for i in range(days)))
        await asyncio.gather(*(one_market(m) for ev in events for m in ev))
    finally:
        if own:
            await client.aclose()
    return res
