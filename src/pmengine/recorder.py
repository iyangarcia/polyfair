"""One-shot order book snapshots of the open daily strike markets.

Polymarket only serves the current book, so history exists only if someone
saves it. This takes one snapshot and exits, which suits a scheduler such as
GitHub Actions cron: no process has to stay up. Each run writes one small
gzipped file, out_dir/YYYY-MM-DD/HHMMSS.json.gz, so files never change after
they are written and git stores each one once.
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from .backfill import event_slug
from .gamma import parse_meta

log = logging.getLogger(__name__)

CLOB_URL = "https://clob.polymarket.com"
CHUNK = 50  # tokens per POST /books request


def _levels(raw: list[dict], best_first_desc: bool, max_levels: int) -> list[list[float]]:
    lv = sorted(((float(x["price"]), float(x["size"])) for x in raw), reverse=best_first_desc)
    return [[p, s] for p, s in lv[:max_levels]]


async def open_markets(client: httpx.AsyncClient, gamma_url: str, asset: str, days_ahead: int,
                       now: datetime) -> list[dict]:
    """Open, order-book-enabled strikes of today's and the next `days_ahead` daily events."""
    days = [now.date() + timedelta(days=i) for i in range(days_ahead + 1)]
    out = []
    for d in days:
        r = await client.get(f"{gamma_url}/events", params={"slug": event_slug(asset, d)})
        r.raise_for_status()
        rows = r.json()
        for raw in rows[0].get("markets", []) if rows else []:
            if raw.get("closed") or not raw.get("enableOrderBook"):
                continue
            meta = parse_meta(raw)
            strike = str(raw.get("groupItemTitle", "")).replace(",", "").replace("$", "")
            out.append({"slug": meta.slug, "strike": float(strike) if strike else None, "end_ts": meta.end_ts,
                        "yes_token": meta.yes_token, "no_token": meta.no_token})
    return out


async def fetch_books(client: httpx.AsyncClient, tokens: list[str], clob_url: str = CLOB_URL,
                      max_levels: int = 50) -> list[dict]:
    books = []
    for i in range(0, len(tokens), CHUNK):
        r = await client.post(f"{clob_url}/books", json=[{"token_id": t} for t in tokens[i:i + CHUNK]])
        r.raise_for_status()
        for b in r.json():
            books.append({
                "asset_id": b["asset_id"],
                "ts": int(b["timestamp"]) / 1000.0,  # exchange time of the book
                "bids": _levels(b.get("bids", []), True, max_levels),  # best (highest) first
                "asks": _levels(b.get("asks", []), False, max_levels),  # best (lowest) first
            })
    return books


async def snapshot(asset: str, out_dir: Path, days_ahead: int = 1, max_levels: int = 50,
                   gamma_url: str = "https://gamma-api.polymarket.com", clob_url: str = CLOB_URL,
                   now: datetime | None = None, client: httpx.AsyncClient | None = None) -> Path | None:
    """Save one snapshot of every open strike's books. Returns the file, or None if nothing is open."""
    now = now or datetime.now(timezone.utc)
    own = client is None
    client = client or httpx.AsyncClient(timeout=30)
    try:
        markets = await open_markets(client, gamma_url, asset, days_ahead, now)
        if not markets:
            log.warning("no open %s strike markets found", asset)
            return None
        tokens = [t for m in markets for t in (m["yes_token"], m["no_token"])]
        books = await fetch_books(client, tokens, clob_url, max_levels)
    finally:
        if own:
            await client.aclose()

    path = out_dir / now.strftime("%Y-%m-%d") / f"{now.strftime('%H%M%S')}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"taken_at": now.timestamp(), "asset": asset, "markets": markets, "books": books},
                  f, separators=(",", ":"))
    log.info("snapshot: %d markets, %d books -> %s", len(markets), len(books), path)
    return path


def read_snapshot(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
