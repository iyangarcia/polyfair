"""Polymarket CLOB market channel: streams L2 books into local OrderBooks."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable

import websockets

from ..orderbook import OrderBook

log = logging.getLogger(__name__)


def apply_message(books: dict[str, OrderBook], msg: dict | list) -> set[str]:
    """Apply one decoded WebSocket frame. Returns the asset ids that changed."""
    touched: set[str] = set()
    for ev in msg if isinstance(msg, list) else [msg]:
        if not isinstance(ev, dict):
            continue
        ts = float(ev.get("timestamp") or 0) / 1000.0
        if "bids" in ev and "asks" in ev:  # full snapshot
            book = books.get(ev.get("asset_id", ""))
            if book is not None:
                book.apply_snapshot(ev["bids"], ev["asks"], ts)
                touched.add(ev["asset_id"])
            continue
        # Level updates. Current schema: price_changes[] each with its own
        # asset_id. Legacy schema: one asset_id with changes[].
        changes = ev.get("price_changes")
        if changes is None and "changes" in ev:
            changes = [{**c, "asset_id": ev.get("asset_id")} for c in ev["changes"]]
        for c in changes or []:
            book = books.get(c.get("asset_id", ""))
            if book is None or not book.ready:
                continue
            book.apply_change(c["side"], float(c["price"]), float(c["size"]), ts)
            touched.add(c["asset_id"])
    return touched


class PolymarketFeed:
    def __init__(self, url: str, asset_ids: list[str], on_update: Callable[[set[str]], None]) -> None:
        self.url = url
        self.asset_ids = asset_ids
        self.on_update = on_update
        self.books: dict[str, OrderBook] = {a: OrderBook() for a in asset_ids}
        self.last_msg_ts = 0.0  # local receive time of the last frame
        self.reconnects = 0

    def silence(self, now: float) -> float:
        return now - self.last_msg_ts

    async def _ping(self, ws) -> None:
        while True:
            await asyncio.sleep(10)
            await ws.send("PING")

    async def run(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.url, ping_interval=None, max_size=None, close_timeout=1) as ws:
                    await ws.send(json.dumps({"assets_ids": self.asset_ids, "type": "market"}))
                    log.info("polymarket ws connected, %d tokens", len(self.asset_ids))
                    backoff = 1.0
                    pinger = asyncio.create_task(self._ping(ws))
                    try:
                        while True:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30)
                            self.last_msg_ts = time.time()
                            if not raw or raw[0] not in "[{":
                                continue  # PONG / keepalive text
                            touched = apply_message(self.books, json.loads(raw))
                            if touched:
                                self.on_update(touched)
                    finally:
                        pinger.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # network errors: reconnect
                log.warning("polymarket ws dropped: %r; reconnecting in %.0fs", e, backoff)
            # Books are untrustworthy until the next snapshot arrives.
            for b in self.books.values():
                b.clear()
            self.reconnects += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
