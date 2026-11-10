"""Order execution: a paper simulator and the live Polymarket SDK adapter."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Protocol

from .models import ExecResult, Fill, Signal
from .orderbook import OrderBook
from .pricing import taker_fee

log = logging.getLogger(__name__)


class Executor(Protocol):
    mode: str

    async def start(self) -> None: ...
    async def execute(self, s: Signal, book: OrderBook) -> ExecResult: ...
    async def close(self) -> None: ...


class PaperExecutor:
    """Fill-and-kill against the live local book after a simulated round trip.

    The book is re-read after the latency sleep, so quotes that are pulled
    while the order is "in flight" are missed, as they would be for real.
    Queue position and our own market impact are not modelled.
    """

    mode = "paper"

    def __init__(self, latency_ms: float = 150.0) -> None:
        self.latency_s = latency_ms / 1000.0
        self._n = 0

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def execute(self, s: Signal, book: OrderBook) -> ExecResult:
        if self.latency_s > 0:
            await asyncio.sleep(self.latency_s)
        if not book.ready:
            return ExecResult(None, "unfilled", "book reset while order in flight")
        filled, notional = book.sweep(s.side, s.shares, s.limit)
        if filled <= 0:
            return ExecResult(None, "unfilled", "no liquidity at limit on arrival")
        price = notional / filled
        fee = filled * taker_fee(price, s.fee_rate, s.fee_exponent)
        self._n += 1
        fill = Fill(time.time(), self.mode, s.market, s.token_id, s.outcome, s.side,
                    filled, price, fee, f"paper-{int(time.time())}-{self._n}")
        return ExecResult(fill, "filled")


class LiveExecutor:
    """Signs and routes fill-and-kill orders through the official Polymarket SDK.

    Requires `pip install pmengine[live]`, POLYMARKET_PRIVATE_KEY and
    PM_ENABLE_LIVE=1. Anything this adapter cannot account for exactly is
    reported as fatal so the engine halts instead of trading on a guess.
    """

    mode = "live"

    def __init__(self) -> None:
        self._client = None

    async def start(self) -> None:
        if os.environ.get("PM_ENABLE_LIVE") != "1":
            raise RuntimeError("live trading is disabled: set PM_ENABLE_LIVE=1 to enable it")
        key = os.environ.get("POLYMARKET_PRIVATE_KEY")
        if not key:
            raise RuntimeError("POLYMARKET_PRIVATE_KEY is not set")
        try:
            from polymarket import AsyncSecureClient
        except ImportError as e:
            raise RuntimeError("live trading needs the SDK: pip install 'pmengine[live]'") from e
        self._client = await AsyncSecureClient.create(
            private_key=key, wallet=os.environ.get("POLYMARKET_WALLET_ADDRESS") or None
        )
        bal = await self._client.get_balance_allowance(asset_type="COLLATERAL")
        log.info("live executor authenticated; collateral balance (raw units): %s", bal.balance)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()

    async def execute(self, s: Signal, book: OrderBook) -> ExecResult:
        try:
            if s.side == "BUY":
                resp = await self._client.place_market_order(
                    asset_id=s.token_id, side="BUY", amount=f"{s.shares * s.limit:.2f}",
                    max_price=str(s.limit), order_type="FAK",
                )
            else:
                resp = await self._client.place_market_order(
                    asset_id=s.token_id, side="SELL", shares=f"{s.shares:.2f}",
                    min_price=str(s.limit), order_type="FAK",
                )
        except Exception as e:
            # The request may or may not have reached the matching engine.
            return ExecResult(None, "error", f"order submission raised {e!r}", fatal=True)

        if not resp.ok:
            unfilled = resp.code in ("fak_not_filled", "fok_not_filled", "unmatched")
            return ExecResult(None, "unfilled" if unfilled else "rejected", f"{resp.code}: {resp.message}")
        if resp.status != "matched":
            return ExecResult(None, "error", f"order {resp.order_id} status {resp.status}: fill unknown", fatal=True)

        making, taking = float(resp.making_amount), float(resp.taking_amount)
        shares, usdc = (taking, making) if s.side == "BUY" else (making, taking)
        price = usdc / shares if shares > 0 else -1.0
        bad_price = price > s.limit + 0.011 if s.side == "BUY" else price < s.limit - 0.011
        if not 0.0 < price < 1.0 or bad_price:
            return ExecResult(None, "error",
                              f"order {resp.order_id}: implausible fill making={making} taking={taking}", fatal=True)
        fee = shares * taker_fee(price, s.fee_rate, s.fee_exponent)  # estimate; charged at match time
        fill = Fill(time.time(), self.mode, s.market, s.token_id, s.outcome, s.side,
                    shares, price, fee, str(resp.order_id))
        return ExecResult(fill, "filled")
