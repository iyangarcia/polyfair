"""Local L2 order book for one outcome token, kept in sync from the WebSocket."""

from __future__ import annotations


def _key(price: float) -> float:
    return round(price, 6)


class OrderBook:
    __slots__ = ("bids", "asks", "ts", "ready")

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.ts = 0.0  # exchange timestamp of the last update, seconds
        self.ready = False  # True once a full snapshot has been applied

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.ready = False

    def apply_snapshot(self, bids: list[dict], asks: list[dict], ts: float) -> None:
        self.bids = {_key(float(l["price"])): float(l["size"]) for l in bids if float(l["size"]) > 0}
        self.asks = {_key(float(l["price"])): float(l["size"]) for l in asks if float(l["size"]) > 0}
        self.ts = ts
        self.ready = True

    def apply_change(self, side: str, price: float, size: float, ts: float) -> None:
        """Set the resting size at one price level. side BUY = bid, SELL = ask."""
        levels = self.bids if side.upper() == "BUY" else self.asks
        if size <= 0:
            levels.pop(_key(price), None)
        else:
            levels[_key(price)] = size
        self.ts = ts

    @property
    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    def asks_sorted(self) -> list[tuple[float, float]]:
        return sorted(self.asks.items())

    def bids_sorted(self) -> list[tuple[float, float]]:
        return sorted(self.bids.items(), reverse=True)

    def sweep(self, side: str, shares: float, limit: float) -> tuple[float, float]:
        """Simulate a marketable order. Returns (shares filled, notional).

        A BUY lifts asks priced <= limit; a SELL hits bids priced >= limit.
        The book itself is not modified.
        """
        buy = side.upper() == "BUY"
        levels = self.asks_sorted() if buy else self.bids_sorted()
        filled = notional = 0.0
        for price, size in levels:
            if (buy and price > limit + 1e-9) or (not buy and price < limit - 1e-9):
                break
            take = min(size, shares - filled)
            filled += take
            notional += take * price
            if filled >= shares - 1e-9:
                break
        return filled, notional
