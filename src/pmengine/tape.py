"""The trade tape: real taker trades of one market, queried by time.

Each Data API trade row is one taker order (every transaction carries exactly
one row), so `side` is the aggressor's side and `price` is what the taker paid
or received. Two kinds of question are answered, and they never mix:

  * before t: what was the market price, and what did takers recently pay?
    Strategies decide on this only.
  * after t: how much could a taker order have filled? A BUY at limit L is
    proven fillable only by later taker BUYs on the same token at <= L, up to
    `participation` of their size (we compete with those takers for the asks).
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Trade:
    ts: float
    token_id: str
    outcome_index: int  # 0 = YES, 1 = NO
    side: str  # taker side, BUY or SELL
    price: float
    size: float

    @property
    def yes_price(self) -> float:
        """The trade's price in YES terms: a NO at q is a YES at 1 - q."""
        return self.price if self.outcome_index == 0 else 1.0 - self.price


class Tape:
    def __init__(self, trades: list[Trade]) -> None:
        self.trades = sorted(trades, key=lambda t: t.ts)
        self.ts = [t.ts for t in self.trades]
        self._by_token: dict[tuple[str, str], list[Trade]] = {}
        for t in self.trades:
            self._by_token.setdefault((t.token_id, t.side), []).append(t)
        self._by_token_ts = {k: [t.ts for t in v] for k, v in self._by_token.items()}

    # ---- before t: what a strategy may know --------------------------------

    def last_yes_price(self, t: float, lookback_s: float) -> float | None:
        """YES price of the most recent trade strictly before t, within lookback."""
        i = bisect_left(self.ts, t) - 1
        if i < 0 or t - self.ts[i] > lookback_s:
            return None
        return self.trades[i].yes_price

    def last_paid(self, token_id: str, side: str, t: float, lookback_s: float) -> float | None:
        """Price of the latest taker `side` trade on this token strictly before t."""
        ts = self._by_token_ts.get((token_id, side))
        if not ts:
            return None
        i = bisect_left(ts, t) - 1
        if i < 0 or t - ts[i] > lookback_s:
            return None
        return self._by_token[(token_id, side)][i].price

    # ---- after t: what an order could have filled ----------------------------

    def fill(self, token_id: str, side: str, shares: float, limit: float,
             t_from: float, t_to: float, participation: float) -> tuple[float, float]:
        """(shares filled, notional) for a taker order live during (t_from, t_to]."""
        ts = self._by_token_ts.get((token_id, side))
        if not ts or shares <= 0:
            return 0.0, 0.0
        rows = self._by_token[(token_id, side)][bisect_right(ts, t_from):bisect_right(ts, t_to)]
        buy = side == "BUY"
        ok = [r for r in rows if (r.price <= limit + 1e-9 if buy else r.price >= limit - 1e-9)]
        ok.sort(key=lambda r: r.price if buy else -r.price)  # best prices first, like sweeping a book
        filled = notional = 0.0
        for r in ok:
            take = min(r.size * participation, shares - filled)
            filled += take
            notional += take * r.price
            if filled >= shares - 1e-9:
                break
        return filled, notional
