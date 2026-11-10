"""Cash, positions and PnL, rebuilt from the fill log."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .models import Fill


@dataclass
class Position:
    market: str = ""
    outcome: str = ""
    shares: float = 0.0
    cost: float = 0.0  # total cost basis, fees included

    @property
    def avg(self) -> float:
        return self.cost / self.shares if self.shares > 0 else 0.0


class Portfolio:
    def __init__(self, bankroll: float) -> None:
        self.cash = bankroll
        self.positions: dict[str, Position] = {}
        self.realized = 0.0
        self.fees = 0.0

    def apply(self, f: Fill) -> None:
        pos = self.positions.setdefault(f.token_id, Position(f.market, f.outcome))
        self.fees += f.fee
        if f.side == "BUY":
            spent = f.shares * f.price + f.fee
            self.cash -= spent
            pos.shares += f.shares
            pos.cost += spent
            return
        # SELL or SETTLE
        shares = min(f.shares, pos.shares)
        basis = pos.avg * shares
        proceeds = shares * f.price - f.fee
        self.cash += proceeds
        self.realized += proceeds - basis
        pos.shares -= shares
        pos.cost -= basis
        if pos.shares < 1e-9:
            del self.positions[f.token_id]

    def replay(self, fills: Iterable[Fill]) -> None:
        for f in fills:
            self.apply(f)

    def held(self, token_id: str) -> float:
        p = self.positions.get(token_id)
        return p.shares if p else 0.0

    def exposure(self, tokens: Iterable[str] | None = None) -> float:
        """Cost basis at risk, for the given tokens or the whole book."""
        if tokens is None:
            return sum(p.cost for p in self.positions.values())
        return sum(self.positions[t].cost for t in tokens if t in self.positions)

    def equity(self, mark: Callable[[str], float | None]) -> float:
        """Cash plus positions marked at the bid (cost basis if no bid is known)."""
        total = self.cash
        for token, p in self.positions.items():
            px = mark(token)
            total += p.shares * px if px is not None else p.cost
        return total
