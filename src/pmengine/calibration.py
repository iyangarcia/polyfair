"""Empirical calibration of market prices, learned from resolved markets.

For a cell (time to expiry, YES price) the model stores how far outcomes
landed from the price: residual = yes_won - price. Its estimate of the
market's bias in that cell is the mean residual, with two protections:

  * Each expiry day counts once. The ~11 strikes of one day settle on the
    same BTC print, so they are one observation, not eleven.
  * The mean is shrunk toward zero by n_days / (n_days + shrink_days), so a
    cell with little history barely moves the fair value away from the price.

`symmetric` pools every sample with its mirror image (1 - p, -residual):
an overpriced YES at 0.8 is an underpriced NO at 0.2. Real trades showed the
late-window bias on YES and NO contracts alike, which is what makes pooling
reasonable; switch it off to test that assumption.

The model is only ever fed markets that resolved before the moment it is
asked about, so a backtest built on it has no lookahead.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict

TTE_EDGES_H = [0.0, 1.0, 2.0, 4.0, 6.0, 12.0, 24.0, 72.0, float("inf")]
PRICE_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98, 1.0]


def cell(tte_h: float, p: float) -> tuple[int, int] | None:
    if tte_h < 0 or not 0.0 < p < 1.0:
        return None
    return bisect_right(TTE_EDGES_H, tte_h) - 1, min(bisect_right(PRICE_EDGES, p) - 1, len(PRICE_EDGES) - 2)


class Calibration:
    def __init__(self, shrink_days: float = 20.0, symmetric: bool = True) -> None:
        self.shrink_days = shrink_days
        self.symmetric = symmetric
        self._cells: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
        self._cache: dict[tuple[int, int], float] = {}

    def add(self, samples: list[tuple[float, float]], yes_won: bool, day: str) -> None:
        """Samples of one resolved market: (hours to expiry, YES price)."""
        y = 1.0 if yes_won else 0.0
        for tte_h, p in samples:
            points = [(p, y - p)]
            if self.symmetric:
                points.append((1.0 - p, p - y))
            for q, resid in points:
                c = cell(tte_h, q)
                if c is None:
                    continue
                acc = self._cells[c][day]
                acc[0] += resid
                acc[1] += 1.0
                self._cache.pop(c, None)

    def stats(self, tte_h: float, p: float) -> tuple[float, int]:
        """(mean of per-day mean residuals, number of days) for the cell."""
        c = cell(tte_h, p)
        if c is None or c not in self._cells:
            return 0.0, 0
        days = self._cells[c]
        means = [s / n for s, n in days.values() if n > 0]
        return (sum(means) / len(means) if means else 0.0), len(means)

    def bias(self, tte_h: float, p: float) -> float:
        """Shrunk estimate of yes_won - p in this cell."""
        c = cell(tte_h, p)
        if c is None:
            return 0.0
        if c not in self._cache:
            m, n = self.stats(tte_h, p)
            self._cache[c] = m * n / (n + self.shrink_days) if n else 0.0
        return self._cache[c]

    def fair(self, tte_h: float, p: float, clamp: float = 0.001) -> float:
        return min(1.0 - clamp, max(clamp, p + self.bias(tte_h, p)))
