"""Pre-trade risk checks, position limits and the kill switch."""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, replace

from .config import RiskConfig
from .models import MarketMeta, Signal
from .portfolio import Portfolio
from .pricing import taker_fee

MIN_MARKETABLE_USD = 1.0  # exchange minimum notional for marketable orders


@dataclass(frozen=True)
class Decision:
    signal: Signal | None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.signal is not None


class RiskManager:
    def __init__(self, cfg: RiskConfig, start_equity: float) -> None:
        self.cfg = cfg
        self.start_equity = start_equity
        self.halted = ""  # non-empty = reason trading is stopped
        self._last_attempt: dict[str, float] = {}
        self._attempts: deque[float] = deque()

    def halt(self, reason: str) -> None:
        if not self.halted:
            self.halted = reason

    def check_global(self, equity: float) -> None:
        """Trip the kill switch on a session drawdown or an operator kill file."""
        if os.path.exists(self.cfg.kill_file):
            self.halt(f"kill file {self.cfg.kill_file} present")
        elif self.start_equity - equity >= self.cfg.max_session_loss_usd:
            self.halt(f"session loss {self.start_equity - equity:.2f} >= {self.cfg.max_session_loss_usd:.2f}")

    def review(self, s: Signal, meta: MarketMeta, pf: Portfolio, now: float) -> Decision:
        """Resize a signal to fit the limits, or reject it."""
        if self.halted:
            return Decision(None, f"halted: {self.halted}")
        if now - self._last_attempt.get(s.token_id, -math.inf) <self.cfg.cooldown_s:
            return Decision(None, "cooldown")
        while self._attempts and now - self._attempts[0] > 60.0:
            self._attempts.popleft()
        if len(self._attempts) >= self.cfg.max_orders_per_min:
            return Decision(None, "order rate limit")

        shares = s.shares
        if s.side == "BUY":
            unit = s.limit + taker_fee(s.limit, s.fee_rate, s.fee_exponent)
            budget = min(
                self.cfg.max_order_usd,
                self.cfg.max_position_usd - pf.exposure((meta.yes_token, meta.no_token)),
                self.cfg.max_total_exposure_usd - pf.exposure(),
                pf.cash,
            )
            shares = min(shares, budget / unit)
        shares = math.floor(shares * 100) / 100  # exchange size precision
        if s.side == "BUY" and (shares < s.min_size or shares * s.limit < MIN_MARKETABLE_USD):
            return Decision(None, "below minimum size after limits")
        if shares <= 0:
            return Decision(None, "nothing to trade")
        return Decision(replace(s, shares=shares))

    def record_attempt(self, token_id: str, now: float) -> None:
        self._last_attempt[token_id] = now
        self._attempts.append(now)
