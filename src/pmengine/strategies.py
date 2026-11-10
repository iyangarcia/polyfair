"""Fair-value models. Each returns the model probability of outcome 0 (YES)."""

from __future__ import annotations

from typing import Protocol

from .config import MarketConfig, OracleConfig
from .feeds.binance import Oracle
from .pricing import SECONDS_PER_YEAR, prob_above, prob_touch

_CLAMP = 0.001


class Strategy(Protocol):
    def fair_yes(self, now: float) -> float | None:
        """Model P(YES), or None when the model must not trade right now."""


class BaseRateStrategy:
    """Base-rate distortion: a fixed prior the crowd is assumed to misprice.

    Typical use is a hyped tail event priced at 6c that historically happens
    2% of the time: base_rate 0.02 makes the engine buy NO while it is cheap.
    """

    def __init__(self, cfg: MarketConfig) -> None:
        self.p = float(cfg.base_rate)  # validated by MarketConfig

    def fair_yes(self, now: float) -> float | None:
        return self.p


class CryptoDriftStrategy:
    """Correlated-market drift: reprice a crypto contract off the live spot.

    Polymarket quotes lag Binance by anything from milliseconds to seconds.
    Fair value is recomputed on every Binance tick and compared with the
    resting Polymarket book.
    """

    def __init__(self, cfg: MarketConfig, expiry_ts: float, oracle: Oracle, ocfg: OracleConfig) -> None:
        self.symbol = cfg.symbol
        self.kind = cfg.kind
        self.strike = float(cfg.strike)
        self.expiry_ts = expiry_ts
        self.oracle = oracle
        self.ocfg = ocfg

    def fair_yes(self, now: float) -> float | None:
        tick = self.oracle.price(self.symbol)
        if tick is None:
            return None
        spot, ts = tick
        if now - ts > self.ocfg.max_age_s:
            return None  # stale oracle: we would be the slow side
        remaining = self.expiry_ts - now
        if remaining < self.ocfg.min_time_to_expiry_s:
            return None  # settlement uses a specific candle; do not guess it
        tau = remaining / SECONDS_PER_YEAR
        sigma = self.oracle.vol(self.symbol)
        if self.kind == "above":
            p = prob_above(spot, self.strike, sigma, tau)
        elif self.kind == "below":
            p = 1.0 - prob_above(spot, self.strike, sigma, tau)
        else:  # touch
            p = prob_touch(spot, self.strike, sigma, tau)
        return min(1.0 - _CLAMP, max(_CLAMP, p))
