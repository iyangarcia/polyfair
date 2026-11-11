"""Pure maths: digital/one-touch probabilities, fees, Kelly, realised vol."""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timezone

SECONDS_PER_YEAR = 365.0 * 24 * 3600


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_above(spot: float, strike: float, sigma: float, tau: float) -> float:
    """P(S_T > K) for driftless geometric Brownian motion.

    sigma is annualised volatility, tau is time to expiry in years.
    """
    if tau <= 0 or sigma <= 0:
        return 1.0 if spot > strike else 0.0
    s = sigma * math.sqrt(tau)
    return norm_cdf((math.log(spot / strike) - 0.5 * s * s) / s)


def prob_touch(spot: float, barrier: float, sigma: float, tau: float) -> float:
    """P(price touches the barrier before T), from the reflection principle.

    Works for a barrier above spot (up-touch) or below spot (down-touch).
    log S has drift mu = -sigma^2/2, which makes exp(2*mu*b/sigma^2) = S/K.
    """
    up = barrier >= spot
    if (up and spot >= barrier) or (not up and spot <= barrier):
        return 1.0
    if tau <= 0 or sigma <= 0:
        return 0.0
    s = sigma * math.sqrt(tau)
    b = math.log(barrier / spot)
    mu_t = -0.5 * s * s
    if up:
        p = norm_cdf((-b + mu_t) / s) + (spot / barrier) * norm_cdf((-b - mu_t) / s)
    else:
        p = norm_cdf((b - mu_t) / s) + (spot / barrier) * norm_cdf((b + mu_t) / s)
    return min(1.0, max(0.0, p))


def taker_fee(price: float, rate: float, exponent: float = 1.0) -> float:
    """Polymarket taker fee in USDC per share: rate * (p * (1 - p)) ** exponent."""
    if rate <= 0 or not 0.0 < price < 1.0:
        return 0.0
    return rate * (price * (1.0 - price)) ** exponent


def kelly_binary(p_true: float, cost: float) -> float:
    """Full-Kelly bankroll fraction for a contract paying 1 that costs `cost`.

    Same formula as f* = (p*q - 1) / (q - 1) with decimal quota q = 1 / cost.
    """
    if not 0.0 < cost < 1.0:
        return 0.0
    return max(0.0, (p_true - cost) / (1.0 - cost))


class EwmaVol:
    """Annualised volatility: an EWMA of realised variance blended with a prior.

    Realised vol over a few minutes is a noisy guide to vol over days, so the
    estimate is  sqrt(w * realised_var + (1 - w) * prior_var)  where w ramps
    from 0 to `realized_weight` over the warm-up period.
    """

    def __init__(
        self,
        prior_vol: float,
        halflife_s: float = 900.0,
        sample_s: float = 1.0,
        warmup_s: float = 300.0,
        realized_weight: float = 0.5,
    ) -> None:
        self.prior_vol = prior_vol
        self.halflife_s = halflife_s
        self.sample_s = sample_s
        self.warmup_s = warmup_s
        self.realized_weight = realized_weight
        self._last_price: float | None = None
        self._last_ts = 0.0
        self._first_ts = 0.0
        self._var_rate: float | None = None  # variance of log returns per second

    def update(self, price: float, ts: float) -> None:
        if self._last_price is None:
            self._last_price, self._last_ts, self._first_ts = price, ts, ts
            return
        dt = ts - self._last_ts
        if dt < self.sample_s:
            return
        r = math.log(price / self._last_price)
        inst = r * r / dt
        if self._var_rate is None:
            self._var_rate = inst
        else:
            alpha = 1.0 - math.exp(-dt * math.log(2.0) / self.halflife_s)
            self._var_rate += alpha * (inst - self._var_rate)
        self._last_price, self._last_ts = price, ts

    @property
    def realized(self) -> float | None:
        if self._var_rate is None:
            return None
        return math.sqrt(self._var_rate * SECONDS_PER_YEAR)

    @property
    def value(self) -> float:
        if self._var_rate is None:
            return self.prior_vol
        ramp = min(1.0, (self._last_ts - self._first_ts) / self.warmup_s) if self.warmup_s > 0 else 1.0
        w = self.realized_weight * ramp
        var = w * self._var_rate * SECONDS_PER_YEAR + (1.0 - w) * self.prior_vol**2
        return math.sqrt(var)


def vol_bucket(ts: float) -> int:
    """0-23: weekday UTC hour; 24-47: weekend UTC hour."""
    d = datetime.fromtimestamp(ts, timezone.utc)
    return (24 if d.weekday() >= 5 else 0) + d.hour


class SeasonalVol:
    """Volatility that depends on the time of day and the day of the week.

    BTC is roughly twice as volatile on weekdays between 12:00 and 16:00 UTC
    (US data releases, US equity open) as overnight, and much calmer at
    weekends. A flat vol misprices any contract whose remaining life covers
    those hours unevenly. This keeps the mean squared 1-minute log return of
    each (weekday/weekend, UTC hour) bucket over a trailing window, and
    integrates it over the exact span to expiry.

    Feed it prices in time order; it only ever uses minutes that have closed.
    """

    def __init__(self, window_days: float = 30.0, min_days: float = 7.0, min_per_bucket: int = 120) -> None:
        self.window_s = window_days * 86400.0
        self.min_s = min_days * 86400.0
        self.min_per_bucket = min_per_bucket
        self._q: deque[tuple[float, int, float]] = deque()  # (minute start ts, bucket, r^2)
        self._sum = [0.0] * 48
        self._n = [0] * 48
        self._cur_min: int | None = None
        self._cur_px = 0.0
        self._closed: tuple[int, float] | None = None  # (minute, close price)
        self._first_ts: float | None = None

    def update(self, price: float, ts: float) -> None:
        m = int(ts // 60)
        if self._cur_min is None:
            self._cur_min, self._cur_px, self._first_ts = m, price, ts
            return
        if m <= self._cur_min:
            self._cur_px = price
            return
        # minute self._cur_min has closed
        if self._closed is not None and self._cur_min - self._closed[0] == 1:
            r2 = math.log(self._cur_px / self._closed[1]) ** 2
            start = self._cur_min * 60.0
            b = vol_bucket(start)
            self._q.append((start, b, r2))
            self._sum[b] += r2
            self._n[b] += 1
            while self._q and start - self._q[0][0] > self.window_s:
                _, ob, or2 = self._q.popleft()
                self._sum[ob] -= or2
                self._n[ob] -= 1
        self._closed = (self._cur_min, self._cur_px)
        self._cur_min, self._cur_px = m, price

    @property
    def ready(self) -> bool:
        return (self._first_ts is not None and self._closed is not None
                and self._closed[0] * 60.0 - self._first_ts >= self.min_s)

    def var_per_s(self, bucket: int) -> float | None:
        if self._n[bucket] < self.min_per_bucket:
            return None
        return max(self._sum[bucket], 0.0) / self._n[bucket] / 60.0

    def sigma(self, t0: float, t1: float) -> float | None:
        """Annualised vol implied by the seasonal profile over [t0, t1], or None if not known yet."""
        if not self.ready or t1 <= t0:
            return None
        total, t = 0.0, t0
        while t < t1:
            nxt = min(t1, (math.floor(t / 3600.0) + 1) * 3600.0)
            v = self.var_per_s(vol_bucket(t))
            if v is None:
                return None
            total += v * (nxt - t)
            t = nxt
        return math.sqrt(total / ((t1 - t0) / SECONDS_PER_YEAR))
