import math
import random

import pytest

from pmengine.pricing import EwmaVol, kelly_binary, norm_cdf, prob_above, prob_touch, taker_fee


def test_prob_above_basics():
    assert prob_above(100, 100, 0.5, 1e-9) == pytest.approx(0.5, abs=1e-3)
    assert prob_above(110, 100, 0.5, 0) == 1.0
    assert prob_above(90, 100, 0.5, 0) == 0.0
    # monotone in spot
    ps = [prob_above(s, 100, 0.5, 0.01) for s in (90, 95, 100, 105, 110)]
    assert ps == sorted(ps)
    # known value: d2 = (ln(1.05) - 0.5*0.04) / 0.2
    assert prob_above(105, 100, 0.2, 1.0) == pytest.approx(norm_cdf((math.log(1.05) - 0.02) / 0.2))


def test_prob_touch_bounds_and_ordering():
    assert prob_touch(100, 100, 0.5, 0.1) == 1.0
    assert prob_touch(101, 100, 0.5, 0.1) == 1.0 or prob_touch(101, 100, 0.5, 0.1) <= 1.0
    up = prob_touch(100, 110, 0.5, 0.1)
    assert prob_above(100, 110, 0.5, 0.1) < up < 1.0  # touching is easier than finishing above
    down = prob_touch(100, 90, 0.5, 0.1)
    assert (1 - prob_above(100, 90, 0.5, 0.1)) < down < 1.0
    assert prob_touch(100, 110, 0.5, 0) == 0.0


def test_prob_touch_matches_monte_carlo():
    rng = random.Random(7)
    sigma, tau, spot, barrier, steps, n = 0.6, 0.05, 100.0, 108.0, 400, 4000
    dt = tau / steps
    hits = 0
    for _ in range(n):
        x = 0.0
        for _ in range(steps):
            x += -0.5 * sigma * sigma * dt + sigma * math.sqrt(dt) * rng.gauss(0, 1)
            if x >= math.log(barrier / spot):
                hits += 1
                break
    # discrete monitoring under-counts touches slightly, so allow a loose band
    assert prob_touch(spot, barrier, sigma, tau) == pytest.approx(hits / n, abs=0.04)


def test_taker_fee_matches_published_table():
    # docs: crypto rate 0.07, 100 shares at $0.50 -> $1.75, at $0.10 -> $0.63
    assert 100 * taker_fee(0.50, 0.07) == pytest.approx(1.75)
    assert 100 * taker_fee(0.10, 0.07) == pytest.approx(0.63)
    assert taker_fee(0.5, 0.0) == 0.0


def test_kelly_equals_quota_formula():
    p, cost = 0.6, 0.5
    quota = 1 / cost
    assert kelly_binary(p, cost) == pytest.approx((p * quota - 1) / (quota - 1))
    assert kelly_binary(0.4, 0.5) == 0.0


def test_ewma_vol_recovers_true_vol():
    rng = random.Random(1)
    true_vol = 0.8
    per_s = true_vol / math.sqrt(365 * 24 * 3600)
    est = EwmaVol(prior_vol=0.3, halflife_s=600, warmup_s=60, realized_weight=1.0)
    price, ts = 100.0, 0.0
    assert est.value == 0.3
    for _ in range(5000):
        ts += 1.0
        price *= math.exp(per_s * rng.gauss(0, 1))
        est.update(price, ts)
    assert est.value == pytest.approx(true_vol, rel=0.15)
