import math
from datetime import datetime, timezone

import pytest

from pmengine.calibration import Calibration
from pmengine.config import MarketConfig, OracleConfig, VolConfig
from pmengine.feeds.binance import Oracle
from pmengine.pricing import SeasonalVol, vol_bucket
from pmengine.strategies import CryptoDriftStrategy
from pmengine.tape import Tape, Trade

MON = datetime(2026, 10, 5, tzinfo=timezone.utc).timestamp()  # a Monday, 00:00 UTC


# ---- tape ------------------------------------------------------------------------

def tape():
    return Tape([
        Trade(100, "Y", 0, "BUY", 0.40, 10),
        Trade(200, "N", 1, "BUY", 0.55, 10),  # YES-equivalent 0.45
        Trade(300, "Y", 0, "BUY", 0.42, 10),
        Trade(310, "Y", 0, "BUY", 0.39, 4),
        Trade(320, "Y", 0, "SELL", 0.38, 6),
    ])


def test_tape_quotes_use_only_trades_strictly_before_t():
    t = tape()
    assert t.last_yes_price(200, 1000) == pytest.approx(0.40)  # the trade at 200 is not yet known
    assert t.last_yes_price(201, 1000) == pytest.approx(0.45)  # NO at 0.55 = YES at 0.45
    assert t.last_yes_price(290, 50) is None  # latest trade (200) is older than the lookback
    assert t.last_paid("Y", "BUY", 305, 1000) == pytest.approx(0.42)
    assert t.last_paid("Y", "SELL", 305, 1000) is None


def test_tape_fill_takes_later_takers_at_or_better_than_limit_best_first():
    t = tape()
    filled, notional = t.fill("Y", "BUY", 100, 0.41, 250, 400, participation=0.5)
    assert filled == pytest.approx(2.0)  # only the 0.39 print is <= 0.41; half of its 4 shares
    assert notional == pytest.approx(2.0 * 0.39)
    filled, _ = t.fill("Y", "BUY", 100, 0.45, 250, 400, participation=1.0)
    assert filled == pytest.approx(14.0)
    assert t.fill("Y", "BUY", 100, 0.45, 310, 400, 1.0)[0] == 0.0  # window starts after both prints
    assert t.fill("Y", "SELL", 100, 0.38, 250, 400, 1.0)[0] == pytest.approx(6.0)
    assert t.fill("Y", "SELL", 100, 0.39, 250, 400, 1.0)[0] == 0.0


# ---- calibration -----------------------------------------------------------------

def test_calibration_shrinks_and_counts_each_day_once():
    c = Calibration(shrink_days=1.0, symmetric=False)
    assert c.fair(3.0, 0.30) == pytest.approx(0.30)  # no history: the price itself
    c.add([(3.0, 0.30)] * 50, yes_won=True, day="d1")  # 50 samples, one day: residual +0.7
    assert c.stats(3.0, 0.30) == (pytest.approx(0.7), 1)
    assert c.bias(3.0, 0.30) == pytest.approx(0.35)  # 0.7 * 1 / (1 + 1)
    c.add([(3.0, 0.30)], yes_won=False, day="d2")  # one sample, residual -0.3, weighs as much as d1
    assert c.stats(3.0, 0.30) == (pytest.approx(0.2), 2)
    assert c.bias(30.0, 0.30) == 0.0  # other time-to-expiry cell is untouched


def test_calibration_symmetric_pools_the_mirror_image():
    c = Calibration(shrink_days=0.0, symmetric=True)
    c.add([(3.0, 0.30)], yes_won=True, day="d1")
    assert c.bias(3.0, 0.30) == pytest.approx(0.7)
    assert c.bias(3.0, 0.70) == pytest.approx(-0.7)  # NO at 0.30 won: YES at 0.70 was overpriced


# ---- seasonal vol ----------------------------------------------------------------

def feed(sv, days, loud_hour=14, loud=0.002, quiet=0.0005):
    """Alternating 1-minute moves: `loud` during one weekday UTC hour, `quiet` otherwise."""
    px, sign = 100.0, 1.0
    for i in range(int(days * 1440)):
        ts = MON + i * 60
        d = datetime.fromtimestamp(ts, timezone.utc)
        step = loud if d.hour == loud_hour else quiet
        px *= math.exp(sign * step)
        sign = -sign
        sv.update(px, ts)
    return MON + days * 86400


def test_seasonal_vol_sees_the_loud_hour_and_integrates_it():
    sv = SeasonalVol(window_days=30, min_days=4, min_per_bucket=30)
    end = feed(sv, 5)  # Mon..Fri
    assert sv.ready
    t0 = end  # Saturday 00:00: no weekend history yet
    assert sv.sigma(t0, t0 + 3600) is None
    fri = MON + 4 * 86400
    quiet = sv.sigma(fri + 3 * 3600, fri + 4 * 3600)
    loud = sv.sigma(fri + 14 * 3600, fri + 15 * 3600)
    assert loud == pytest.approx(4 * quiet, rel=0.05)
    both = sv.sigma(fri + 13 * 3600, fri + 15 * 3600)  # one quiet hour + one loud hour
    assert both == pytest.approx(math.sqrt((quiet**2 + loud**2) / 2), rel=0.01)


def test_seasonal_vol_not_ready_before_min_days_and_oracle_falls_back():
    sv = SeasonalVol(min_days=7, min_per_bucket=30)
    feed(sv, 2)
    assert not sv.ready and sv.sigma(MON, MON + 3600) is None

    o = Oracle(VolConfig(model="seasonal", seasonal_min_days=7))
    o.update("BTCUSDT", 100.0, MON)
    assert o.vol_to("BTCUSDT", MON, MON + 3600) == pytest.approx(o.vol("BTCUSDT"))


def test_vol_bucket_splits_weekday_and_weekend():
    assert vol_bucket(MON + 14 * 3600) == 14
    assert vol_bucket(MON + 5 * 86400 + 14 * 3600) == 24 + 14  # Saturday


def test_crypto_drift_respects_max_time_to_expiry():
    o = Oracle()
    o.update("BTCUSDT", 100.0, MON)
    cfg = MarketConfig(name="m", slug="m", strategy="crypto_drift", kind="above", strike=100.0)
    far = CryptoDriftStrategy(cfg, MON + 86400, o, OracleConfig(max_age_s=1e9, max_time_to_expiry_s=6 * 3600))
    assert far.fair_yes(MON) is None
    assert far.fair_yes(MON + 86400 - 3600) is not None
