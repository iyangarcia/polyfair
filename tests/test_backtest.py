from datetime import datetime, timezone

import pytest

from pmengine.backtest import BtConfig, bootstrap_mean, daily_pnl, market_from_record, simulate
from pmengine.config import Settings

DAY = 86400.0
T0 = datetime(2026, 10, 1, 16, tzinfo=timezone.utc).timestamp()


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def rec(slug, end, yes_won, trades, strike=100.0):
    return {"slug": slug, "condition_id": "c", "yes_token": f"{slug}-Y", "no_token": f"{slug}-N",
            "strike": strike, "start_ts": iso(end - 3 * DAY), "end_ts": end, "yes_won": yes_won,
            "fee_rate": 0.07, "trades": trades}


def buys(slug, t0, t1, price, every=60.0, outcome=0, size=1000.0):
    tok = f"{slug}-Y" if outcome == 0 else f"{slug}-N"
    out, t = [], t0
    while t < t1:
        out.append({"ts": t, "token_id": tok, "outcome_index": outcome, "side": "BUY", "price": price, "size": size})
        t += every
    return out


def flat_klines(start, end, price=100.0):
    return [(t, price) for t in range(int(start), int(end), 60)]


def drift_cfg(**kw):
    return BtConfig(strategy="crypto_drift", warmup_days=0, window_days=1.0, **kw)


def test_drift_buys_underpriced_yes_from_the_tape_and_settles():
    end = T0 + DAY
    m = market_from_record(rec("m", end, True, buys("m", end - 2 * DAY, end - 3600, 0.40)))
    k = flat_klines(end - 3 * DAY, end)
    r = simulate([m], Settings(), drift_cfg(), k)
    entries = [t for t in r.trades if t["side"] == "BUY"]
    assert entries and all(t["outcome"] == "YES" and t["price"] == pytest.approx(0.40) for t in entries)
    shares = sum(t["shares"] for t in entries)
    cost = sum(t["shares"] * t["price"] + t["fee"] for t in entries)
    assert cost <= Settings().risk.max_position_usd + 1e-6
    assert r.trades[-1]["side"] == "SETTLE"
    assert r.pnl == pytest.approx(shares - cost, abs=0.05)  # trade rows are rounded
    assert r.pnl_by_day[datetime.fromtimestamp(end, timezone.utc).strftime("%Y-%m-%d")] == pytest.approx(r.pnl, abs=0.05)

    lost = simulate([market_from_record(rec("m", end, False, buys("m", end - 2 * DAY, end - 3600, 0.40)))],
                    Settings(), drift_cfg(), k)
    assert lost.pnl == pytest.approx(-cost, abs=0.05)


def test_fills_need_later_trades_a_quote_alone_is_not_enough():
    end = T0 + DAY
    first_tick = end - DAY
    # one taker buy just before the first decision: a quote, but nothing trades afterwards
    m = market_from_record(rec("m", end, True, buys("m", first_tick - 60, first_tick - 59, 0.40)))
    r = simulate([m], Settings(), drift_cfg(), flat_klines(end - 3 * DAY, end))
    assert r.orders > 0 and r.filled == 0 and r.trades == []


def test_fill_price_and_size_come_from_the_later_trades():
    end = T0 + DAY
    # asks proven at 0.38 by later takers, quote seen at 0.40
    trades = buys("m", end - 2 * DAY, end - 3600, 0.40, every=600) + \
        buys("m", end - 2 * DAY + 30, end - 3600, 0.38, every=600, size=10)
    m = market_from_record(rec("m", end, True, trades))
    r = simulate([m], Settings(), drift_cfg(participation=0.5), flat_klines(end - 3 * DAY, end))
    entries = [t for t in r.trades if t["side"] == "BUY"]
    assert entries
    assert all(t["shares"] <= 5.0 + 1e-6 or t["price"] > 0.38 for t in entries)  # 50% of a 10-share print


def test_no_trading_without_candles_or_with_stale_candles():
    end = T0 + DAY
    m = market_from_record(rec("m", end, True, buys("m", end - 2 * DAY, end - 3600, 0.40)))
    late = flat_klines(end + DAY, end + 2 * DAY)  # every candle closes after expiry
    assert simulate([m], Settings(), drift_cfg(), late).trades == []


def test_calibrated_learns_only_from_resolved_markets():
    old_end = T0
    new_end = T0 + 3 * DAY
    # history: YES at 0.40 kept winning
    old = market_from_record(rec("old", old_end, True, buys("old", old_end - 2.5 * DAY, old_end - 3600, 0.40)))
    new = market_from_record(rec("new", new_end, False, buys("new", new_end - 2 * DAY, new_end - 3600, 0.40)))
    cfg = BtConfig(strategy="calibrated", warmup_days=1, window_days=1.0, shrink_days=3.0)
    r = simulate([old, new], Settings(), cfg)
    entries = [t for t in r.trades if t["side"] == "BUY"]
    assert entries and all(t["market"] == "new" and t["outcome"] == "YES" for t in entries)
    assert r.test_days == ["2026-10-04"]

    alone = simulate([new], Settings(), BtConfig(strategy="calibrated", warmup_days=0, window_days=1.0))
    assert alone.trades == []  # nothing resolved yet: fair value is the market price


def test_daily_pnl_counts_untraded_days_and_bootstrap_brackets_the_mean():
    from pmengine.backtest import BtResult

    r = BtResult(1000.0, pnl_by_day={"d1": 10.0, "d3": -4.0}, test_days=["d1", "d2", "d3"])
    assert daily_pnl(r) == [10.0, 0.0, -4.0]
    lo, hi = bootstrap_mean([1.0, 2.0, 3.0, 4.0])
    assert lo <= 2.5 <= hi
