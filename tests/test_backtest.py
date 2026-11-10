import pytest

from pmengine.backtest import BtMarket, simulate
from pmengine.config import Settings
from pmengine.models import MarketMeta

DAY = 86400.0
T0 = 1_000_000.0


def market(price: float, yes_won: bool) -> BtMarket:
    meta = MarketMeta("m", "q", "0xc", "Y", "N", 5.0, 0.07, 1.0, T0 + DAY, True, None)
    hist = [(T0 + 60.0 * i, price) for i in range(1, 600)]
    return BtMarket("m", meta, 100.0, T0 - DAY, T0 + DAY, yes_won, hist)


def flat_klines(start: float, n: int, price: float = 100.0):
    return [(start + 60.0 * i, price) for i in range(n)]


def test_buys_underpriced_yes_and_settles_at_outcome():
    # spot == strike a day out -> fair ~0.49; market offers YES at 0.41
    r = simulate([market(0.40, True)], flat_klines(T0 - 3600, 1500), Settings(), "BTCUSDT", 0.01, 200)
    buys = [t for t in r.trades if t["side"] == "BUY"]
    assert buys and all(t["outcome"] == "YES" and t["price"] == pytest.approx(0.41) for t in buys)
    shares = sum(t["shares"] for t in buys)
    cost = sum(t["shares"] * t["price"] + t["fee"] for t in buys)
    assert cost <= Settings().risk.max_position_usd + 1e-6
    assert r.trades[-1]["side"] == "SETTLE"
    assert r.pnl == pytest.approx(shares - cost, abs=0.01)  # trade rows are rounded
    assert r.pnl_by_market["m"] == pytest.approx(r.pnl)

    lost = simulate([market(0.40, False)], flat_klines(T0 - 3600, 1500), Settings(), "BTCUSDT", 0.01, 200)
    assert lost.pnl == pytest.approx(-cost, abs=0.01)


def test_no_trade_when_price_is_fair_or_spread_too_wide():
    k = flat_klines(T0 - 3600, 1500)
    assert simulate([market(0.49, True)], k, Settings(), "BTCUSDT", 0.01, 200).trades == []
    assert simulate([market(0.40, True)], k, Settings(), "BTCUSDT", 0.08, 200).trades == []


def test_no_lookahead_candles_after_the_quote_are_not_used():
    late = flat_klines(T0 + 2 * DAY, 100)  # every candle closes after the market expired
    assert simulate([market(0.40, True)], late, Settings(), "BTCUSDT", 0.01, 200).trades == []


def test_placeholder_prices_right_after_listing_are_ignored():
    m = market(0.40, True)
    m.start_ts = T0  # listed at the first quote: the whole 10h of history is inside the skip window
    assert simulate([m], flat_klines(T0 - 3600, 1500), Settings(), "BTCUSDT", 0.01, 200).trades == []
