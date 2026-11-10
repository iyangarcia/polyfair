import time

import pytest

from pmengine.config import MarketConfig, OracleConfig, RiskConfig, Settings, StrategyConfig
from pmengine.execution import PaperExecutor
from pmengine.feeds.binance import Oracle
from pmengine.feeds.polymarket import apply_message
from pmengine.gamma import parse_meta
from pmengine.models import Fill, MarketMeta
from pmengine.orderbook import OrderBook
from pmengine.portfolio import Portfolio
from pmengine.risk import RiskManager
from pmengine.signals import generate
from pmengine.store import Store
from pmengine.strategies import CryptoDriftStrategy

META = MarketMeta("slug", "q", "0xc", "YES_T", "NO_T", 5.0, 0.07, 1.0, None, False, None)
CFG = StrategyConfig()


def book(bids, asks) -> OrderBook:
    b = OrderBook()
    b.apply_snapshot([{"price": str(p), "size": str(s)} for p, s in bids],
                     [{"price": str(p), "size": str(s)} for p, s in asks], 1.0)
    return b


# ---- order book and feed ------------------------------------------------

def test_book_snapshot_changes_and_sweep():
    b = book([(0.40, 100), (0.39, 50)], [(0.42, 30), (0.43, 70)])
    assert (b.best_bid, b.best_ask) == (0.40, 0.42)
    b.apply_change("SELL", 0.42, 0, 2.0)  # level pulled
    assert b.best_ask == 0.43
    b.apply_change("BUY", 0.41, 10, 3.0)
    assert b.best_bid == 0.41
    filled, notional = b.sweep("SELL", 40, 0.40)
    assert filled == 40 and notional == pytest.approx(10 * 0.41 + 30 * 0.40)
    assert b.sweep("BUY", 100, 0.42) == (0, 0)  # nothing at or below the limit


def test_apply_message_current_and_legacy_schemas():
    books = {"A": OrderBook(), "B": OrderBook()}
    snap = [{"event_type": "book", "asset_id": "A", "timestamp": "1000",
             "bids": [{"price": "0.5", "size": "10"}], "asks": [{"price": "0.6", "size": "10"}]}]
    assert apply_message(books, snap) == {"A"}
    change = {"event_type": "price_change", "timestamp": "2000", "price_changes": [
        {"asset_id": "A", "price": "0.55", "size": "5", "side": "BUY"},
        {"asset_id": "B", "price": "0.45", "size": "5", "side": "SELL"},  # B has no snapshot yet
        {"asset_id": "UNKNOWN", "price": "0.1", "size": "5", "side": "BUY"},
    ]}
    assert apply_message(books, change) == {"A"}
    assert books["A"].best_bid == 0.55 and not books["B"].asks
    legacy = {"event_type": "price_change", "asset_id": "A", "timestamp": "3000",
              "changes": [{"price": "0.55", "size": "0", "side": "BUY"}]}
    apply_message(books, legacy)
    assert books["A"].best_bid == 0.5


def test_parse_meta_reads_fees_and_tokens():
    raw = {"slug": "s", "question": "Q?", "conditionId": "0x1", "clobTokenIds": '["1", "2"]',
           "orderMinSize": 5, "feesEnabled": True, "feeSchedule": {"rate": 0.07, "exponent": 1},
           "endDate": "2026-11-10T17:00:00Z", "closed": False, "outcomePrices": '["0.4", "0.6"]'}
    m = parse_meta(raw)
    assert (m.yes_token, m.no_token, m.fee_rate) == ("1", "2", 0.07)
    assert m.end_ts == 1794330000.0
    assert parse_meta({**raw, "feesEnabled": False}).fee_rate == 0.0


# ---- signals ------------------------------------------------------------

def test_buy_signal_takes_only_levels_with_edge():
    yes = book([(0.40, 100)], [(0.42, 30), (0.45, 70), (0.50, 500)])
    no = book([(0.50, 100)], [(0.58, 100)])
    sigs = generate("m", 0.50, yes, no, META, CFG, 0.02, 1000.0, 0.0, 0.0)
    assert len(sigs) == 1
    s = sigs[0]
    # at 0.45: fee 0.0173, edge 0.0327 ok; at 0.50: edge negative
    assert (s.side, s.outcome, s.limit, s.touch) == ("BUY", "YES", 0.45, 0.42)
    assert s.edge == pytest.approx(0.50 - 0.42 - 0.07 * 0.42 * 0.58)
    cost = 0.45 + 0.07 * 0.45 * 0.55
    assert s.shares == pytest.approx(1000 * 0.25 * (0.50 - cost) / (1 - cost) / cost)


def test_no_signal_when_fees_eat_the_edge():
    yes = book([(0.46, 100)], [(0.47, 100)])
    no = book([(0.52, 100)], [(0.54, 100)])
    assert generate("m", 0.50, yes, no, META, CFG, 0.02, 1000.0, 0.0, 0.0) == []


def test_no_leg_buys_when_yes_is_overpriced():
    yes = book([(0.10, 100)], [(0.11, 100)])
    no = book([(0.89, 100)], [(0.90, 100)])
    sigs = generate("m", 0.02, yes, no, META, CFG, 0.02, 1000.0, 0.0, 0.0)
    assert [(s.side, s.outcome) for s in sigs] == [("BUY", "NO")]


def test_exit_when_bid_exceeds_fair_and_no_double_leg():
    yes = book([(0.60, 40)], [(0.61, 100)])
    no = book([(0.20, 100)], [(0.21, 100)])  # NO looks cheap, but we hold YES
    sigs = generate("m", 0.50, yes, no, META, CFG, 0.02, 1000.0, held_yes=25.0, held_no=0.0)
    assert [(s.side, s.outcome, s.shares) for s in sigs] == [("SELL", "YES", 25.0)]


def test_absurd_edge_is_refused():
    yes = book([(0.10, 100)], [(0.11, 100)])
    no = book([(0.88, 100)], [(0.90, 100)])
    assert generate("m", 0.90, yes, no, META, CFG, 0.02, 1000.0, 0.0, 0.0) == []


# ---- strategy -----------------------------------------------------------

def test_crypto_strategy_guards_stale_oracle_and_expiry():
    now = time.time()
    oracle = Oracle()
    cfg = MarketConfig(name="m", slug="s", strategy="crypto_drift", kind="above", strike=100)
    strat = CryptoDriftStrategy(cfg, now + 3600, oracle, OracleConfig())
    assert strat.fair_yes(now) is None  # no price yet
    oracle.update("BTCUSDT", 100.0, now)
    assert strat.fair_yes(now) == pytest.approx(0.5, abs=0.01)
    assert strat.fair_yes(now + 5) is None  # price is 5s old
    near = CryptoDriftStrategy(cfg, now + 60, oracle, OracleConfig())
    assert near.fair_yes(now) is None  # too close to settlement


def test_config_validation():
    with pytest.raises(ValueError):
        MarketConfig(name="m", slug="s", strategy="crypto_drift")
    with pytest.raises(ValueError):
        MarketConfig(name="m", slug="s", strategy="base_rate", base_rate=1.5)
    m = MarketConfig(name="m", slug="s", strategy="base_rate", base_rate=0.02)
    with pytest.raises(ValueError):
        Settings(markets=[m, m])


# ---- portfolio, risk, execution, store ----------------------------------

def fill(side, shares, price, fee=0.0) -> Fill:
    return Fill(0.0, "paper", "m", "YES_T", "YES", side, shares, price, fee)


def test_portfolio_round_trip_and_settlement():
    pf = Portfolio(1000)
    pf.apply(fill("BUY", 100, 0.40, fee=1.0))
    assert pf.cash == pytest.approx(959) and pf.exposure() == pytest.approx(41)
    assert pf.equity(lambda t: 0.45) == pytest.approx(959 + 45)
    pf.apply(fill("SELL", 50, 0.50, fee=0.5))
    assert pf.realized == pytest.approx(50 * 0.50 - 0.5 - 20.5)
    pf.apply(fill("SETTLE", 50, 1.0))
    assert pf.positions == {} and pf.cash == pytest.approx(959 + 24.5 + 50)
    assert pf.realized == pytest.approx(pf.cash - 1000)


def make_signal(shares=1000.0, side="BUY"):
    yes = book([(0.40, 5000)], [(0.42, 5000)])
    no = book([(0.50, 100)], [(0.60, 100)])
    held = 10.0 if side == "SELL" else 0.0
    fair = 0.30 if side == "SELL" else 0.50
    return next(s for s in generate("m", fair, yes, no, META, CFG, 0.02, 1e6, held, 0.0) if s.side == side)


def test_risk_caps_size_cooldown_and_halt(tmp_path):
    cfg = RiskConfig(max_order_usd=25, kill_file=str(tmp_path / "KILL"))
    risk = RiskManager(cfg, start_equity=1000)
    pf = Portfolio(1000)
    sig = make_signal()
    d = risk.review(sig, META, pf, now=100.0)
    unit = 0.42 + 0.07 * 0.42 * 0.58
    assert d.ok and d.signal.shares == pytest.approx(25 / unit, abs=0.01)
    assert d.signal.shares * unit <= 25

    risk.record_attempt(sig.token_id, 100.0)
    assert risk.review(sig, META, pf, now=101.0).reason == "cooldown"
    assert risk.review(sig, META, pf, now=103.0).ok

    pf.apply(fill("BUY", 236, 0.42))  # 99.12 of the 100 per-market cap used
    assert not risk.review(sig, META, pf, now=110.0).ok  # remaining budget < min size

    risk.check_global(equity=960)
    assert not risk.halted
    risk.check_global(equity=949)
    assert "session loss" in risk.halted
    assert risk.review(sig, META, Portfolio(1000), now=200.0).reason.startswith("halted")


def test_kill_file_halts(tmp_path):
    kill = tmp_path / "KILL"
    risk = RiskManager(RiskConfig(kill_file=str(kill)), 1000)
    risk.check_global(1000)
    assert not risk.halted
    kill.write_text("stop")
    risk.check_global(1000)
    assert "kill file" in risk.halted


async def test_paper_executor_fills_at_book_and_misses_pulled_quotes():
    ex = PaperExecutor(latency_ms=0)
    sig = RiskManager(RiskConfig(), 1000).review(make_signal(), META, Portfolio(1000), 0.0).signal
    b = book([(0.40, 100)], [(0.42, 20), (0.44, 100)])
    res = await ex.execute(sig, b)
    assert res.status == "filled" and res.fill.shares == 20 and res.fill.price == pytest.approx(0.42)
    assert res.fill.fee == pytest.approx(20 * 0.07 * 0.42 * 0.58)
    b.apply_change("SELL", 0.42, 0, 2.0)  # quote pulled before we arrive
    assert (await ex.execute(sig, b)).status == "unfilled"


def test_store_replay_rebuilds_portfolio():
    store = Store(":memory:")
    store.record_fill(fill("BUY", 100, 0.40, 1.0))
    store.record_fill(fill("SELL", 100, 0.50, 1.0))
    store.record_order("paper", make_signal(), "filled", "", 3.2)
    pf = Portfolio(1000)
    pf.replay(store.fills("paper"))
    assert pf.cash == pytest.approx(1008) and pf.realized == pytest.approx(8)
    assert store.fills("live") == []
    assert store.order_stats("paper")[0][:2] == ("filled", 1)
