import json
from datetime import datetime, timezone

import httpx
import pytest

from pmengine.config import DiscoverConfig, Settings, load_settings
from pmengine.engine import Engine, discover
from pmengine.execution import PaperExecutor
from pmengine.models import Fill
from pmengine.paper import append_summary, should_run
from pmengine.store import Store

NOW = datetime(2026, 11, 11, 11, 5, tzinfo=timezone.utc)


def raw(slug, strike, closed=False, prices=None):
    return {"slug": slug, "question": "q", "conditionId": "c", "clobTokenIds": json.dumps([f"{slug}-Y", f"{slug}-N"]),
            "outcomePrices": json.dumps(prices) if prices else None, "closed": closed, "enableOrderBook": True,
            "endDate": "2026-11-11T17:00:00Z", "groupItemTitle": strike}


async def test_discover_builds_crypto_drift_configs_for_open_strikes():
    def handler(req: httpx.Request):
        assert req.url.params["slug"] == "bitcoin-above-on-november-11-2026"
        return httpx.Response(200, json=[{"markets": [raw("a-74k", "74,000"), raw("a-76k", "76,000"),
                                                      raw("gone", "70,000", closed=True)]}])

    s = Settings(discover=[DiscoverConfig(asset="bitcoin", symbol="BTCUSDT", min_edge=0.03)])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        cfgs = await discover(c, s, NOW)
    assert [(m.name, m.slug, m.strike, m.kind, m.strategy, m.min_edge) for m in cfgs] == [
        ("a-74k", "a-74k", 74000.0, "above", "crypto_drift", 0.03),
        ("a-76k", "a-76k", 76000.0, "above", "crypto_drift", 0.03),
    ]


async def test_positions_from_an_earlier_day_settle_on_startup(tmp_path):
    store = Store(str(tmp_path / "e.db"))
    store.record_fill(Fill(1.0, "paper", "old-74k", "old-74k-N", "NO", "BUY", 100.0, 0.30, 1.47))
    store.record_fill(Fill(1.0, "paper", "not-a-slug", "x-Y", "YES", "BUY", 10.0, 0.50, 0.0))
    eng = Engine(Settings(), PaperExecutor(0), store)
    eng.portfolio.replay(store.fills("paper"))

    def handler(req: httpx.Request):
        slug = req.url.params["slug"]
        if slug == "old-74k" and req.url.params["closed"] == "true":
            return httpx.Response(200, json=[raw("old-74k", "74,000", closed=True, prices=["0", "1"])])
        return httpx.Response(200, json=[])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        await eng._settle_untracked(c, tracked=set())
    fills = store.fills("paper")
    settle = [f for f in fills if f.side == "SETTLE"]
    assert len(settle) == 1 and settle[0].market == "old-74k" and settle[0].price == 1.0  # NO won
    assert eng.portfolio.held("old-74k-N") == 0
    assert eng.portfolio.realized == pytest.approx(100.0 - (30.0 + 1.47))
    assert eng.portfolio.held("x-Y") == 10.0  # unknown market: left open, not guessed
    store.close()


def test_should_run_from_any_start_in_the_window_before_expiry():
    expiry = datetime(2026, 11, 11, 17, tzinfo=timezone.utc).timestamp()
    at = lambda h, m: datetime(2026, 11, 11, h, m, tzinfo=timezone.utc).timestamp()  # noqa: E731
    assert should_run(expiry, at(11, 5), 1, 6)[0]  # on time: 5h55 before
    assert should_run(expiry, at(14, 30), 1, 6)[0]  # a late start still trades what is left
    assert not should_run(expiry, at(10, 5), 1, 6)[0]  # 6h55 before: too early
    assert not should_run(expiry, at(16, 5), 1, 6)[0]  # under 1h left: not worth a day
    assert not should_run(expiry, at(17, 30), 1, 6)[0]  # after expiry
    summer = datetime(2026, 7, 15, 16, tzinfo=timezone.utc).timestamp()
    assert should_run(summer, datetime(2026, 7, 15, 10, 5, tzinfo=timezone.utc).timestamp(), 1, 6)[0]
    assert not should_run(summer, datetime(2026, 7, 15, 9, 5, tzinfo=timezone.utc).timestamp(), 1, 6)[0]
    assert should_run(None, at(11, 5), 1, 6) == (False, "no open markets to trade today")


def test_summary_row_counts_todays_fills(tmp_path):
    store = Store(str(tmp_path / "e.db"))
    store.record_fill(Fill(10.0, "paper", "m", "m-Y", "YES", "BUY", 10.0, 0.40, 0.17))
    store.record_fill(Fill(50.0, "paper", "m", "m-Y", "YES", "SETTLE", 10.0, 1.0, 0.0))
    row = append_summary(tmp_path / "summary.csv", "2026-11-11", Settings(), store, since=20.0, minutes=355)
    assert row["fills_today"] == 1 and row["buys_today"] == 0
    assert row["realized_total"] == pytest.approx(10.0 - 4.17, abs=0.01)
    append_summary(tmp_path / "summary.csv", "2026-11-12", Settings(), store, since=100.0, minutes=90)
    lines = (tmp_path / "summary.csv").read_text().splitlines()
    assert lines[0].startswith("day,minutes,") and len(lines) == 3
    assert lines[1].startswith("2026-11-11,355,") and lines[2].startswith("2026-11-12,90,")
    store.close()


def test_summary_written_before_the_minutes_column_is_upgraded(tmp_path):
    path = tmp_path / "summary.csv"
    path.write_text("day,fills_today,buys_today,cash,open_cost,realized_total,fees_total\n"
                    "2026-11-11,1,1,989.85,10.15,0.0,0.35\n")
    store = Store(str(tmp_path / "e.db"))
    append_summary(path, "2026-11-12", Settings(), store, since=0.0, minutes=120)
    lines = path.read_text().splitlines()
    assert lines[0] == "day,minutes,fills_today,buys_today,cash,open_cost,realized_total,fees_total"
    assert lines[1] == "2026-11-11,,1,1,989.85,10.15,0.0,0.35"
    assert lines[2].startswith("2026-11-12,120,")
    store.close()


def test_committed_paper_config_is_the_backtested_setup():
    s = load_settings("paper.yaml")
    assert s.vol.model == "seasonal" and s.oracle.max_time_to_expiry_s == 6 * 3600
    assert s.strategy.min_edge == 0.02 and s.discover and not s.markets
    assert "binance.vision" in s.oracle.binance_ws and "binance.vision" in s.oracle.binance_rest
