import json
from datetime import date, datetime, timezone

import httpx
import pytest

from pmengine.backfill import backfill, event_slug, fetch_trades, market_path, read_market
from pmengine.recorder import read_snapshot, snapshot


def raw_market(slug, closed=True, prices=("1", "0"), strike="76,000"):
    return {
        "slug": slug, "question": "q", "conditionId": f"c-{slug}", "clobTokenIds": json.dumps([f"{slug}-Y", f"{slug}-N"]),
        "outcomePrices": json.dumps(list(prices)) if prices else None, "closed": closed, "enableOrderBook": True,
        "endDate": "2026-11-05T16:00:00Z", "startDate": "2026-10-29T16:00:00Z", "groupItemTitle": strike,
    }


def trade(ts, price, tx="0x1"):
    return {"timestamp": ts, "token_id": "T", "outcome_index": 0, "side": "BUY", "price": price,
            "size": 10, "transaction_hash": tx}


def test_event_slug():
    assert event_slug("bitcoin", date(2026, 11, 5)) == "bitcoin-above-on-november-5-2026"


async def test_fetch_trades_follows_cursor_dedupes_and_sorts():
    pages = {
        None: {"data": [trade(30, 0.5, "0x3"), trade(20, 0.4, "0x2")], "pagination": {"has_more": True, "next_cursor": "p2"}},
        "p2": {"data": [trade(20, 0.4, "0x2"), trade(10, 0.3)], "pagination": {"has_more": False, "next_cursor": None}},
    }
    seen = []

    def handler(req: httpx.Request):
        seen.append(dict(req.url.params))
        return httpx.Response(200, json=pages[req.url.params.get("cursor")])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        out = await fetch_trades(c, "cond", "https://d")
    assert [t["ts"] for t in out] == [10, 20, 30]  # oldest first, duplicate dropped
    assert seen[0]["limit"] == "1000" and seen[0]["conditionId"] == "cond" and seen[1]["cursor"] == "p2"


async def test_backfill_writes_resolved_skips_existing_and_open(tmp_path):
    events = {
        "bitcoin-above-on-november-5-2026": [raw_market("a"), raw_market("b", closed=False, prices=("0.6", "0.4"))],
        "bitcoin-above-on-november-4-2026": [raw_market("c", prices=("0", "1"))],
    }
    trade_calls = []

    def handler(req: httpx.Request):
        if req.url.path == "/events":
            return httpx.Response(200, json=[{"markets": events[req.url.params["slug"]]}])
        trade_calls.append(req.url.params["conditionId"])
        return httpx.Response(200, json={"data": [trade(1, 0.2)], "pagination": {"has_more": False}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        r = await backfill("bitcoin", 2, tmp_path, "https://g", "https://d", date(2026, 11, 5), c)
        assert (r.fetched, r.unresolved, r.skipped, r.trades) == (2, 1, 0, 2)
        a = read_market(market_path(tmp_path, "a"))
        assert a["yes_won"] is True and a["strike"] == 76000.0 and a["event"] == "bitcoin-above-on-november-5-2026"
        assert read_market(market_path(tmp_path, "c"))["yes_won"] is False
        assert not market_path(tmp_path, "b").exists()

        again = await backfill("bitcoin", 2, tmp_path, "https://g", "https://d", date(2026, 11, 5), c)
    assert (again.fetched, again.skipped) == (0, 2)
    assert sorted(trade_calls) == ["c-a", "c-c"]  # second run downloads nothing


async def test_snapshot_saves_open_books_best_level_first(tmp_path):
    now = datetime(2026, 11, 11, 14, 37, 5, tzinfo=timezone.utc)

    def handler(req: httpx.Request):
        if req.url.path == "/events":
            slug = req.url.params["slug"]
            if slug == "bitcoin-above-on-november-11-2026":
                return httpx.Response(200, json=[{"markets": [raw_market("x", closed=False, prices=None),
                                                              raw_market("gone", closed=True)]}])
            return httpx.Response(200, json=[])
        tokens = [t["token_id"] for t in json.loads(req.content)]
        return httpx.Response(200, json=[{
            "asset_id": t, "timestamp": "1794396152997",
            "bids": [{"price": "0.40", "size": "5"}, {"price": "0.42", "size": "1"}, {"price": "0.41", "size": "2"}],
            "asks": [{"price": "0.47", "size": "3"}, {"price": "0.45", "size": "4"}],
        } for t in tokens])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        path = await snapshot("bitcoin", tmp_path, days_ahead=1, max_levels=2, gamma_url="https://g",
                              clob_url="https://c", now=now, client=c)
    assert path == tmp_path / "2026-11-11" / "143705.json.gz"
    snap = read_snapshot(path)
    assert [m["slug"] for m in snap["markets"]] == ["x"]  # closed market left out
    assert {b["asset_id"] for b in snap["books"]} == {"x-Y", "x-N"}
    book = snap["books"][0]
    assert book["bids"] == [[0.42, 1.0], [0.41, 2.0]] and book["asks"] == [[0.45, 4.0], [0.47, 3.0]]
    assert book["ts"] == pytest.approx(1794396152.997)


async def test_snapshot_returns_none_when_nothing_open(tmp_path):
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=[]))
    async with httpx.AsyncClient(transport=transport) as c:
        assert await snapshot("bitcoin", tmp_path, gamma_url="https://g", client=c) is None
