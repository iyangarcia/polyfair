"""Market discovery and metadata from Polymarket's public Gamma API."""

from __future__ import annotations

import json
from datetime import datetime

import httpx

from .models import MarketMeta


def _ts(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def parse_meta(raw: dict) -> MarketMeta:
    tokens = json.loads(raw["clobTokenIds"])
    if len(tokens) != 2:
        raise ValueError(f"{raw.get('slug')}: only binary markets are supported")
    fees = raw.get("feeSchedule") or {}
    fees_on = bool(raw.get("feesEnabled")) and bool(fees)
    prices = None
    if raw.get("outcomePrices"):
        p = [float(x) for x in json.loads(raw["outcomePrices"])]
        prices = (p[0], p[1])
    return MarketMeta(
        slug=raw["slug"],
        question=raw.get("question", ""),
        condition_id=raw.get("conditionId", ""),
        yes_token=tokens[0],
        no_token=tokens[1],
        min_size=float(raw.get("orderMinSize") or 5),
        fee_rate=float(fees.get("rate", 0)) if fees_on else 0.0,
        fee_exponent=float(fees.get("exponent", 1)) if fees_on else 1.0,
        end_ts=_ts(raw.get("endDate")),
        closed=bool(raw.get("closed")),
        outcome_prices=prices,
    )


async def fetch_meta(client: httpx.AsyncClient, base_url: str, slug: str) -> MarketMeta:
    # closed=true is needed as well, otherwise resolved markets are not returned
    for closed in ("false", "true"):
        r = await client.get(f"{base_url}/markets", params={"slug": slug, "closed": closed})
        r.raise_for_status()
        rows = r.json()
        if rows:
            return parse_meta(rows[0])
    raise LookupError(f"no Polymarket market with slug {slug!r}")


async def scan(base_url: str, query: str, limit: int = 25, pages: int = 10) -> list[dict]:
    """Active markets whose question contains every word of `query`, by 24h volume."""
    words = query.lower().split()
    out: list[dict] = []
    async with httpx.AsyncClient(timeout=20) as client:
        for page in range(pages):
            r = await client.get(
                f"{base_url}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": 100,
                    "offset": page * 100,
                    "order": "volume24hr",
                    "ascending": "false",
                },
            )
            r.raise_for_status()
            rows = r.json()
            for raw in rows:
                q = (raw.get("question") or "").lower()
                if all(w in q for w in words) and raw.get("enableOrderBook"):
                    out.append(raw)
                    if len(out) >= limit:
                        return out
            if len(rows) < 100:
                break
    return out
