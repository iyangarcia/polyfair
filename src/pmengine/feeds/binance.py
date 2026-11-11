"""External price oracle: Binance best bid/ask stream."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable

import websockets

from ..config import VolConfig
from ..pricing import EwmaVol, SeasonalVol

log = logging.getLogger(__name__)


class Oracle:
    """Latest mid price and volatility estimate per symbol."""

    def __init__(self, vol_cfg: VolConfig | None = None) -> None:
        self._cfg = vol_cfg or VolConfig()
        self._price: dict[str, tuple[float, float]] = {}  # symbol -> (mid, local ts)
        self._vol: dict[str, EwmaVol] = {}
        self._seasonal: dict[str, SeasonalVol] = {}

    def update(self, symbol: str, mid: float, ts: float) -> None:
        self._price[symbol] = (mid, ts)
        if self._cfg.model == "seasonal":
            sv = self._seasonal.get(symbol)
            if sv is None:
                sv = self._seasonal[symbol] = SeasonalVol(self._cfg.seasonal_days, self._cfg.seasonal_min_days)
            sv.update(mid, ts)
        est = self._vol.get(symbol)
        if est is None:
            c = self._cfg
            est = self._vol[symbol] = EwmaVol(
                prior_vol=c.prior.get(symbol, c.default_prior),
                halflife_s=c.halflife_s,
                warmup_s=c.warmup_s,
                realized_weight=c.realized_weight,
            )
        est.update(mid, ts)

    def price(self, symbol: str) -> tuple[float, float] | None:
        return self._price.get(symbol)

    def vol(self, symbol: str) -> float:
        est = self._vol.get(symbol)
        return est.value if est else self._cfg.prior.get(symbol, self._cfg.default_prior)

    def vol_to(self, symbol: str, t0: float, t1: float) -> float:
        """Vol to use for a contract alive over [t0, t1]: seasonal when known, else the flat estimate."""
        sv = self._seasonal.get(symbol)
        s = sv.sigma(t0, t1) if sv is not None else None
        return s if s is not None else self.vol(symbol)


class BinanceFeed:
    def __init__(self, base_url: str, symbols: list[str], oracle: Oracle, on_update: Callable[[str], None]) -> None:
        streams = "/".join(f"{s.lower()}@bookTicker" for s in symbols)
        self.url = f"{base_url}/stream?streams={streams}"
        self.oracle = oracle
        self.on_update = on_update

    async def run(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.url, close_timeout=1) as ws:
                    log.info("binance ws connected")
                    backoff = 1.0
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        d = json.loads(raw).get("data") or {}
                        if "b" not in d:
                            continue
                        mid = (float(d["b"]) + float(d["a"])) / 2.0
                        self.oracle.update(d["s"], mid, time.time())
                        self.on_update(d["s"])
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("binance ws dropped: %r; reconnecting in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
