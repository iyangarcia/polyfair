"""YAML configuration, validated with pydantic."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RiskConfig(_Model):
    max_order_usd: float = 25.0
    max_position_usd: float = 100.0  # per market, cost basis
    max_total_exposure_usd: float = 500.0
    max_session_loss_usd: float = 50.0  # kill switch
    max_orders_per_min: int = 20
    cooldown_s: float = 2.0  # per token, after any order attempt
    max_feed_silence_s: float = 30.0
    kill_file: str = "data/KILL"  # create this file to halt trading


class StrategyConfig(_Model):
    min_edge: float = 0.02  # per share, net of fees, required to enter
    max_edge: float = 0.25  # larger edges are treated as a model/config error
    exit_edge: float = 0.0  # sell when bid - fee - fair >= exit_edge
    kelly_fraction: float = Field(0.25, gt=0, le=1)


class VolConfig(_Model):
    # ewma: one number from a prior blended with recent realised vol.
    # seasonal: vol by UTC hour and weekday/weekend from trailing 1-minute returns,
    #   integrated over the exact time to expiry; falls back to ewma until it has
    #   seasonal_min_days of history.
    model: Literal["ewma", "seasonal"] = "ewma"
    prior: dict[str, float] = {"BTCUSDT": 0.50, "ETHUSDT": 0.65}  # annualised
    default_prior: float = 0.60
    halflife_s: float = 900.0
    warmup_s: float = 300.0
    realized_weight: float = Field(0.5, ge=0, le=1)
    seasonal_days: float = 30.0
    seasonal_min_days: float = 7.0


class OracleConfig(_Model):
    # Binance blocks US addresses (GitHub's runners among them) on these; the
    # market-data mirrors wss://data-stream.binance.vision and
    # https://data-api.binance.vision serve the same data to everyone.
    binance_ws: str = "wss://stream.binance.com:9443"
    binance_rest: str = "https://api.binance.com"
    max_age_s: float = 2.0  # no trading on a price older than this
    min_time_to_expiry_s: float = 300.0  # stand down close to settlement
    max_time_to_expiry_s: float | None = None  # only trade this close to expiry (None: always)


class PaperConfig(_Model):
    latency_ms: float = 150.0  # simulated order round trip


class MarketConfig(_Model):
    name: str
    slug: str  # Polymarket market slug (see `pmengine scan`)
    strategy: Literal["crypto_drift", "base_rate"]
    min_edge: float | None = None  # overrides strategy.min_edge

    # crypto_drift
    symbol: str = "BTCUSDT"
    kind: Literal["above", "below", "touch"] | None = None
    strike: float | None = None
    expiry: datetime | None = None  # defaults to the market's end date

    # base_rate
    base_rate: float | None = None  # your estimate of P(outcome 0 / YES)

    @model_validator(mode="after")
    def _check(self) -> "MarketConfig":
        if self.strategy == "crypto_drift":
            if self.kind is None or self.strike is None:
                raise ValueError(f"{self.name}: crypto_drift needs kind and strike")
        elif self.base_rate is None or not 0.0 < self.base_rate < 1.0:
            raise ValueError(f"{self.name}: base_rate must be between 0 and 1")
        if self.expiry is not None and self.expiry.tzinfo is None:
            self.expiry = self.expiry.replace(tzinfo=timezone.utc)
        return self


class DiscoverConfig(_Model):
    """Trade every open strike of the daily "<asset> above $K" ladder, found at startup."""

    asset: Literal["bitcoin", "ethereum"] = "bitcoin"
    symbol: str = "BTCUSDT"
    days_ahead: int = 0  # 0: only the ladder expiring today (UTC date)
    min_edge: float | None = None


class Settings(_Model):
    bankroll: float = 1000.0  # capital allocated to the engine, USDC
    db_path: str = "data/engine.db"
    status_interval_s: float = 30.0
    clob_ws: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    gamma_url: str = "https://gamma-api.polymarket.com"
    risk: RiskConfig = RiskConfig()
    strategy: StrategyConfig = StrategyConfig()
    vol: VolConfig = VolConfig()
    oracle: OracleConfig = OracleConfig()
    paper: PaperConfig = PaperConfig()
    markets: list[MarketConfig] = []
    discover: list[DiscoverConfig] = []

    @model_validator(mode="after")
    def _unique(self) -> "Settings":
        names = [m.name for m in self.markets]
        if len(set(names)) != len(names):
            raise ValueError("market names must be unique")
        return self


def load_settings(path: str | Path) -> Settings:
    with open(path, encoding="utf-8") as f:
        return Settings.model_validate(yaml.safe_load(f))
