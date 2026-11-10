"""Plain data types shared across the engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketMeta:
    """Exchange-side facts about a market, resolved from the Gamma API."""

    slug: str
    question: str
    condition_id: str
    yes_token: str  # outcome index 0 ("Yes" / "Up")
    no_token: str  # outcome index 1 ("No" / "Down")
    min_size: float  # minimum order size in shares
    fee_rate: float
    fee_exponent: float
    end_ts: float | None
    closed: bool
    outcome_prices: tuple[float, float] | None


@dataclass(frozen=True)
class Signal:
    """A marketable order the strategy wants to send."""

    market: str
    token_id: str
    outcome: str  # "YES" | "NO"
    side: str  # "BUY" | "SELL"
    fair: float  # model probability for this token
    touch: float  # best ask (BUY) or best bid (SELL) when the signal fired
    limit: float  # worst price we accept
    shares: float
    edge: float  # per share at the touch, net of fees
    fee_rate: float
    fee_exponent: float
    min_size: float


@dataclass(frozen=True)
class Fill:
    ts: float
    mode: str  # "paper" | "live"
    market: str
    token_id: str
    outcome: str
    side: str  # "BUY" | "SELL" | "SETTLE"
    shares: float
    price: float  # average
    fee: float  # total USDC
    order_id: str = ""


@dataclass(frozen=True)
class ExecResult:
    fill: Fill | None
    status: str  # "filled" | "unfilled" | "rejected" | "error"
    detail: str = ""
    fatal: bool = False  # True -> engine must halt and a human must reconcile
