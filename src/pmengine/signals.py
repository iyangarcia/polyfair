"""Turns a fair value and an order book into marketable orders."""

from __future__ import annotations

from .config import StrategyConfig
from .models import MarketMeta, Signal
from .orderbook import OrderBook
from .pricing import kelly_binary, taker_fee


def buy_signal(
    market: str,
    token_id: str,
    outcome: str,
    fair: float,
    book: OrderBook,
    meta: MarketMeta,
    cfg: StrategyConfig,
    min_edge: float,
    equity: float,
) -> Signal | None:
    """Lift every ask level that still clears min_edge after fees.

    EV per share = fair - price - fee, the CLOB version of P_true * quota - 1.
    Size is fractional Kelly on the worst accepted price, capped by the depth
    actually resting at acceptable prices.
    """
    limit = None
    depth = 0.0
    for price, size in book.asks_sorted():
        if fair - price - taker_fee(price, meta.fee_rate, meta.fee_exponent) < min_edge:
            break
        limit = price
        depth += size
    if limit is None:
        return None
    touch = book.best_ask
    edge = fair - touch - taker_fee(touch, meta.fee_rate, meta.fee_exponent)
    if edge > cfg.max_edge:
        return None  # too good to be true: wrong strike/expiry/base rate or a broken feed
    cost = limit + taker_fee(limit, meta.fee_rate, meta.fee_exponent)
    stake = equity * cfg.kelly_fraction * kelly_binary(fair, cost)
    shares = min(depth, stake / cost)
    if shares <= 0:
        return None
    return Signal(market, token_id, outcome, "BUY", fair, touch, limit, shares, edge,
                  meta.fee_rate, meta.fee_exponent, meta.min_size)


def sell_signal(
    market: str,
    token_id: str,
    outcome: str,
    fair: float,
    book: OrderBook,
    meta: MarketMeta,
    cfg: StrategyConfig,
    held: float,
) -> Signal | None:
    """Exit a held position into bids that pay more than fair value."""
    if held <= 0:
        return None
    limit = None
    depth = 0.0
    for price, size in book.bids_sorted():
        if price - taker_fee(price, meta.fee_rate, meta.fee_exponent) - fair < cfg.exit_edge:
            break
        limit = price
        depth += size
    if limit is None:
        return None
    touch = book.best_bid
    edge = touch - taker_fee(touch, meta.fee_rate, meta.fee_exponent) - fair
    return Signal(market, token_id, outcome, "SELL", fair, touch, limit, min(depth, held), edge,
                  meta.fee_rate, meta.fee_exponent, meta.min_size)


def generate(
    market: str,
    fair_yes: float,
    book_yes: OrderBook,
    book_no: OrderBook,
    meta: MarketMeta,
    cfg: StrategyConfig,
    min_edge: float,
    equity: float,
    held_yes: float,
    held_no: float,
) -> list[Signal]:
    out: list[Signal] = []
    legs = (
        (meta.yes_token, "YES", fair_yes, book_yes, held_yes, held_no),
        (meta.no_token, "NO", 1.0 - fair_yes, book_no, held_no, held_yes),
    )
    for token, outcome, fair, book, held, held_other in legs:
        if not book.ready:
            continue
        s = sell_signal(market, token, outcome, fair, book, meta, cfg, held)
        if s:
            out.append(s)
        # Never build both legs: holding YES and NO together is a locked loss of fees.
        if held_other <= 0:
            b = buy_signal(market, token, outcome, fair, book, meta, cfg, min_edge, equity)
            if b:
                out.append(b)
    return out
