# pmengine

Event-driven fair-value trading engine for the Polymarket CLOB.

It keeps a local L2 order book per outcome token from Polymarket's WebSocket,
reprices each market from an external model on every tick, and sends
fill-and-kill orders when the book is mispriced by more than fees plus a
required edge. Paper trading is the default; live trading is behind two latches.

```
Binance bookTicker ws ──► Oracle (spot + EWMA vol) ─┐
                                                    ├─► Strategy ─► fair P(YES)
Polymarket market ws ──► OrderBook per token ───────┤
                                                    ▼
                          signals (edge after fees, fractional Kelly)
                                                    ▼
                          RiskManager (caps, cooldown, kill switch)
                                                    ▼
                          Executor: Paper (simulated) │ Live (Polymarket SDK)
                                                    ▼
                          SQLite: orders, fills, equity curve
```

## Strategies

| Strategy | Config | Fair value |
| - | - | - |
| `crypto_drift` | `kind: above` / `below` | `P(S_T > K)` under driftless GBM off the live Binance mid |
| `crypto_drift` | `kind: touch` | One-touch barrier probability (reflection principle) |
| `base_rate` | `base_rate: p` | Your fixed estimate of P(YES); the engine buys whichever side the crowd misprices against it |

Entry: lift every ask level where `fair - price - fee >= min_edge`.
Size: `kelly_fraction * (fair - cost) / (1 - cost)` of equity, capped by resting depth and risk limits.
Exit: sell a held token into bids where `bid - fee - fair >= exit_edge`; otherwise hold to resolution.

Fees are Polymarket's taker fee, `rate * p * (1 - p)` per share, with the rate read
per market from the Gamma API (0.07 on crypto markets at the time of writing, which is
1.75c per share at 50c). That is the main hurdle for the drift strategy.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"        # Linux/macOS: .venv/bin/pip
cp config.example.yaml config.yaml
.venv/Scripts/pmengine scan "bitcoin above"  # find current market slugs
.venv/Scripts/pmengine run                   # paper mode
.venv/Scripts/pmengine report                # positions, PnL, order stats
.venv/Scripts/python -m pytest
```

Docker: `docker compose up --build` (mounts `config.yaml` and `data/`).

Stop trading without stopping the process: create the kill file (`data/KILL`).

## Backtest

```bash
.venv/Scripts/pmengine backtest --days 30
```

Replays resolved daily "Bitcoin above $X" markets (`--asset ethereum` also works)
through the same signal, risk and portfolio code as the engine, and settles each
position at the real outcome. It sends no orders and needs no keys. Data is
Polymarket's 1-minute price history and Binance 1-minute candles, cached under
`data/cache/`. Trades go to `data/backtest_trades.csv`.

Read the result with these in mind:

- Polymarket publishes past prices, not past order books. The quote is rebuilt as
  `price +/- half_spread`, and the report runs three spreads so you can see how
  much the answer depends on that guess.
- One-minute data cannot show the sub-second latency edge. What is tested is
  whether the model's disagreement with the market pays.
- The Brier score line compares the model's probabilities with the market's
  against what actually happened. If the market's is lower, the model has no
  forecasting edge and any profit is luck or a volatility bet.
- `base_rate` markets are not backtested: their fair value is your own input.

## Collecting data

Two kinds of history, collected two ways:

**Trades (`pmengine backfill`).** Polymarket keeps every trade of a resolved
market, so this needs nothing running in the background. It downloads the full
trade history of each resolved daily strike market into `data/trades/`, one
`<slug>.json.gz` per market with its metadata (strike, expiry, outcome) and its
trades oldest first. Re-run it whenever you like; it skips markets it already has.

```bash
.venv/Scripts/pmengine backfill --days 120    # ~2 min, ~1,250 markets
```

The trade `side` is stored as the Data API returns it. Whether it is the
aggressor's side has not been verified yet.

**Order books (`pmengine record` + GitHub Actions).** Polymarket only serves the
current book, so book history exists only if it is saved as it happens.
`pmengine record` takes one snapshot of every open strike's book (50 levels a
side, best first) and exits, writing `data/books/YYYY-MM-DD/HHMMSS.json.gz`.
`.github/workflows/record-books.yml` runs it every 30 minutes and commits each
file to the `data` branch, so no machine of yours has to stay on. To read the
history locally:

```bash
git fetch origin data && git worktree add ../polyfair-data data
```

Snapshots are about 13 KB, about 20 MB a month. A private repo on the Free plan gets
2,000 Actions minutes a month and this uses about 1,440; on a public repo minutes
are free and the schedule can be tightened to `*/15`. GitHub can run scheduled
jobs late, and it pauses them after 60 days without repo activity.

## Live trading

Live mode uses the official unified SDK (`polymarket-client`). The older
`py-clob-client` packages are deprecated by Polymarket.

1. `pip install -e ".[live]"`
2. Put `POLYMARKET_PRIVATE_KEY` and `PM_ENABLE_LIVE=1` in `.env` (see `.env.example`)
3. `pmengine run --live`

**The live adapter has never been run against the real exchange.** It was written
from the SDK's signatures and docs. Before trusting it, place one minimum-size
order and check that the recorded fill (shares, price) matches your account.
The adapter halts the engine on anything it cannot account for exactly: a
submission error, a `delayed` status, or fill amounts that imply an impossible price.

In live mode the engine still tracks positions from its own fill log, not from
the exchange. Do not trade the same markets by hand from the same wallet.

## Known limits

- Paper fills ignore queue position, other takers racing for the same quote, and
  the exchange's matching delay on marketable orders. Paper PnL is an upper bound.
- The crypto model is driftless lognormal with one volatility number. It has no
  skew, no jumps and no intraday seasonality. `vol.prior` matters a lot for tails.
- `touch` markets: the engine does not know whether the barrier was already hit
  before it started. Settlement uses a specific Binance candle, so the engine
  stands down `min_time_to_expiry_s` before expiry.
- One evaluation runs at a time, so a slow order delays the other markets.
- A position only settles in the books while its market is still in `config.yaml`.
- Only binary markets. Outcome 0 is treated as YES ("Up" on up/down markets).
- Check that Polymarket is available to you where you live before going live.

## Layout

```
src/pmengine/
  pricing.py      digital / one-touch probabilities, fees, Kelly, EWMA vol
  orderbook.py    L2 book, sweep simulation
  feeds/          Polymarket and Binance WebSocket clients
  gamma.py        market metadata and discovery
  backfill.py     trade history of resolved markets
  recorder.py     one-shot order book snapshots
  strategies.py   fair-value models
  signals.py      fair value + book -> orders
  risk.py         limits and kill switch
  execution.py    paper simulator, live SDK adapter
  portfolio.py    cash, positions, PnL
  store.py        SQLite
  engine.py       event loop
  cli.py          run / scan / report
```
