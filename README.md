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
| `calibrated` (backtest only) | `--strategy calibrated` | Market price plus the bias that resolved markets showed at that price and time to expiry |

Volatility for `crypto_drift` comes from `vol.model`:

- `ewma` (default): one number, a prior blended with recent realised vol.
- `seasonal`: vol by UTC hour and weekday/weekend from the trailing 30 days of
  1-minute returns, integrated over the exact time left to expiry. At startup
  the engine downloads those 30 days of candles so it is ready at once. See the
  research notes below for why this matters.

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
.venv/Scripts/pmengine backfill --days 120     # real trades, once
.venv/Scripts/pmengine backtest --strategy crypto_drift --vol-model seasonal --window-days 0.25
.venv/Scripts/pmengine backtest --strategy calibrated
```

A walk-forward backtest on the real taker trades from `pmengine backfill`, run
through the same signal, risk and portfolio code as the engine. It sends no
orders and needs no keys. Every `--step-s` (300 s) of each market's last
`--window-days`:

1. **Quotes come from trades before the decision.** A token's ask is what the
   latest taker paid for it, its bid what the latest taker seller got (or 1 -
   the other token's ask). No trade within `--lookback-s`: no quote, no trade.
2. **Fills come from trades after it.** An order fills only against real taker
   trades on the same side, after `--latency-s`, within `--fill-window-s`, at
   or better than its limit, and only `--participation` (50%) of their size.
   Each Data API row is one taker order, so its `side` is the aggressor's.
3. Positions settle at the real outcome. The first `--warmup-days` (30) expiry
   days are never traded; they only train `calibrated` and warm the vol model.

The report gives PnL, fill rate, a 95% interval for mean daily PnL from
resampling whole days, a Sharpe ratio, both halves of the test period, and the
Brier score of the model against the market. Read it with these in mind:

- The ~11 strikes of a day settle on the same BTC print. They are one bet, so
  the statistics count days, not markets. A few months is few observations.
- If the market's Brier score is lower than the model's, the model has no
  forecasting edge and any profit is luck or a volatility bet.
- A trade print proves liquidity existed, not that you would have won it, and
  1-minute candles cannot test sub-second latency. Paper and live will differ.
- `base_rate` markets are not backtested: their fair value is your own input.

## Research notes (Jul 15 - Nov 10 2026, 115 expiry days, 631k trades)

What was tested, in order. Analysis scripts were run outside the repo; the
numbers come from the backtest and from these data.

1. **Is the market calibrated?** Mostly yes. Across all taker buys, tokens won
   about as often as their price implied, with no significant longshot bias.
   This is why the original flat-vol model lost: it argued with a market that
   was usually right.
2. **Late-window bias.** 1-6 h before expiry, tokens bought at 10-50c won about
   8c per share more than they cost after fees (95% CI +2c to +15c), on YES
   and NO alike. But sampling prices on a fixed 5-minute grid, as a strategy
   would, halves it (+4.9c, not significant): much of it was other takers'
   timing. The `calibrated` strategy, which learns these biases walk-forward,
   lost 45% over 85 test days. Its predicted edge did not repeat.
3. **Flat-vol `crypto_drift` on real fills** made +404%, but without its best
   3 days it lost money. It is a bet on BTC moving more than priced that paid
   in a few volatile stretches.
4. **Why late tails are cheap.** The vol implied by the strike ladder matches
   realised vol a day before expiry (32.7% vs 33.7%) but not 6 h before (35.8%
   vs 44.9%, implied below realised on 79% of days, CI -12 to -6 points). BTC on
   weekdays is about twice as volatile 12:00-16:00 UTC (US data, US equity
   open) as overnight, and these markets settle at noon New York time (16:00
   UTC in summer, 17:00 UTC in winter), right after. The market prices those
   hours as if vol were flat.
5. **Test of that hypothesis**, fixed before running it: `crypto_drift`, seasonal
   vol, last 6 h only, defaults otherwise. +128% over 85 test days, Sharpe 3.5,
   max drawdown $414, model Brier 0.0367 vs market 0.0370 (the only model here
   that beats the market). Every nearby variant was positive (3 h, 12 h, 3 d
   windows, higher min edge, 25% participation with 5 s latency); flat vol with
   the same window made +14%. **But** the daily-PnL interval still includes
   zero (-2.6 to +33.2 a day), nearly all of the profit came in October, and
   without its best 5 days it is flat.

Status: a lead with a mechanism, not a proven edge. The fair test is days the
backtest has not seen. Re-run `backfill` and the same backtest command each
month without changing any setting, and paper-trade it meanwhile (below).

## Daily paper trading (GitHub Actions)

`.github/workflows/paper-trade.yml` paper-trades the backtested setup every
day with no machine of yours on. It never trades live: it runs
`pmengine paper-day`, which only uses the paper executor.

- `paper.yaml` is the backtested setup, unchanged. It finds each day's strike
  ladder by itself (`discover:`), trades only the last 6 hours, and reads Binance
  through its market-data mirrors, which GitHub's US runners can reach.
- The job starts at 10:05 and 11:05 UTC. Only the start 5-6 hours before that
  day's expiry trades (10:05 in summer, 11:05 in winter); the other exits in
  seconds. A run lasts up to about 5h55, inside GitHub's 6-hour job limit.
- State lives in `paper/` on the `data` branch: the fill log `engine.db`,
  `summary.csv` (one row per day), `logs/` and `runs/` markers that stop a day
  running twice. Positions are held to resolution and settle at the start of
  the next day's run.
- About 6 job-hours a day: free on a public repo, far over the Free plan's
  allowance on a private one.

To read the results, check the `data` branch out next to the repo (once) and
open `paper/summary.csv`, or the run's summary on the Actions tab:

```bash
git worktree add ../polyfair-data data    # later: git -C ../polyfair-data pull
```

The **Run workflow** button, with `force` and `max_minutes`, starts a short
test run at any time.

Paper fills are optimistic (no queue position, no other takers racing for the
quote), and the live engine evaluates on every tick rather than every 5
minutes as the backtest does, so expect paper and backtest to differ somewhat.

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

Each Data API row is one taker order (every transaction has exactly one row),
so the stored `side` is the aggressor's side.

**Order books (`pmengine record` + GitHub Actions).** Polymarket only serves the
current book, so book history exists only if it is saved as it happens.
`pmengine record` takes one snapshot of every open strike's book (50 levels a
side, best first) and exits, writing `data/books/YYYY-MM-DD/HHMMSS.json.gz`.
`.github/workflows/record-books.yml` runs it every 15 minutes and commits each
file to the `data` branch, so no machine of yours has to stay on. To read the
history locally:

```bash
git fetch origin data && git worktree add ../polyfair-data data
```

Snapshots are about 13 KB, about 40 MB a month. This uses about 2,900 Actions
minutes a month: free on a public repo, more than a private repo on the Free
plan gets (2,000; use `7,37 * * * *` there). Both workflows push through
`.github/scripts/push-data.sh`, which retries from a fresh clone when the other
pushed first. GitHub can run scheduled jobs late, and it pauses them after 60
days without repo activity.

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
- A position settles in the books when its market is configured, or, for
  discovered markets (named by their slug), at the next startup after it resolves.
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
  paper.py        one unattended paper-trading day
  tape.py         trade tape: quotes before t, fills after t
  calibration.py  empirical price calibration from resolved markets
  backtest.py     walk-forward backtest on the tape
  strategies.py   fair-value models
  signals.py      fair value + book -> orders
  risk.py         limits and kill switch
  execution.py    paper simulator, live SDK adapter
  portfolio.py    cash, positions, PnL
  store.py        SQLite
  engine.py       event loop
  cli.py          run / scan / report
```
