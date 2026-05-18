# Tradeify-Style AI Trading Bot — MVP

> **Status:** Day 1 of the build plan complete (project skeleton, config,
> storage, logging, Docker). All trading logic — strategies, labeling,
> training, backtesting, risk engine, paper executor, scheduler, agents,
> reports — lands on Days 2 through 7.
>
> **Live trading is disabled.** This repo refuses to boot in `LIVE` mode
> unless both (a) `LIVE_ADAPTER_CONFIRMED=true` and (b) a real
> broker/exchange adapter has been implemented. Until then, the bot can
> only run `BACKTEST`, `TRAIN`, or `PAPER` modes.

## Goal

A modular, testable, paper-first trading bot for futures (MES/MNQ) and
crypto (BTC/etc.) that:

- runs 24/7 as a service but only trades during configured windows,
- optimizes for **realistic profitability** after slippage, commissions,
  drawdown, and unseen data — not maximum backtest profit,
- avoids overfitting, lookahead bias, HFT behavior, and unrealistic fills,
- treats a deterministic risk engine as authoritative; LLM agents are
  advisory only.

## Modes

| Mode      | What it does                                    | Status        |
|-----------|-------------------------------------------------|---------------|
| BACKTEST  | Simulate strategy on historical OHLCV CSVs      | Day 4         |
| TRAIN     | Label setups, train baseline + boosted models   | Day 3         |
| PAPER     | Live data → strategy → model → risk → simulator | Day 5         |
| LIVE      | **Locked.** Requires real adapter + opt-in flag | Out of scope  |

## Architecture (text diagram)

```
config/  ──▶  data/  ──▶  features/  ──▶  strategies/  ──▶  Setup
                                                              │
                          labeling/ ─── (TRAIN only) ─────────┤
                                                              ▼
                                              models/ predictor (prob ≥ τ)
                                                              │
                                                              ▼
                                          risk/  ── deterministic gate ──▶
                                                              │
                                                              ▼
                                  execution/  paper_executor (LIVE locked)
                                                              │
                                                              ▼
                                  backtesting/ engine + portfolio + metrics
                                                              │
              ┌───────────────┬──────────────┬────────────────┼───────────────┐
              ▼               ▼              ▼                ▼               ▼
          storage/      notifications/   reports/        scheduler/       agents/
          (sqlite)      (discord)        (md+json)       (apscheduler)    (advisory)
```

## Quick start

```bash
# 1. Create a venv (Python 3.11+ required)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Copy and edit your environment
cp .env.example .env

# 3. Smoke test (Day 1 deliverable)
python -m app.main --mode PAPER --dry-run
```

Expected output: a structured-log `boot` line, a `db.initialized` line,
and `dry_run.complete`.

## Docker

```bash
docker compose build
docker compose up
```

The Day-1 default command is the `--dry-run` smoke test. Edit
`docker-compose.yml` to flip it to a real mode once the runner exists.

## Configuration

All configuration is read from `.env` (and overridable via real env
vars). See `.env.example` for the full list with comments. The most
important safety knobs:

- `MODE` — `BACKTEST | TRAIN | PAPER | LIVE`. `LIVE` is locked.
- `LIVE_ADAPTER_CONFIRMED` — must be `true` to even attempt LIVE.
- `MAX_DAILY_LOSS`, `MAX_TRADES_PER_DAY`, `FORCE_FLAT_TIME` — hard risk.
- `CONFIDENCE_THRESHOLD` — probability needed for the model to approve.
- `DISCORD_WEBHOOK_URL` — alerting (optional).

## Adding historical CSV data

Drop OHLCV CSVs under `data/historical/<INSTRUMENT>/` with columns:

```
timestamp,open,high,low,close,volume
```

Loader implementation arrives on Day 2.

## Training, backtesting, paper trading

These commands are wired up on Days 3, 4, and 5 respectively:

```bash
python -m app.main --mode TRAIN          # Day 3
python -m app.main --mode BACKTEST       # Day 4
python -m app.main --mode PAPER          # Day 5
```

## Reports

Daily and backtest reports are written to `data/reports/` as Markdown +
JSON. (Day 6.)

## Discord notifications

Set `DISCORD_WEBHOOK_URL` in `.env`. The notification service handles
rate limits and retries. (Day 5.)

## Live trading — explicitly out of scope

This MVP intentionally **does not** include a working live broker or
exchange adapter. The placeholders in `execution/` raise on instantiation
to make it physically impossible to route real-money orders by accident.

To enable live trading later you must:

1. Implement a real adapter against your broker/exchange.
2. Replace `execution/live_executor_placeholder.py` with one that
   delegates to that adapter.
3. Set `LIVE_ADAPTER_CONFIRMED=true` in your environment.
4. Add integration tests covering dry-run order submission against a
   sandbox account before flipping `MODE=LIVE`.

## Build plan (one-week MVP)

| Day | Focus                                                        |
|-----|--------------------------------------------------------------|
|  1  | Skeleton + config + storage + logging + Docker (this commit) |
|  2  | Data loader, indicators, feature builder, VWAP/EMA strategy  |
|  3  | TP/SL labeler, time-split + walk-forward, model trainer      |
|  4  | Backtester, fills/commissions, risk engine, compliance       |
|  5  | Paper executor, scheduler, Discord notifications             |
|  6  | Reports + the bulk of the test suite                         |
|  7  | LLM advisory agents (with strict schemas) + polish + README  |

## Project layout

```
app/                    process entrypoint + logging
config/                 settings (Pydantic), instruments, trading windows
data/                   CSV loader, candle schema, market data service
features/               indicators, feature builder
strategies/             VWAP/EMA pullback, ORB
labeling/               TP/SL labeler
models/                 trainer, predictor, registry
validation/             time split, walk-forward, leakage checks
backtesting/            engine, portfolio, fills, metrics
risk/                   deterministic engine, kill switch, position sizing
compliance/             generic + Tradeify-inspired rules
execution/              base, paper, LIVE placeholders (refuse to instantiate)
agents/                 LLM advisory agents (schemas, no execution access)
notifications/          Discord webhook + dispatcher
reports/                daily, backtest, trade journal
scheduler/              APScheduler service + market hours
storage/                SQLAlchemy engine + ORM tables
tests/                  pytest tests (label, leakage, risk, lockout, ...)
```
