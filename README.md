# Tradeify-Style AI Trading Bot — MVP

> **Status:** Days 1–6 of the build plan complete. The bot can now run in
> `BACKTEST`, `TRAIN` (smoke), and `PAPER` mode end-to-end, write per-session
> Markdown reports + CSV trade journals, and ship Discord alerts. LLM advisory
> agents are Day 7. Live trading remains explicitly out of scope.
>
> **Live trading is disabled.** This repo refuses to boot in `LIVE` mode
> unless both (a) `LIVE_ADAPTER_CONFIRMED=true` and (b) a real
> broker/exchange adapter has been implemented. Until then, the bot can
> only run `BACKTEST`, `TRAIN`, or `PAPER`.

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

| Mode      | What it does                                           | Status        |
|-----------|--------------------------------------------------------|---------------|
| BACKTEST  | Simulate strategy on historical OHLCV CSVs             | Day 4 ✓       |
| TRAIN     | Label setups, train baseline + boosted models          | Day 3 ✓       |
| PAPER     | Live data → strategy → model → risk → simulator         | Day 5 ✓       |
| Reports   | Per-session Markdown report + CSV trade journal        | Day 6 ✓       |
| Agents    | LLM advisory agents (no execution access)              | Day 7         |
| LIVE      | **Locked.** Requires real adapter + opt-in flag         | Out of scope  |

## Architecture

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
                                              ┌───────────────┴────────────┐
                                              ▼                            ▼
                                  backtesting/ engine                paper/ loop
                                  (Day 4)                            (Day 5)
                                              │                            │
                                              └────────────┬───────────────┘
                                                           ▼
                                          execution/ paper_executor (LIVE locked)
                                                           │
              ┌─────────────────┬───────────┴──────┬──────────────┬────────────────┐
              ▼                 ▼                  ▼              ▼                ▼
          storage/         notifications/      reports/       scheduler/       agents/
          (sqlite)         (discord)           (md+csv+json)  (apscheduler)    (advisory)
                                               (Day 6)        (Day 5)          (Day 7)
```

## Quick start

```bash
# 1. Create a venv (Python 3.11+ required)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Copy and edit your environment
cp .env.example .env

# 3. Smoke test (Day 1 deliverable — config + DB only)
python -m app.main --mode PAPER --dry-run
```

Expected output: a structured-log `boot` line, a `db.initialized` line,
and `dry_run.complete`.

## Smoke runs (one per day's deliverable)

```bash
# Day 2 — features + first strategy on synthetic OHLCV
python -m app.main --mode PAPER --smoke-features

# Day 3 — train a logistic-regression model end-to-end
python -m app.main --mode PAPER --smoke-train

# Day 4 — full backtest on synthetic OHLCV
python -m app.main --mode PAPER --smoke-backtest

# Day 5 — paper trading loop: a few bar cycles, then exit
python -m app.main --mode PAPER --smoke-paper --paper-cycles 50

# Day 6 — render a daily Markdown report + CSV trade journal
python -m app.main --mode PAPER --smoke-daily-report
```

Each smoke run is deterministic and lands in well under a minute.

## Real modes

```bash
# Backtest a real CSV with an optional model gate.
python -m app.main --mode BACKTEST \
  --backtest-csv data/historical/MES/1m.csv \
  --model-name vwap_ema_pullback_lr \
  --model-version latest

# Run the paper service forever (APScheduler).
python -m app.main --mode PAPER \
  --paper-csv data/historical/MES/1m.csv \
  --model-name vwap_ema_pullback_lr
```

`--mode PAPER` without smoke flags starts a `BlockingScheduler` that runs
`bar_close`, `force_flat`, `end_of_day`, and `heartbeat` jobs until you
`Ctrl-C`. End-of-day automatically writes the daily report + trade
journal and pings Discord with the file paths.

## Reports

Per-session artifacts land under `data/reports/` (override with
`REPORTS_DIR`):

```
data/reports/
├── backtest_<UTC>.md        # Day 4 backtest report
├── backtest_<UTC>.json
├── daily/
│   ├── daily_2026-05-18_MES.md     # Day 6 — markdown
│   └── daily_2026-05-18_MES.json   # Day 6 — payload
└── journals/
    └── trade_journal_2026-05-18_MES.csv  # Day 6 — per-trade audit trail
```

The Markdown daily report contains: summary metrics (Sharpe, profit factor,
drawdown, etc.), compliance flags (HFT detection + Tradeify-style rules),
risk-block counts by rule, a per-trade table, notification delivery
counts, and a snapshot of the active risk config.

## Discord notifications

Set `DISCORD_WEBHOOK_URL` in `.env`. The notification service rate-limits
to ~25 messages / 60s, drops low-priority kinds when the budget is full,
and squeezes through high-priority kinds (`trade.*`, `system.error`,
`forced_flat`, `eod.summary`). Failures never crash the bot — every
attempt is persisted to the `notifications` table for audit.

Webhook empty? The service falls back to log-only mode and still records
each "attempt" so the audit trail stays consistent.

## Docker

```bash
docker compose build
docker compose up
```

The default command is the Day-1 `--dry-run` smoke test. Edit
`docker-compose.yml` to flip to `--mode PAPER` once you've validated
your environment.

## Configuration

All configuration is read from `.env` (and overridable via real env
vars). See `.env.example` for the full list with comments. Key knobs:

- `MODE` — `BACKTEST | TRAIN | PAPER | LIVE`. `LIVE` is locked.
- `LIVE_ADAPTER_CONFIRMED` — must be `true` to even attempt LIVE.
- `MAX_DAILY_LOSS`, `MAX_TRADES_PER_DAY`, `FORCE_FLAT_TIME` — hard risk.
- `CONFIDENCE_THRESHOLD` — probability needed for the model to approve.
- `DISCORD_WEBHOOK_URL` — alerting (optional).
- `BAR_INTERVAL_SECONDS` — paper service tick rate (default 60).
- `ROLLING_WINDOW_BARS` — feature recomputation window (default 500).
- `PAPER_CSV_PATH` — optional CSV replayed by the paper service.
- `HEARTBEAT_LOCAL_TIME` — daily Discord heartbeat (default `08:00`).
- `REPORTS_DIR` — where Markdown + JSON + CSV reports land.

## Adding historical CSV data

Drop OHLCV CSVs under `data/historical/<INSTRUMENT>/` with columns:

```
timestamp,open,high,low,close,volume
```

The loader assumes naive timestamps are UTC and converts to your
configured `TIMEZONE` on load.

## Running the test suite

```bash
.venv/bin/python -m pytest -q
```

Currently 220+ tests covering: settings + LIVE lockout, candles, CSV
loading, indicators, feature builder + leakage, strategy contract, TP/SL
labeling, time-split + walk-forward, predictor + drift, model registry,
fills, position sizing, risk engine, kill switch, portfolio, backtest
engine + accounting, compliance, paper executor, paper loop end-to-end,
shared trade-management exits, market-hour predicates, scheduler
behavior, daily report content, trade-journal CSV schema, Discord rate
limiting, notification persistence, model-failure safety, and the
`agents/` ↔ `execution/`/`risk/` import-isolation invariant.

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

| Day | Focus                                                        | Status |
|-----|--------------------------------------------------------------|--------|
|  1  | Skeleton + config + storage + logging + Docker              | ✓      |
|  2  | Data loader, indicators, feature builder, VWAP/EMA strategy | ✓      |
|  3  | TP/SL labeler, time-split + walk-forward, model trainer     | ✓      |
|  4  | Backtester, fills/commissions, risk engine, compliance      | ✓      |
|  5  | Paper executor, scheduler, Discord notifications            | ✓      |
|  6  | Daily report + trade journal + scheduler EOD wiring         | ✓      |
|  7  | LLM advisory agents (with strict schemas) + polish          | —      |

## Project layout

```
app/                    process entrypoint + logging
config/                 settings (Pydantic), instruments, trading windows
data/                   CSV loader, candle schema, market data feeds
features/               indicators, feature builder
strategies/             VWAP/EMA pullback, ORB
labeling/               TP/SL labeler
models/                 trainer, predictor, registry
validation/             time split, walk-forward, leakage checks
backtesting/            engine, portfolio, fills, metrics, trade_management
risk/                   deterministic engine, kill switch, position sizing
compliance/             generic + Tradeify-inspired rules
execution/              base, paper executor, LIVE placeholders (refuse)
agents/                 LLM advisory agents (schemas, no execution access)
notifications/          Discord webhook + rate-limited dispatcher
paper/                  PaperTradingLoop (Day 5)
reports/                daily report, trade journal, backtest report
scheduler/              APScheduler service + market hours
storage/                SQLAlchemy engine + ORM tables
tests/                  pytest tests (220+ covering all the above)
```
