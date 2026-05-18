# Tradeify-Style AI Trading Bot — MVP

> **Status:** Days 1–7 of the build plan complete — the one-week MVP is
> done. The bot runs in `BACKTEST`, `TRAIN` (smoke), and `PAPER` mode end-
> to-end, writes per-session Markdown reports + CSV trade journals, ships
> Discord alerts, and runs five advisory LLM agents at end-of-day. Live
> trading remains explicitly out of scope.
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
| TRAIN     | Real OHLCV CSV → features → setups → labels → train + register  | ✓     |
| PAPER     | Live data → strategy → model → risk → simulator         | Day 5 ✓       |
| Reports   | Per-session Markdown report + CSV trade journal        | Day 6 ✓       |
| Agents    | LLM advisory agents (no execution access)              | Day 7 ✓       |
| Analysis  | Per-trade analysis, mistake classification, retrain    | Day 8 ✓       |
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

# Day 7 — run the LLM agent orchestrator with a deterministic mock LLM
python -m app.main --mode PAPER --smoke-agents

# Day 8 — per-trade analysis + mistake classification + mistake report
python -m app.main --mode PAPER --smoke-trade-analysis
```

Each smoke run is deterministic and lands in well under a minute.

## Real modes

The end-to-end real flow is **train → backtest → paper**. Each step
takes a real OHLCV CSV (`timestamp, open, high, low, close, volume`)
and the train step's saved model name flows into the next two.

### 1. Train a model from a real CSV

```bash
python -m app.main \
  --mode TRAIN \
  --train-csv data/historical/MES/1m.csv \
  --model-name vwap_ema_pullback_lr \
  --model-kind logreg
```

Useful overrides (all optional):

```bash
  --model-kind lightgbm        # requires `pip install lightgbm`
  --train-frac 0.70            # chronological train fraction (default 0.70)
  --val-frac 0.15              # chronological val fraction (default 0.15)
  --max-hold-bars 20           # TP/SL labeler horizon (default: settings.MAX_HOLD_BARS)
```

What the runner does:

1. Loads the CSV via the OHLCV loader (validates required columns,
   drops bad timestamps and OHLC violations, dedupes, sorts).
2. Builds canonical features using the same feature builder the
   strategy and predictor use.
3. Detects setups with `VWAPEMAPullback`.
4. Labels each setup with `label_setups` (TP / SL / time-out, with
   conservative same-bar SL-first ambiguity resolution).
5. Splits **strictly chronologically** by setup timestamp (no
   shuffling, ever).
6. Trains the selected model with walk-forward CV across (train +
   val) and calibrates probabilities on the val window.
7. Saves the model under `data/models/<model-name>/<version>/` with
   `model.pkl` + a rich `metadata.json` (CSV path, strategy, model
   kind, train/val/test ranges, label distribution, val and test
   metrics, `MAX_HOLD_BARS`, confidence threshold).

The runner refuses to train and exits non-zero on:

| Condition | Exit code |
|-----------|-----------|
| `--train-csv` missing or path doesn't exist | 4 / 7 |
| `--model-name` missing | 4 |
| CSV smaller than ~200 rows (warmup floor) | 4 |
| Strategy produces fewer than 100 setups | 4 |
| Labels are all-positive or all-negative | 4 |
| Required feature column missing | 4 |
| `train_frac + val_frac >= 1` | 4 |
| `--model-kind lightgbm` chosen but LightGBM not installed | 4 |

Console output during a run includes: `train.csv_loaded`,
`train.features_built`, `train.setups_detected`, `train.labels_built`,
`train.split` (with each window's start/end timestamps),
`train.metrics_val`, `train.metrics_test`, and `train.saved` (with
the version + on-disk path).

### 2. Backtest with the saved model

```bash
python -m app.main --mode BACKTEST \
  --backtest-csv data/historical/MES/1m.csv \
  --model-name vwap_ema_pullback_lr \
  --model-version latest
```

`--model-version latest` resolves to the most recent version in
`data/models/<model-name>/`. Backtest writes a Markdown + JSON report
under `data/reports/`.

### 3. Run paper mode with the saved model

```bash
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

## LLM advisory agents (Day 7)

Five agents run after the deterministic daily report writes its
artifacts. They are **read-only**: they cannot place trades, change risk
limits, or modify model thresholds. The only behavioural bridge from the
agent layer back into trading is the existing
`risk_engine.evaluate(..., high_risk_news_window=…)` flag, which can
**only block** entries — never approve them.

| Agent              | Schema (Pydantic)        | Purpose                                                              |
|--------------------|--------------------------|----------------------------------------------------------------------|
| `NewsAgent`        | `NewsAssessment`         | Macro/news risk; sets the `high_risk_window` flag block-only.        |
| `RiskExplainerAgent` | `RiskExplainerOutput`  | Plain-English explanations of today's risk blocks.                   |
| `TradeJournalAgent` | `TradeJournalNarrative` | Highlights / mistakes / lessons from today's trades.                 |
| `ReportAgent`      | `ReportCommentary`       | Headline + bullets appended to the daily Markdown.                   |
| `ModelReviewAgent` | `ModelReviewOutput`      | Calibration / drift commentary; advisory `retrain_recommended` only. |

**Safety guarantees enforced by code + tests:**

- `agents/` may not import `execution/` or `risk/` (`tests/test_agent_isolation.py`).
- Every output is validated against a frozen `extra="forbid"` schema; a
  parse or validation failure is persisted as `agent_outputs` with
  `schema_valid=False` and the bot keeps running.
- Notification or LLM failures never crash the scheduler or paper loop.
- Pre-session `NewsAgent` failures **keep** the previous high-risk flag
  rather than silently re-enabling trading.

**Running for real:** set `ENABLE_LLM_AGENTS=true` and provide
`OPENAI_API_KEY` in `.env`. Without those the orchestrator is a no-op
and the bot behaves exactly as in Days 1–6. The smoke command always
uses `MockLLMClient`, so it never makes a real network call.

## Trade analysis & mistake learning (Day 8)

After every closed trade the bot now runs a structured post-mortem and
classifies mistakes deterministically. The system **learns through
logging, validated retraining, and explicit promotion** — never through
in-place edits to strategy code, risk caps, or the
`CONFIDENCE_THRESHOLD`.

Pipeline (per close):

```
PaperTradingLoop.close_position
  → PostTradeAnalysisService.on_trade_closed
       → TradeAnalyzer.analyze_closed_trade   # deterministic
       → MistakeClassifier.classify          # rule-based, multiple tags
       → persist trade_analyses + trade_mistake_tags
       → reports/post_trade_report.py        # per-trade Markdown
       → TradeAnalysisAgent (LLM, optional)  # plain-English narrative
       → NotificationService.notify("trade.analysis", ...)
```

Pipeline (end-of-day, batch):

```
SchedulerService._safe_end_of_day
  → reports/mistake_report.py
       → PatternMiner.aggregate(...)
       → ImprovementSuggester.propose(...)   # validation_status="proposed"
       → persist_candidates(...)             # logged only, never auto-applied
```

Pipeline (operator-driven retraining + promotion):

```
python -m app.main --retrain-from-feedback   # build dataset + train candidate + comparison report
python -m app.main --promote-model VERSION   # only if PromotionDecision says so
```

See [Feedback retraining workflow](#feedback-retraining-workflow-paper--candidate--review--promote)
below for the full step-by-step.

Mistake tag enum (deterministic; tagged by `MistakeClassifier`):
`false_positive` (model approved + lost materially),
`low_confidence_trade`, `bad_orderflow_confirmation`,
`orderflow_divergence`, `entered_during_chop`, `low_volume_trade`,
`bad_time_of_day`, `high_volatility_spike`, `news_risk_trade`,
`stop_too_tight`, `target_too_far`, `poor_risk_reward`,
`strategy_conflict`, `slippage_loss`, `timeout_exit`, `rule_violation`,
`unknown`.

### Safety invariants

- The classifier is the **single source of truth** for mistake tags.
  The LLM agent only narrates — it cannot add or remove tags.
- `ImprovementSuggester` writes rows with
  `validation_status="proposed"`. **Never** automatic.
- `analysis/promotion.py` returns a `PromotionDecision`; calling code
  refuses to advance the model registry pointer unless every gate is
  satisfied (expectancy, profit factor, max-drawdown, false-positive
  rate, walk-forward stability).
- All analysis steps are wrapped — a per-trade analysis or LLM failure
  cannot break the trade-close path or the scheduler.
- `agents/trade_analysis_agent.py` still passes
  `tests/test_agent_isolation.py`: no `execution/` or `risk/` imports.

```bash
# Production-ish (one-shot end-of-day):
ENABLE_LLM_AGENTS=true OPENAI_API_KEY=sk-... python -m app.main --mode PAPER --smoke-agents

# Forever paper service: agents run after each EOD report and the
# pre-session NewsAgent runs daily at NEWS_CHECK_LOCAL_TIME.
ENABLE_LLM_AGENTS=true OPENAI_API_KEY=sk-... python -m app.main --mode PAPER
```

## Feedback retraining workflow: paper → candidate → review → promote

The bot only learns through **logged paper trades + an explicit
operator-driven retrain step**. Strategy code, risk caps, and
`CONFIDENCE_THRESHOLD` are *never* modified in place.

### 1. Run paper mode and let trades accumulate

```bash
python -m app.main --mode PAPER \
  --paper-csv data/historical/MES/1m.csv \
  --model-name vwap_ema_pullback_lr
```

Each closed paper trade is persisted with:

- the frozen feature snapshot at entry (`feature_snapshots.features`),
- the model's calibrated probability + threshold + approval flag
  (`model_predictions`),
- the realized PnL, exit reason, slippage, and commission (`closed_trades`),
- structured post-mortem + mistake tags (`trade_analyses`,
  `trade_mistake_tags`).

**How long to collect:** target at least `FEEDBACK_MIN_ROWS` closed
trades (default **100**) covering both wins *and* losses across multiple
sessions. With a paper-friendly cadence (~5–8 trades/day) that is
roughly **2–4 weeks of trading days**. Retraining on fewer rows is
explicitly refused — see step 3.

### 2. (Optional) Inspect the dataset before retraining

The retrain step writes both a CSV and a JSON dump under
`data/reports/feedback/`. You can also generate them ahead of time by
re-running `--retrain-from-feedback`; the dataset is built first and
training only runs after it.

### 3. Retrain a candidate model

```bash
python -m app.main --retrain-from-feedback
```

This:

1. Builds the feedback dataset from `closed_trades` ⨯
   `feature_snapshots` ⨯ `model_predictions` ⨯ `trade_mistake_tags`.
2. Refuses to train if there are fewer than `FEEDBACK_MIN_ROWS` rows
   (or fewer rows with full feature vectors). Exits with code `4`.
3. Splits **strictly chronologically** by entry timestamp
   (70% train / 15% val / 15% test). No shuffling, ever.
4. Trains a logistic-regression baseline; calibrates on val.
5. Optionally trains LightGBM if installed (metrics-only — the saved
   artifact is always the calibrated logreg).
6. Evaluates on the holdout test split: accuracy, precision, recall,
   false-positive rate, ROC-AUC, **expectancy per trade**,
   **profit factor**, **drawdown proxy**, and **calibration MAE** —
   all on the trades the candidate model would have approved.
7. Saves the candidate to
   `data/models/<model-name>_candidate/<version>/` with
   `metadata.json` carrying `candidate=true`, the realized trade
   metrics, the mistake-tag counts, and the chronological split
   ranges.
8. Writes a Markdown promotion report under
   `data/reports/feedback/promotion_*.md`.

**The candidate is never automatically promoted.** Mistake tags are
stored in metadata as a diagnostic signal; they are *not* used as
labels unless you opt in with `--feedback-use-mistake-tags`.

Useful overrides:

```bash
# Override min rows (e.g. for an early-validation pilot)
python -m app.main --retrain-from-feedback --feedback-min-rows 60

# Train against a non-default incumbent and pick a custom candidate name
python -m app.main --retrain-from-feedback \
  --model-name vwap_ema_pullback_lr \
  --candidate-model-name vwap_ema_pullback_lr_2026_q2

# Treat mistake-tagged trades as negative labels (advanced)
python -m app.main --retrain-from-feedback --feedback-use-mistake-tags
```

### 4. Review the candidate

Read the markdown report in `data/reports/feedback/`:

```
data/reports/feedback/
├── feedback_dataset.csv
├── feedback_dataset.json
└── promotion_<candidate-name>_<candidate-version>_<UTC>.md
```

The report shows incumbent vs candidate side-by-side, deltas, and any
**failed promotion gates** (expectancy, profit factor, max drawdown,
false-positive rate, walk-forward stability).

`metadata.json` for the candidate also contains the full test-split
metrics + chronological split ranges so you can reproduce the
evaluation independently.

### 5. Promote — only after validation

If, **and only if**, the report's PromotionDecision says `PROMOTE` and
you've reviewed the metrics, run:

```bash
python -m app.main --promote-model <candidate-version> \
  --model-name vwap_ema_pullback_lr_candidate
```

`--promote-model` re-runs the gate check at promotion time. Even if you
hand it a candidate version that *currently* clears the gates, the
incumbent metadata can change between retrain and promote — promotion
re-validates against whatever's live. The runner logs the decision and
refuses to advance the registry pointer if any gate fails.

### Safety summary

- **No auto-promotion.** `--retrain-from-feedback` only ever writes a
  new *candidate* version; it never touches the incumbent's directory
  or any "current" pointer.
- **Min rows enforced.** Below `FEEDBACK_MIN_ROWS` the trainer raises
  `InsufficientFeedbackError` and the CLI exits non-zero **before**
  any model is saved.
- **Chronological split.** `validation/time_split.chronological_split`
  is the only splitter the candidate trainer is allowed to call.
- **Mistake tags ≠ labels** by default. They become labels only when
  the operator explicitly opts in.
- **Tests:** `tests/test_feedback_retrain.py` covers feedback dataset
  creation, the insufficient-rows block, chronological split ordering,
  candidate save path, no-auto-promotion, and the override flags.

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
- `ENABLE_LLM_AGENTS` — Day 7 master switch (default `false`).
- `OPENAI_API_KEY` — required when agents are enabled.
- `LLM_MODEL` — OpenAI chat model (default `gpt-4o-mini`).
- `LLM_TIMEOUT_SECONDS` — per-call timeout (default 30).
- `AGENTS_RUN_AT_EOD` — whether the EOD scheduler job triggers the agents.
- `NEWS_CHECK_LOCAL_TIME` — pre-session NewsAgent cron time (default `09:25`).
- `FEEDBACK_MIN_ROWS` — minimum closed trades (with feature snapshots)
  required before `--retrain-from-feedback` will train a candidate
  (default 100). Below this the trainer refuses to run.
- `FEEDBACK_USE_MISTAKE_TAGS_AS_LABEL` — when true, derive candidate
  labels from mistake tags (default false; tags are metadata only).

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

Currently 260+ tests covering: settings + LIVE lockout, candles, CSV
loading, indicators, feature builder + leakage, strategy contract, TP/SL
labeling, time-split + walk-forward, predictor + drift, model registry,
fills, position sizing, risk engine, kill switch, portfolio, backtest
engine + accounting, compliance, paper executor, paper loop end-to-end,
shared trade-management exits, market-hour predicates, scheduler
behavior, daily report content, trade-journal CSV schema, Discord rate
limiting, notification persistence, model-failure safety, agent schema
strictness, LLM-client gating + mock dispatch, agent parse-failure
handling, orchestrator persistence + per-agent failure isolation,
pre-session-news flag persistence on failure, the NewsAgent → paper-loop
high-risk-news block-only chain, and the `agents/` ↔
`execution/`/`risk/` import-isolation invariant.

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
|  7  | LLM advisory agents (with strict schemas) + polish          | ✓      |
|  8  | Trade analysis + mistake learning + retrain/promote workflow | ✓     |

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
analysis/               post-trade analyzer, mistake classifier, pattern miner,
                        improvement suggester, feedback dataset, promotion
notifications/          Discord webhook + rate-limited dispatcher
paper/                  PaperTradingLoop (Day 5)
reports/                daily report, trade journal, backtest report,
                        per-trade post-mortem, daily mistake digest
scheduler/              APScheduler service + market hours
storage/                SQLAlchemy engine + ORM tables
tests/                  pytest tests (220+ covering all the above)
```
