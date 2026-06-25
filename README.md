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
3. Detects setups with the strategy resolved through the registry
   (`--strategy`, default `vwap_ema_pullback`; pass
   `--strategy opening_range_breakout` to train the ORB plug-in).
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

## Multi-symbol mode

The bot supports a configurable symbol universe. Single-symbol bots
keep working unchanged (default: `ENABLED_SYMBOLS=MES`). Multi-symbol
mode kicks in automatically when `ENABLED_SYMBOLS` lists more than one
symbol.

### Configure the universe

```ini
# .env
ENABLED_SYMBOLS=MES,MNQ,MGC,MCL,MYM,M2K
PRIMARY_SYMBOL=MES                # optional; defaults to INSTRUMENT
MAX_ACTIVE_SYMBOLS=2              # max simultaneous open positions across symbols
MAX_TRADES_PER_SYMBOL_PER_DAY=4   # per-(symbol, session_date) cap
MAX_TOTAL_TRADES_PER_DAY=8        # global cap across all symbols
MARKET_TYPE=futures               # all symbols must share the same market type
```

Built-in registered symbols: `MES`, `MNQ`, `MGC`, `MCL`, `MYM`, `M2K`,
`BTC`, `ETH`. Add a new instrument by registering an
:class:`InstrumentSpec` in `config/instruments.py`.

`SymbolUniverse.from_settings(settings)` parses + validates the env:

- Duplicates are rejected (`SymbolUniverseError`).
- Unknown symbols are rejected with the supported set listed in the
  message.
- Mixing `futures` and `crypto` in one universe is rejected.
- Empty `ENABLED_SYMBOLS` falls back to `[INSTRUMENT]`.

### Per-symbol CSV layout

Drop one CSV per symbol under `data/historical/<SYMBOL>/<timeframe>.csv`:

```
data/historical/
├── MES/1m.csv
├── MNQ/1m.csv
├── MGC/1m.csv
└── ...
```

Each CSV must have the standard columns: `timestamp, open, high, low,
close, volume`. A missing or corrupt file disables only that symbol —
the orchestrator boots the rest and surfaces the disabled set in the
log line `paper.multi_symbol_loaded`.

### Multi-symbol paper mode

```bash
ENABLED_SYMBOLS="MES,MNQ,MGC,MCL,MYM,M2K" \
  python -m app.main --mode PAPER \
  --model-name vwap_ema_pullback_lr
```

What the orchestrator does each tick:

1. Polls every per-symbol incremental feed.
2. Runs each symbol's :class:`PaperTradingLoop` independently — its
   own portfolio, fills model, kill switch, risk engine, and
   strategy registry instance.
3. Before any submit, an `entry_gate` callback checks the caps:
   - `MAX_TRADES_PER_SYMBOL_PER_DAY` (and the existing
     `MAX_TRADES_PER_DAY` — whichever is lower wins);
   - `MAX_TOTAL_TRADES_PER_DAY` summed across all symbols;
   - `MAX_ACTIVE_SYMBOLS` against the count of per-symbol loops
     currently holding an open position.
4. Cap-blocked setups are recorded as `risk_blocks` rows with a
   structured rule (`per_symbol_day_cap`, `total_day_cap`,
   `max_active_symbols`) so the audit trail mirrors deterministic
   risk-engine blocks.
5. Per-symbol failures are isolated. A feed crash on `MNQ` cannot
   stop `MES` from running.

The strategy registry's existing conflict resolver still enforces "no
long+short on the same symbol" within a single tick (see "Strategies"
below). Across symbols, the `MAX_ACTIVE_SYMBOLS` cap is the safety
guarantee.

### Multi-symbol backtest

```python
from backtesting.engine import run_multi_symbol_backtest

result = run_multi_symbol_backtest(
    settings=settings,
    ohlcv_by_symbol={"MES": mes_df, "MNQ": mnq_df, "MGC": mgc_df},
    setups_by_symbol={"MES": mes_setups, "MNQ": mnq_setups, "MGC": mgc_setups},
)
print(result.to_dict()["per_symbol"])
print("best:", result.best_symbol(), "worst:", result.worst_symbol())
```

Each symbol runs through its own :class:`BacktestEngine` (independent
portfolio, fills, risk state). The aggregate equity curve is the
exit-time-sorted union of per-symbol trades; aggregate metrics are
computed off the merged trade list.

### Daily report — per-symbol breakdown

The daily report's payload now includes a `by_symbol` block with
trades, wins, win rate, net PnL, profit factor, expectancy, false
positives, and risk blocks per symbol. The Markdown render adds a
"Performance by symbol" table plus best/worst-symbol callouts. The
Discord EOD message (`eod.summary`) carries the same `by_symbol`
list so operators see which symbol paid the bills today.

### Broker adapters (paper/demo only)

Workflows are broker-agnostic — they only talk to
`integrations.broker_router.build_broker(settings)`. The router picks
an adapter based on `BROKER_PROVIDER`:

| `BROKER_PROVIDER` | When to use | Asset class | Status |
|-------------------|-------------|-------------|--------|
| `mock`            | Default for DRY_RUN, tests, local simulation | any | Always available |
| `alpaca`          | Current paper-automation sandbox | US **equities** (no futures) | Requires Alpaca paper keys |
| `tradovate`       | Future futures execution / funded-account-compatible | Futures (MES/MNQ/etc.) | Requires Tradovate demo API access |

LIVE is always refused regardless of provider — the router raises
`LiveExecutionRefused` and the workflow runner refuses to build a
context.

#### Alpaca PAPER (current sandbox)

Use Alpaca while Tradovate demo API access is unavailable. The
adapter refuses to construct unless **all** rails pass:

- `ALPACA_PAPER=true`
- `ALPACA_BASE_URL` contains `paper-api.alpaca.markets`
- `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` are set

`.env`:

```ini
BROKER_PROVIDER=alpaca
WORKFLOW_EXECUTION_MODE=PAPER
AUTONOMOUS_TRADING_ENABLED=true
ALPACA_API_KEY=your-paper-key
ALPACA_SECRET_KEY=your-paper-secret
ALPACA_PAPER=true
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

> **Alpaca does not support futures.** Symbols `MES, MNQ, ES, NQ,
> MGC, MCL, MYM, M2K` are rejected at every order method with the
> message *"Alpaca adapter does not support futures. Use local
> simulator or futures broker adapter."* For futures, run the local
> mock adapter or wait until Tradovate demo credentials are
> available.

#### Tradovate DEMO (future-use)

Kept for futures/funded-account-compatible execution. Construction
requires demo credentials and `demo.tradovateapi.com` in the URL —
both unavailable until you have a funded Tradovate account, but the
adapter and config are wired so the path is ready.

```ini
BROKER_PROVIDER=tradovate
TRADOVATE_DEMO=true
TRADOVATE_BASE_URL=https://demo.tradovateapi.com/v1
TRADOVATE_USERNAME=...
TRADOVATE_PASSWORD=...
TRADOVATE_APP_ID=...
TRADOVATE_APP_VERSION=1.0.0
TRADOVATE_CLIENT_ID=...
TRADOVATE_CLIENT_SECRET=...
```

#### Smoke commands

```bash
python -m app.main --workflow premarket          # DRY_RUN, mock, no orders
python -m app.main --workflow run-day            # full DRY_RUN sequence
# Live workflow paper orders (Alpaca example):
WORKFLOW_EXECUTION_MODE=PAPER BROKER_PROVIDER=alpaca \
AUTONOMOUS_TRADING_ENABLED=true \
python -m app.main --workflow market-open --no-workflow-dry-run
```

> Workflows must call only `broker_router`. A test
> (`tests/test_workflow_broker_isolation.py`) fails the build if any
> file under `workflows/` imports `AlpacaPaperClient`,
> `TradovateDemoClient`, or `MockBroker` directly.

### Tradovate DEMO/PAPER broker adapter

The bot executes paper futures orders **directly through the
Tradovate demo API** — TradingView is not in the execution path.

```
Python bot → Tradovate demo API → simulated futures fills → Discord
```

- **DRY_RUN** (default): the broker router always returns the
  in-memory `MockBroker`. No network calls, no Tradovate auth, no
  external orders.
- **PAPER + `BROKER_PROVIDER=tradovate`**: the router builds a
  `TradovateDemoClient` only when **all** safety rails pass:
  - `TRADOVATE_DEMO=true`
  - `TRADOVATE_BASE_URL` contains `demo.tradovateapi.com`
  - `TRADOVATE_USERNAME` / `TRADOVATE_PASSWORD` / `TRADOVATE_APP_ID` set
- **LIVE**: the router raises `LiveExecutionRefused` immediately.
  The existing repo-wide LIVE lockout still applies.

`.env` block (paper/demo only — never commit real credentials):

```ini
BROKER_PROVIDER=tradovate
DEFAULT_ORDER_TYPE=limit
TRADOVATE_USERNAME=your-demo-user
TRADOVATE_PASSWORD=your-demo-pass
TRADOVATE_APP_ID=tradeify-bot
TRADOVATE_APP_VERSION=1.0.0
TRADOVATE_CLIENT_ID=your-client-id
TRADOVATE_CLIENT_SECRET=your-client-secret
TRADOVATE_DEMO=true
TRADOVATE_BASE_URL=https://demo.tradovateapi.com/v1
TRADOVATE_WS_URL=wss://demo.tradovateapi.com/v1/websocket
```

Smoke-test the wiring without sending orders:

```bash
python -m app.main --workflow premarket            # DRY_RUN by default
python -m app.main --workflow run-day              # full DRY_RUN sequence
```

Run with the Tradovate demo adapter (still simulation, never live):

```bash
WORKFLOW_EXECUTION_MODE=PAPER \
AUTONOMOUS_TRADING_ENABLED=true \
BROKER_PROVIDER=tradovate \
python -m app.main --workflow market-open --no-workflow-dry-run
```

Supported symbols (rejected otherwise): `MES, MNQ, ES, NQ, MGC, MCL,
MYM, M2K`. Limit orders are preferred — `place_market_order` is
allowed only when `DEFAULT_ORDER_TYPE=market`.

Every order method returns a structured `OrderResult` (JSON-safe via
`to_payload()`). Credentials are never logged: outgoing request
bodies pass through `redact_secrets` first.

> **Warnings**
> - Demo / simulation only. There is no live execution path.
> - TradingView is **optional** for charting, manual monitoring, and
>   external alert ideas. **Automated execution does not go through
>   TradingView.** Webhook signals (if configured) still pass model,
>   risk, and broker validation, and only place demo orders.
> - Setting `TRADOVATE_DEMO=false` or pointing `TRADOVATE_BASE_URL`
>   away from `demo.tradovateapi.com` raises
>   `TradovateConfigurationError` at construction.

### Workflow signal engine (real strategy + model)

The autonomous workflow no longer assumes a direction. `market-open`
runs the **real VWAP/EMA strategy** (and an optional model gate) per
symbol via `workflows/signal_engine.py`:

```
market-open(symbol)
  → load data/historical/<SYM>/1m.csv
  → build canonical features
  → VWAPEMAPullback.detect_setups (latest bar)
  → optional model gate (WORKFLOW_MODEL_NAME)
  → WorkflowSignal(direction, entry, stop, target, confidence) OR skip
```

- No setup / no data / model rejection → the symbol is **skipped**
  (never a forced long).
- `WORKFLOW_MODEL_NAME` (optional) gates entries with a trained model;
  leave empty for strategy-only mode.
- The resulting direction/stop/target flow into the equity bracket
  order or the options layer.

### Continuous intraday loop (autonomous paper)

The intraday scanner is the genuinely autonomous mode. It re-scans the
universe every `WORKFLOW_SCAN_INTERVAL_MINUTES` during the trading
window and places bracket orders on approved setups:

```bash
# Dry run (real signals, no orders) — safe to leave running
python -m app.main --workflow-intraday

# Live paper (places Alpaca paper bracket orders on approved setups)
python -m app.main --workflow-intraday --no-workflow-dry-run
```

Per cycle, for each symbol: refresh bars → `SignalEngine` → long-only
filter → risk caps → bracket order → Discord alert. Outside the trading
window it sleeps without scanning. LIVE is refused; DRY_RUN never orders.

#### Daily risk guardrails (enforced every cycle)

Before any entry, the loop applies deterministic daily caps and stops
the session when they trip:

| Guardrail | Behavior |
|-----------|----------|
| `MAX_DAILY_LOSS` | Flatten all positions + **halt** new entries for the day |
| `MAX_DAILY_PROFIT` | Stop opening new trades for the day (lock in) |
| `MAX_TRADES_PER_DAY` | Block further entries once reached |
| `FORCE_FLAT_TIME` | Flatten all open positions at session close |
| Already held / working order | Dedupe — never stack a second order on the same symbol |

Day P&L comes from the broker account; position/order reconciliation
runs once per cycle so the loop never double-enters a name it already
holds. Counters reset at the start of each session date.

#### Dynamic universe (research-agent watchlist, allowlist-gated)

By default the loop scans `ENABLED_SYMBOLS`. Set
`WORKFLOW_DYNAMIC_UNIVERSE=true` to expand the scan set using a **vetted
liquid allowlist** (`config/equity_allowlist.py`):

```ini
WORKFLOW_DYNAMIC_UNIVERSE=true
WORKFLOW_MAX_UNIVERSE=15
WORKFLOW_AGENT_WATCHLIST=true   # let research agents rank allowlist names
WORKFLOW_LONG_ONLY=true
WORKFLOW_SCAN_INTERVAL_MINUTES=5
```

Safety model (unchanged guarantee): research agents may only **rank
allowlist names** to prioritize the scan — they can never invent a
ticker and never decide a trade. Every symbol (pinned or discovered)
still passes the deterministic gate: data → VWAP/EMA setup → model →
risk → broker validation. `ENABLED_SYMBOLS` are always scanned first;
the cap bounds total symbols per cycle.

### Position sizing, time-of-day filter, paper report, deployment

- **Risk-based sizing** (`USE_RISK_BASED_SIZING=true`): equity trades are
  sized as `RISK_PER_TRADE / per-share stop distance`, capped by
  `MAX_SHARES_PER_TRADE` and `MAX_NOTIONAL_PER_TRADE` — so a $50 and a
  $900 stock risk the same and never exceed buying power.
- **Time-of-day filter** (`STRATEGY_SKIP_OPEN_MINUTES` /
  `STRATEGY_SKIP_CLOSE_MINUTES`): skip the noisy RTH open/close.
- **Live paper report**: `python -m app.main --paper-report` prints a
  broker-sourced snapshot (equity, day P&L, positions, working orders)
  and sends a Discord summary.
- **Deployment**: `deploy/` has a `run_intraday.sh` wrapper, a macOS
  `launchd` plist, and a Linux `systemd` unit for hands-off auto-start.
  See `deploy/README.md`.

> **Out-of-sample validation:** the 0.75/1.5 R:R was confirmed on a
> 75/25 time split — test-set profit factor 1.069 (positive), vs the old
> 1.0/2.0 which went negative out-of-sample (PF 0.952). PDT rules do not
> apply to paper accounts; revisit before any live trading.

### Live data ingestion (Alpaca bars)

Download real 1-minute bars into the repo convention
(`data/historical/<SYM>/1m.csv`) so signals run on real data:

```bash
# ENABLED_SYMBOLS, last 30 days
python -m app.main --download-data

# Specific symbols / window
python -m app.main --download-data --download-symbols SPY,QQQ,AAPL --download-days 60
```

Read-only market data (`data/alpaca_bars.py`) — never places orders.
Uses the same paper keys but only the data endpoints. One symbol
failing does not abort the others.

### Multi-asset support (futures, equities, options)

The VWAP/EMA strategy is instrument-agnostic — it only needs OHLCV bars.
The bot now supports three asset classes:

| Asset    | Symbols (examples)        | Fills model        | Broker |
|----------|---------------------------|--------------------|--------|
| Futures  | MES, MNQ, MGC, MCL…       | `FuturesFillsModel` | Tradovate demo / mock |
| Equities | SPY, QQQ, AAPL, MSFT…     | `EquityFillsModel`  | Alpaca paper / mock |
| Options  | (on the equity underlying) | premium-based      | Alpaca paper / mock |

Equities use penny ticks + per-share commission (Alpaca paper is
commission-free). Any equity ticker can be registered on demand via
`config.instruments.register_equity`.

### Options trading layer (paper / simulation only)

Options sit on top of the underlying's VWAP/EMA signal — a long signal
on SPY becomes a call (or bull call spread), a short signal becomes a
put (or bear put spread). There is **no live path**.

```
VWAP/EMA signal (SPY long)
  → selection (ATM call / vertical / iron condor)
  → options risk gate (premium cap, DTE, max positions)
  → execution (Alpaca paper or mock)
  → position manager (greeks, profit target, expiry auto-close, rolling)
  → Discord alert (underlying + strategy + reason)
```

The `options/` package:

| Module | Role |
|--------|------|
| `contract.py` | `OptionContract` + OCC symbol encode/decode |
| `greeks.py` | Black-Scholes greeks (pure Python, no SciPy) |
| `chain.py` | `SyntheticChainProvider` (tests/backtest) + `AlpacaChainProvider` |
| `selection.py` | ATM directional, vertical spread, iron condor selection |
| `risk_rules.py` | Deterministic entry gate (premium / DTE / open positions) |
| `execution.py` | `MockOptionsExecutor` + `AlpacaOptionsExecutor` (single + multi-leg) |
| `position_manager.py` | Tracks positions, greeks, profit/stop, expiry, rolling |
| `trader.py` | `OptionsTrader` — wires signal → selection → risk → execution |

`.env`:

```ini
OPTIONS_ENABLED=true
OPTIONS_STRATEGY=atm_directional      # or vertical_spread / iron_condor
OPTIONS_ENABLED_UNDERLYINGS=SPY,QQQ,AAPL,MSFT
OPTIONS_DEFAULT_DTE=30
OPTIONS_MAX_PREMIUM_PER_TRADE=500
OPTIONS_MAX_OPEN_POSITIONS=5
OPTIONS_AUTO_CLOSE_DTE=5
OPTIONS_AUTO_ROLL=true
OPTIONS_PROFIT_TARGET_PCT=50
OPTIONS_STOP_LOSS_PCT=-50
```

> **Demo / simulation only.** Options execution runs against Alpaca's
> paper endpoints or the in-memory mock. The position manager
> auto-closes near expiry, takes profit / stops out by percentage, and
> can roll positions forward. All advisory + paper — no live execution.

#### How options wire into the workflow

When `OPTIONS_ENABLED=true`, the market-open workflow routes a
directional signal on an allowed underlying (`OPTIONS_ENABLED_UNDERLYINGS`)
into the options layer instead of buying shares:

```
market-open: SPY long signal
  → workflows/options_execution.execute_options_signal
  → OptionsTrader.handle_signal (selection → risk → execution)
midday: OptionsTrader.manage() → profit/stop/expiry/roll
```

DRY_RUN never routes to options (it short-circuits to a simulated
hold). Symbols not listed in `OPTIONS_ENABLED_UNDERLYINGS` fall through
to the normal equity/broker path.

### Equity bracket orders (Alpaca)

Equity entries use a **native Alpaca bracket order** — the entry limit
plus an attached `stop_loss` (and a synthesized `take_profit` when none
is supplied) are submitted as one `order_class=bracket` request.

This replaced the earlier "entry then a separate stop" flow, which
Alpaca rejected with `403 Forbidden` because the protective stop was
sent before the entry filled. With a bracket order Alpaca arms the
protective legs automatically once the entry fills.

- `BaseBroker.place_bracket_order(...)` — default falls back to a plain
  limit entry for adapters without native brackets (e.g. Tradovate).
- `AlpacaPaperClient.place_bracket_order(...)` — native bracket payload.
- `MockBroker.place_bracket_order(...)` — simulated fill carrying the
  protective legs in `raw`.

The workflow calls this via
`workflows/order_execution.execute_entry_with_stops(...)`.

### Parallel paper evaluation

Run Alpaca paper and futures local simulation side-by-side as
independent evaluation tracks. They share **nothing**: no broker state,
no open positions, no trade IDs, no state files.

| Track         | Purpose                          | Symbols           | Broker      |
|---------------|----------------------------------|-------------------|-------------|
| `alpaca`      | Test broker/API plumbing         | SPY, QQQ, AAPL… | Alpaca Paper |
| `futures_sim` | Test futures strategy/model      | MES, MNQ, MGC… | Local MockBroker |

```ini
ENABLE_PARALLEL_PAPER=true
PARALLEL_BROKERS=futures_sim,alpaca
ALPACA_ENABLED_SYMBOLS=SPY,QQQ,AAPL,MSFT
FUTURES_SIM_ENABLED_SYMBOLS=MES,MNQ,MGC,MCL
ALPACA_EVALUATION_ID=alpaca_2week_test
FUTURES_SIM_EVALUATION_ID=futures_sim_2week_test
```

```bash
# Launch both tracks
python -m app.main --start-parallel-paper --no-workflow-dry-run

# Check status
python -m app.main --parallel-paper-status

# Generate reports
python -m app.main --parallel-paper-report
```

Reports land in `data/evaluations/<evaluation_id>/` per track plus
a `data/evaluations/parallel_summary.md` combined overview. The
summary states explicitly:

> Alpaca tests broker/API plumbing. futures_sim tests futures
> strategy/model behavior. Do not compare PnL directly across them.

Every Discord alert includes `broker_provider` + `evaluation_id`.

### TradingView webhook (optional input)

Webhooks are an *additional* signal source — the bot's primary
scanner is the per-symbol paper loop, which polls feeds on the
configured cadence regardless of whether any webhook is wired.

There are two layers:

1. **Validator** — `webhook.validate_webhook_signal(...)`. Pure
   function; turns a JSON payload into a typed `WebhookSignal` and
   filters by `ENABLED_SYMBOLS`. Suitable for embedding in any
   HTTP layer.
2. **HTTP endpoint** — the `webhooks/` package ships a FastAPI app
   that wraps the validator with risk gating + paper-trade execution.

```python
from webhook import validate_webhook_signal
from config.instruments import SymbolUniverse

universe = SymbolUniverse.from_settings(settings)
signal = validate_webhook_signal(
    {"symbol": "MES", "direction": "long", "price": 4500.25, "stop": 4495.0, "target": 4510.0},
    universe=universe,
    expected_secret="hunter2",  # optional shared secret
)
```

Validator behavior:

- **Symbols not in `ENABLED_SYMBOLS` are rejected** with
  `InvalidWebhookSignal`. The bot never trades a symbol operations
  did not whitelist, even if a third-party alert says to.
- Missing required fields, malformed prices, unknown directions, or
  a bad shared secret all produce `InvalidWebhookSignal` (HTTP layers
  should map this to `400 Bad Request`).
- `direction` accepts both `long`/`short` and `buy`/`sell`.
- An accepted signal returns a typed `WebhookSignal` ready to be
  turned into a setup by the caller. Live broker execution is out
  of scope; the MVP only paper-trades.

## TradingView webhook server

The `webhooks/` package adds a runnable FastAPI app that ingests
TradingView alerts and converts them into paper trades through the
exact same risk engine the paper loop uses. The endpoint never
talks to a live broker — even when `MODE=LIVE` (live mode is locked
behind a separate flag and the live executor is a placeholder).

### Endpoint

```
POST /webhooks/tradingview
```

Body shape (TradingView's standard alert message JSON):

```json
{
  "secret": "your-shared-secret",
  "source": "tradingview",
  "symbol": "MNQ1!",
  "time": "{{time}}",
  "price": "{{close}}",
  "action": "long",
  "strategy": "vwap_pullback",
  "timeframe": "{{interval}}",
  "stop": "{{plot('stop')}}",
  "target": "{{plot('target')}}"
}
```

- `action` is one of `long` / `short` / `close` (`buy` / `sell` are
  accepted as aliases).
- `secret` is checked against `TRADINGVIEW_WEBHOOK_SECRET`. Can also
  be sent as the `X-Webhook-Secret` header. When the env var is
  unset the secret check is skipped (do not run open to the
  internet without a secret).
- `stop` / `target` are optional. When omitted they default to
  `WEBHOOK_DEFAULT_STOP_TICKS` / `WEBHOOK_DEFAULT_TARGET_TICKS`
  ticks from `price`. Always include them in your alert message
  for accurate execution.

### Pipeline (per request)

```
receive  ->  validate (FastAPI 422 on bad shape)
         ->  check TRADINGVIEW_WEBHOOK_SECRET           (401 on mismatch)
         ->  normalize symbol  ("MNQ1!" -> "MNQ", etc.)
         ->  reject if symbol not in ENABLED_SYMBOLS
         ->  optional model gate (off by default — webhook signals
                                  do not carry a feature snapshot)
         ->  risk_engine.evaluate(...)                  authoritative
         ->  PaperExecutor.submit(...)                  paper-only
```

Symbol normalization handles TradingView's front-month continuous
suffixes and exchange prefixes:

| Input               | Output |
| ------------------- | ------ |
| `MNQ1!`, `MES1!`    | `MNQ`, `MES` |
| `NQ1!`, `ES1!`      | `NQ`, `ES`   |
| `BINANCE:BTCUSDT`   | `BTC` (when `MARKET_TYPE=crypto`) |
| `ETH/USDT`          | `ETH` (when `MARKET_TYPE=crypto`) |

### Notifications

Every webhook touch emits a notification (Discord + the
`notifications` audit table) so operators can replay an alert end
to end:

| Kind                   | Trigger                                  |
| ---------------------- | ---------------------------------------- |
| `webhook.received`     | Payload arrived (before validation).     |
| `webhook.invalid`      | Bad secret / unknown symbol / bad shape. |
| `webhook.approved`     | Risk engine accepted, before submit.     |
| `webhook.blocked`      | Risk engine refused, with rule + reason. |
| `webhook.trade_opened` | PaperExecutor filled an entry.           |
| `webhook.closed`       | `action=close` flattened a position.     |

A failing notifier (e.g. Discord 5xx) never propagates: the
endpoint always returns a structured `WebhookResponse`.

### Run a local server

```bash
# Install deps if you haven't already
pip install -r requirements.txt

# Configure the universe + secret in .env
ENABLED_SYMBOLS=MES,MNQ,MGC
PRIMARY_SYMBOL=MES
TRADINGVIEW_WEBHOOK_SECRET=replace-me-with-something-random
DISCORD_WEBHOOK_URL=...        # optional but recommended
MODE=PAPER

# Boot the FastAPI app — uvicorn entry uses webhooks.create_app
python -m uvicorn "webhooks:create_app" --factory --host 127.0.0.1 --port 8000
```

The standalone server runs its own in-process `PaperExecutor` —
its portfolio is **not** shared with the per-symbol paper loop.
For a single combined process (paper loop + webhook ingest sharing
state) build the app from your own service code:

```python
from webhooks import build_webhook_router

router = build_webhook_router(
    settings=settings,
    universe=universe,
    executor=paper_loop_executor,           # share with paper loop
    notifier_notify=notifier.notify,
    high_risk_news_fn=orchestrator.high_risk_news_active,
)
fastapi_app.include_router(router)
```

### Expose the server to TradingView

TradingView only POSTs to public URLs. Use a tunnel:

**Option A: ngrok**

```bash
# Free plan is enough for testing.
ngrok http 8000
# -> Forwarding https://<random>.ngrok-free.app -> http://localhost:8000
```

Use `https://<random>.ngrok-free.app/webhooks/tradingview` as the
alert URL.

**Option B: Cloudflare Tunnel** (recommended for longer runs)

```bash
brew install cloudflared            # or your platform's package manager
cloudflared tunnel --url http://localhost:8000
# -> https://<your-tunnel>.trycloudflare.com
```

Use `https://<your-tunnel>.trycloudflare.com/webhooks/tradingview`.

For production-style runs, set up a named tunnel + DNS record so
the URL doesn't change on each restart.

### Create a TradingView alert

1. Open the chart for a symbol you have enabled (e.g. `CME_MINI:MES1!`).
2. Add the alert (Alt-A on a chart). Set "Condition" to your
   indicator / strategy of choice.
3. In **Notifications** -> **Webhook URL**, paste your tunnel URL +
   `/webhooks/tradingview`.
4. In **Message**, paste this JSON. TradingView substitutes the
   `{{...}}` placeholders at fire time:

   ```json
   {
     "secret": "replace-me-with-something-random",
     "source": "tradingview",
     "symbol": "{{ticker}}",
     "time": "{{time}}",
     "price": "{{close}}",
     "action": "long",
     "strategy": "{{strategy.order.alert_message}}",
     "timeframe": "{{interval}}"
   }
   ```

5. Save. When the condition fires TradingView POSTs the JSON above
   to your bot.

### Security warnings

- **Do not put broker credentials, API keys, or account numbers in
  TradingView alert messages.** TradingView stores alert text
  server-side and surfaces it in their UI; treat it as
  semi-public.
- Always set `TRADINGVIEW_WEBHOOK_SECRET` and run the bot behind
  HTTPS (ngrok / Cloudflare Tunnel both terminate TLS for you).
  The secret is only useful when the channel is encrypted —
  otherwise an on-path observer can replay it.
- Rotate the secret if it ever appears in a screenshot, gist, or
  shared chart.
- Keep the host firewalled to localhost (the tunnel does the
  public-facing job). Do not bind uvicorn to `0.0.0.0` in dev.

## Strategies

Strategies are plug-ins. The in-tree set lives under `strategies/` and
registers itself with `strategies.registry.STRATEGY_REGISTRY` at import
time. Built-ins:

- `vwap_ema_pullback` — VWAP/EMA trend pullback (default).
- `opening_range_breakout` — fresh breakout above/below the session's
  opening range.

`MODE=TRAIN` and `MODE=BACKTEST` run a single named strategy via
`--strategy NAME`. Paper mode reads `ENABLED_STRATEGIES` (comma-
separated env var, default `vwap_ema_pullback`) to decide which
strategies to run. Passing `--strategy NAME` to paper mode overrides
the env and runs only that one strategy.

```bash
# Train ORB instead of the default
python -m app.main --mode TRAIN \
  --train-csv data/historical/MES/1m.csv \
  --model-name orb_lr \
  --strategy opening_range_breakout

# Run multiple strategies in paper mode
ENABLED_STRATEGIES="vwap_ema_pullback,opening_range_breakout" \
  python -m app.main --mode PAPER \
  --paper-csv data/historical/MES/1m.csv \
  --model-name vwap_ema_pullback_lr
```

### Multi-strategy conflict resolution

When two enabled strategies fire on the same instrument and the same
bar, the loop never takes both opposing positions. The resolver lives
in `strategies/registry.py` and applies these rules in order:

1. Same-direction signals on the same symbol all survive (e.g. two
   strategies both going long).
2. If long *and* short are present on the same symbol:
   - both sides scored + approved → keep the higher confidence;
   - either side missing a model confidence → drop both (we refuse to
     guess);
   - neither side approved → drop both.
3. Every conflict is logged (`strategy.conflict`) and emits a
   notification with the winner + dropped setup ids for the audit
   trail.

### Adding a new strategy

```python
# strategies/my_strategy.py
from strategies.base import Strategy, Setup, StrategyParams

class MyStrategy(Strategy):
    name = "my_strategy"
    @classmethod
    def _default_params(cls): return StrategyParams()
    def detect_setups(self, features_df): ...
```

Then register it in `strategies/registry.py`:

```python
from strategies.my_strategy import MyStrategy
STRATEGY_REGISTRY.register(MyStrategy)
```

`tests/test_strategy_registry.py` enforces that every registered
strategy is reachable via `--strategy` and via `ENABLED_STRATEGIES`,
plus that the conflict resolver rules above hold.

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

**Running for real:** set `ENABLE_LLM_AGENTS=true` and provide at
least one provider key (`OPENAI_API_KEY`, `PERPLEXITY_API_KEY`,
`ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`) in `.env`. Without an
enabled key, the orchestrator is a no-op and the bot behaves exactly
as in Days 1–6. The smoke command always uses `MockLLMClient`, so it
never makes a real network call.

### Agent provider routing

Each advisory agent picks the right tool for its job:

- **Perplexity** for live, web-grounded research — `NewsAgent`,
  `MacroNewsAgent`, `StrategyResearchAgent`.
- **OpenAI** for structured analysis / reporting — `TradeAnalysisAgent`,
  `ModelReviewAgent`, `ReportAgent`, `RiskExplainerAgent`,
  `TradeJournalAgent`, `BacktestCriticAgent`.
- **Deterministic code** by default — `ModelDriftAgent`,
  `DataQualityAgent`. They never call an LLM unless the operator
  explicitly opts in (set the agent's `*_AGENT_PROVIDER` to a real
  provider AND its `*_AGENT_MODEL` to a real model).

The router lives at `agents/providers/router.py`; agents themselves
are unchanged. `Anthropic` and `Gemini` are recognized as valid
provider names; their providers ship in this repo and activate as
soon as `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` is set.

| Agent (`name`)          | Default provider     | Default model env                | Override env var                    |
| ----------------------- | -------------------- | -------------------------------- | ----------------------------------- |
| `news`                  | Perplexity           | `PERPLEXITY_DEFAULT_MODEL`       | `NEWS_AGENT_PROVIDER` / `NEWS_AGENT_MODEL` |
| `macro_news`            | Perplexity           | `PERPLEXITY_DEFAULT_MODEL`       | `MACRO_NEWS_AGENT_PROVIDER` / `MACRO_NEWS_AGENT_MODEL` |
| `strategy_research`     | Perplexity           | `STRATEGY_RESEARCH_AGENT_MODEL` (`sonar-deep-research`) | `STRATEGY_RESEARCH_AGENT_PROVIDER` / `STRATEGY_RESEARCH_AGENT_MODEL` |
| `trade_analysis`        | OpenAI               | `OPENAI_DEFAULT_MODEL`           | `TRADE_ANALYSIS_AGENT_PROVIDER` / `TRADE_ANALYSIS_AGENT_MODEL` |
| `report`                | OpenAI               | `OPENAI_DEFAULT_MODEL`           | `REPORT_AGENT_PROVIDER` / `REPORT_AGENT_MODEL` |
| `risk_explainer`        | OpenAI               | `OPENAI_DEFAULT_MODEL`           | `RISK_EXPLAINER_AGENT_PROVIDER` / `RISK_EXPLAINER_AGENT_MODEL` |
| `trade_journal`         | OpenAI               | `OPENAI_DEFAULT_MODEL`           | `TRADE_JOURNAL_AGENT_PROVIDER` / `TRADE_JOURNAL_AGENT_MODEL` |
| `model_review`          | OpenAI               | `OPENAI_REVIEW_MODEL`            | `MODEL_REVIEW_AGENT_PROVIDER` / `MODEL_REVIEW_AGENT_MODEL` |
| `backtest_critic`       | OpenAI               | `OPENAI_REVIEW_MODEL`            | `BACKTEST_CRITIC_AGENT_PROVIDER` / `BACKTEST_CRITIC_AGENT_MODEL` |
| `model_drift`           | none (stats)         | n/a                              | `MODEL_DRIFT_AGENT_PROVIDER` / `MODEL_DRIFT_AGENT_MODEL` |
| `data_quality`          | none (deterministic) | n/a                              | `DATA_QUALITY_AGENT_PROVIDER` / `DATA_QUALITY_AGENT_MODEL` |

Each `*_AGENT_PROVIDER` accepts `openai`, `perplexity`, `anthropic`,
`gemini`, or `none` / `off` / `disabled` (turns the agent off without
removing it from the orchestrator). The Settings layer rejects any
other value at config-load time so a typo cannot silently disable a
critical agent.

**Routing rules:**

- An agent is enabled only when **(a)** its provider name is
  recognized and **(b)** that provider's API key is configured. A
  missing key disables that agent — every other agent keeps running.
- Per-agent model overrides flow through to the underlying provider
  instance, so `BacktestCriticAgent` runs on `OPENAI_REVIEW_MODEL`
  while `TradeJournalAgent` runs on `OPENAI_DEFAULT_MODEL` even
  though both are routed to OpenAI.
- The router never logs API keys. Construction / disable logs only
  carry `provider`, `model`, `agent`, and a `reason` string.
- Provider failures (network, auth, bad shape) become
  `LLMClientError` -> the existing `BaseAgent.run()` handler catches
  them and persists `AgentResult(schema_valid=False, error=...)`.
  No agent failure can ever crash the scheduler or the paper loop.

#### Where keys go

- **`/Users/<you>/.../.env`** (your local file, `.gitignore`d) holds
  real keys.
- **`.env.example`** is committed and contains placeholders only.
- **`config/settings.py`** holds the field definitions + default
  models. It does not store keys.
- **`agents/providers/router.py`** reads keys via `Settings`,
  constructs providers lazily, and never echoes keys anywhere.

**NEVER commit `.env`.** If a real key ever lands in the repo, rotate
it immediately at the provider, then `git filter-repo` (or `bfg`) it
out of history.

#### `.env` block — copy from `.env.example` and fill in your keys

```ini
# =========================
# LLM / RESEARCH API KEYS
# =========================

# Used by OpenAI-powered agents:
# TradeAnalysisAgent, ModelReviewAgent, ReportAgent,
# RiskExplainerAgent, TradeJournalAgent, BacktestCriticAgent.
OPENAI_API_KEY=

# Used by Perplexity-powered agents:
# NewsAgent, MacroNewsAgent, StrategyResearchAgent.
PERPLEXITY_API_KEY=

# Optional future providers. Not required right now.
ANTHROPIC_API_KEY=
GEMINI_API_KEY=


# =========================
# AGENT PROVIDER ROUTING
# =========================

# Web/current research agents.
NEWS_AGENT_PROVIDER=perplexity
MACRO_NEWS_AGENT_PROVIDER=perplexity
STRATEGY_RESEARCH_AGENT_PROVIDER=perplexity

# Structured reasoning / reporting agents.
TRADE_ANALYSIS_AGENT_PROVIDER=openai
MODEL_REVIEW_AGENT_PROVIDER=openai
REPORT_AGENT_PROVIDER=openai
RISK_EXPLAINER_AGENT_PROVIDER=openai
TRADE_JOURNAL_AGENT_PROVIDER=openai
BACKTEST_CRITIC_AGENT_PROVIDER=openai

# Stats-only by default. Set to "openai" only if you want optional narrative polish.
MODEL_DRIFT_AGENT_PROVIDER=none

# Data quality should stay deterministic/code-based by default.
DATA_QUALITY_AGENT_PROVIDER=none


# =========================
# PERPLEXITY MODELS
# =========================

# Default Perplexity model for current market/news research.
PERPLEXITY_DEFAULT_MODEL=sonar-pro

# Fast/current research.
NEWS_AGENT_MODEL=sonar-pro
MACRO_NEWS_AGENT_MODEL=sonar-pro

# Heavier research. Use sparingly, such as weekly strategy research.
STRATEGY_RESEARCH_AGENT_MODEL=sonar-deep-research


# =========================
# OPENAI MODELS
# =========================

# OpenAI default for normal structured outputs.
# Keep this configurable because available model names may change.
OPENAI_DEFAULT_MODEL=gpt-4o-mini

# Stronger model for harder reviews/audits.
OPENAI_REVIEW_MODEL=gpt-4o

# Per-agent OpenAI model mapping. The ``${VAR}`` shorthand below is
# *documentation only* — pydantic-settings does not expand env vars.
# Leave a per-agent value empty / "none" / "${OPENAI_DEFAULT_MODEL}"
# to inherit the right default; set a concrete model name to override.
TRADE_ANALYSIS_AGENT_MODEL=${OPENAI_DEFAULT_MODEL}
TRADE_JOURNAL_AGENT_MODEL=${OPENAI_DEFAULT_MODEL}
REPORT_AGENT_MODEL=${OPENAI_DEFAULT_MODEL}
RISK_EXPLAINER_AGENT_MODEL=${OPENAI_DEFAULT_MODEL}

MODEL_REVIEW_AGENT_MODEL=${OPENAI_REVIEW_MODEL}
BACKTEST_CRITIC_AGENT_MODEL=${OPENAI_REVIEW_MODEL}

# Model drift should mostly be deterministic stats.
# Set MODEL_DRIFT_AGENT_PROVIDER=openai and this to an OpenAI model
# only if narrative summaries are needed.
MODEL_DRIFT_AGENT_MODEL=none

# DataQualityAgent should not use an LLM by default.
DATA_QUALITY_AGENT_MODEL=none
```

#### Safety guarantees (still enforced)

- LLM agents are advisory only. **No** agent or provider can call
  `execution/`, change risk caps, modify model thresholds, or
  auto-promote a model. The architectural-isolation test
  (`tests/test_agent_isolation.py`) fails the build if any module
  under `agents/` imports `execution/` or `risk/`.
- Failed provider calls (network, auth, schema) raise
  `ProviderError` → `LLMClientError`, which `BaseAgent.run()`
  catches and persists as a clean `AgentResult(schema_valid=False)`
  row. PAPER mode and the scheduler keep running.
- Missing API keys disable the affected agents only — never the
  whole bot. Every disable / construct event is logged with the
  agent name, provider, and reason, but **never** the key value.

### Autonomous-paper / research agents

The agents below are advisory-only inputs to specific workflows
rather than members of the EOD batch. They share the same provider
router and the same architectural-isolation guarantees.

- `MacroNewsAgent` (web-grounded). Researches today's macro / event
  calendar across `ENABLED_SYMBOLS` and emits a
  `MacroNewsAssessment(risk_level, affected_symbols, blocked_windows,
  key_events, sources, summary)`. The orchestrator method
  `run_macro_news()` flips the existing `high_risk_news_active` flag
  on `risk_level=="high"` or any populated `blocked_windows` —
  block-only, exactly like `NewsAgent`. Affected symbols are filtered
  against the operator's universe before reaching anything
  operational.

- `BacktestCriticAgent` (reasoning). Reviews a backtest summary plus
  the recent paper-mode metrics and emits a `BacktestCritique` with
  weak spots (time windows, symbols, confidence buckets, regimes).
  **Every recommendation is experiment-shaped** (hypothesis +
  experiment plan + risks). The schema has no field that could
  encode "change parameter X to Y" — that's intentional. Invoke via
  `orchestrator.run_backtest_critic(backtest_summary)` immediately
  after a backtest finishes.

- `ModelDriftAgent` (stats-first, optional LLM polish). Compares
  paper metrics (win rate, expectancy, profit factor, drawdown,
  false positive rate) against the model's training expectations and
  emits a `ModelDriftReport` with per-metric deltas + a single
  `severity` ∈ `none|watch|warn|alert`. `alert` flips
  `retrain_recommended=True`. Promotion still requires the
  deterministic walk-forward + `--promote-model` workflow — the
  agent only *advises*. Invoked from daily / weekly review code via
  `orchestrator.run_model_drift_review()`. The deterministic stats
  path runs even with no LLM configured.

- `StrategyResearchAgent` (web-grounded). Surfaces strategy / filter
  ideas worth backtesting, written as `StrategyExperimentIdea`
  records (title + hypothesis + experiment plan + risks +
  related_filters). Cannot suggest code or threshold changes — the
  schema enforces shape and the system prompt enforces tone.
  Invoked off-cycle via `orchestrator.run_strategy_research()`.

- `DataQualityAgent` (deterministic, no LLM). Pre-paper-loop
  pre-flight check. Detects empty feeds, stale feeds, missing
  candles, duplicate timestamps, geometrically impossible OHLCV,
  and aggregate data gaps. Returns a `DataQualityReport`; the
  operative field is `blocked_symbols`. Paper mode reads that list
  and refuses to start the per-symbol loop on any blocked symbol.
  Direct entry point: `DataQualityAgent().run_with_feeds(...)` or
  `orchestrator.run_data_quality_check(feeds_by_symbol={...})`.

**Workflow hooks**

```python
# Pre-paper data-quality gate
report = orchestrator.run_data_quality_check(feeds_by_symbol={
    "MES": mes_df, "MNQ": mnq_df, "MGC": mgc_df,
})
if report and report.blocked_symbols:
    log.warning("paper.skipping_symbols", blocked=report.blocked_symbols)

# Pre-session macro news (sets high_risk_news_active block-only)
orchestrator.run_macro_news()

# After backtest
orchestrator.run_backtest_critic(backtest_summary=result.to_dict())

# Daily / weekly review
drift = orchestrator.run_model_drift_review()
if drift and drift.retrain_recommended:
    log.warning(
        "model.drift_alert",
        severity=drift.severity,
        reason=drift.reason,
    )

# Off-cycle ideation
ideas = orchestrator.run_strategy_research(
    backtest_summary=result.to_dict(),
)
```

All five agents follow the same safety rules as the EOD batch:
they cannot import `execution/` or `risk/`, they cannot place
trades or modify any risk setting, and `tests/test_agent_isolation.py`
fails the build if any new agent breaks that contract.

**Citations.** When a Perplexity-routed agent runs, the underlying
`PerplexityProvider` returns a list of `Citation(url, title, snippet)`
objects alongside the text. The MVP surfaces them via
`ProviderLLMClient.last_citations`; a future iteration will pin them
to the persisted `agent_outputs` row so daily reports can render
"sources the news agent read".

**API keys are secrets.** Never commit any of `OPENAI_API_KEY`,
`PERPLEXITY_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` to
the repo. They belong in `.env` (which is `.gitignore`d) or in your
runtime secret store. The bot's structured logger never emits the
keys, but provider errors include upstream HTTP bodies — review them
before pasting into a chat / issue.

**Safety properties (still enforced):**

- LLM agents cannot place trades or change any risk setting,
  threshold, or strategy parameter. The new providers do not change
  this — the architectural isolation test
  (`tests/test_agent_isolation.py`) scans every file under `agents/`,
  including `agents/providers/`, for any import of `execution/` or
  `risk/` and refuses the build if one is added.
- Models are **not** auto-promoted by any agent. Promotion still
  requires the deterministic walk-forward workflow + an explicit
  `--promote-model` invocation.
- Webhook signals (`/webhooks/tradingview`) and the multi-symbol
  paper loop both run the existing risk engine before any paper
  trade. Agents are advisory inputs — they cannot route a fill.

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
python -m app.main --retrain-from-feedback \
  --model-name vwap_ema_pullback_lr \
  --min-feedback-rows 100             # build dataset + train candidate + comparison report
python -m app.main --promote-model VERSION \
  --model-name vwap_ema_pullback_lr_candidate   # only if PromotionDecision says so
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
python -m app.main \
  --retrain-from-feedback \
  --model-name vwap_ema_pullback_lr \
  --min-feedback-rows 100
```

`--model-name` is **required** — it's the incumbent's registry name.
The candidate is saved as `<model-name>_candidate` unless you override
it with `--candidate-model-name`.

This:

1. Builds the feedback dataset from `closed_trades` ⨯
   `feature_snapshots` ⨯ `model_predictions` ⨯ `trade_mistake_tags`,
   then writes it to `data/reports/feedback/feedback_dataset.{csv,json}`.
2. Refuses to train if there are fewer than `--min-feedback-rows`
   (default **100**, or whatever `FEEDBACK_MIN_ROWS` env var sets) rows
   with full feature vectors. Exits with code `4`.
3. Splits **strictly chronologically** by entry timestamp
   (70% train / 15% val / 15% test). No shuffling, ever.
4. Trains the kind selected by `--feedback-model-kind` (default
   `logreg`); calibrates on val.
5. If LightGBM is installed it also trains the *other* kind for
   metrics-only comparison — never saved as the candidate artifact.
6. Evaluates on the holdout test split: accuracy, precision, recall,
   false-positive rate, ROC-AUC, **expectancy per trade**,
   **profit factor**, **drawdown proxy**, and **calibration MAE** —
   all on the trades the candidate model would have approved.
7. Saves the candidate to
   `data/models/<model-name>_candidate/<version>/` with
   `metadata.json` carrying `candidate=true`, `source=feedback`,
   `model_kind`, realized trade metrics, mistake-tag counts, and the
   chronological split ranges.
8. Writes a Markdown promotion report under
   `data/reports/feedback/promotion_*.md`.

**The candidate is never automatically promoted.** Mistake tags are
stored in metadata as a diagnostic signal; they are *not* used as
labels unless you opt in with `--use-mistake-tags-as-label`.

Useful overrides:

```bash
# Override min rows (e.g. for an early-validation pilot)
python -m app.main --retrain-from-feedback \
  --model-name vwap_ema_pullback_lr \
  --min-feedback-rows 60

# Pick a custom candidate name
python -m app.main --retrain-from-feedback \
  --model-name vwap_ema_pullback_lr \
  --candidate-model-name vwap_ema_pullback_lr_2026_q2

# Train a LightGBM candidate (requires `pip install lightgbm`)
python -m app.main --retrain-from-feedback \
  --model-name vwap_ema_pullback_lr \
  --feedback-model-kind lightgbm

# Treat mistake-tagged trades as negative labels (advanced)
python -m app.main --retrain-from-feedback \
  --model-name vwap_ema_pullback_lr \
  --use-mistake-tags-as-label
```

The previous flag names `--feedback-min-rows` and
`--feedback-use-mistake-tags` are kept as **aliases** so existing
operator scripts and earlier README revisions still work.

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
- `OPENAI_API_KEY` / `PERPLEXITY_API_KEY` / `ANTHROPIC_API_KEY` /
  `GEMINI_API_KEY` — provider keys. Each is independent; agents
  routed to a provider with no key are gracefully disabled.
- `LLM_MODEL` — legacy default for OpenAI when `OPENAI_MODEL` is unset
  (default `gpt-4o-mini`).
- `OPENAI_MODEL` / `PERPLEXITY_MODEL` / `ANTHROPIC_MODEL` /
  `GEMINI_MODEL` — per-provider model overrides.
- `NEWS_AGENT_PROVIDER` / `MACRO_NEWS_AGENT_PROVIDER` /
  `STRATEGY_RESEARCH_AGENT_PROVIDER` /
  `TRADE_ANALYSIS_AGENT_PROVIDER` /
  `MODEL_REVIEW_AGENT_PROVIDER` / `REPORT_AGENT_PROVIDER` /
  `RISK_EXPLAINER_AGENT_PROVIDER` / `TRADE_JOURNAL_AGENT_PROVIDER` —
  per-agent provider routing. Set to `none` to disable an agent.
- `LLM_TIMEOUT_SECONDS` — per-call timeout (default 30).
- `AGENTS_RUN_AT_EOD` — whether the EOD scheduler job triggers the agents.
- `NEWS_CHECK_LOCAL_TIME` — pre-session NewsAgent cron time (default `09:25`).
- `FEEDBACK_MIN_ROWS` — minimum closed trades (with feature snapshots)
  required before `--retrain-from-feedback` will train a candidate
  (default 100). Below this the trainer refuses to run.
- `FEEDBACK_USE_MISTAKE_TAGS_AS_LABEL` — when true, derive candidate
  labels from mistake tags (default false; tags are metadata only).
- `TRADINGVIEW_WEBHOOK_SECRET` — shared secret for the
  `POST /webhooks/tradingview` endpoint. When unset, the secret check
  is skipped (do **not** expose the endpoint publicly without one).
- `WEBHOOK_DEFAULT_STOP_TICKS` / `WEBHOOK_DEFAULT_TARGET_TICKS` —
  fallback stop / target distance (in instrument ticks) used when a
  TradingView alert payload omits `stop` / `target` (defaults 20 / 40).

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
agents/providers/       Multi-provider LLM router (OpenAI/Perplexity/Anthropic/Gemini)
analysis/               post-trade analyzer, mistake classifier, pattern miner,
                        improvement suggester, feedback dataset, promotion
notifications/          Discord webhook + rate-limited dispatcher
paper/                  PaperTradingLoop (Day 5)
reports/                daily report, trade journal, backtest report,
                        per-trade post-mortem, daily mistake digest
scheduler/              APScheduler service + market hours
storage/                SQLAlchemy engine + ORM tables
webhook/                lightweight TradingView payload validator
webhooks/               FastAPI ingest app for TradingView alerts
tests/                  pytest tests (430+ covering all the above)
```
