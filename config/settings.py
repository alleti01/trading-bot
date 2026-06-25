"""Global application settings.

All configuration flows through a single Pydantic ``Settings`` object loaded
from environment variables (and an optional ``.env`` file). The two safety
properties this module guarantees:

1. **LIVE-mode lockout.** If ``MODE=LIVE`` and ``LIVE_ADAPTER_CONFIRMED`` is
   not explicitly true, settings construction raises immediately. The bot
   cannot boot into LIVE mode by accident.
2. **Strict validation.** Times, timezones, ranges, etc. are validated up-
   front so misconfiguration fails fast at startup, not mid-trade.
"""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Annotated, Literal, Optional
from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Mode = Literal["BACKTEST", "TRAIN", "PAPER", "LIVE"]
MarketType = Literal["futures", "crypto", "equity", "option"]
WorkflowExecutionMode = Literal["DRY_RUN", "PAPER", "LIVE"]
BrokerProvider = Literal["mock", "alpaca", "tradovate"]
DefaultOrderType = Literal["limit", "market"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Identity ------------------------------------------------------
    APP_NAME: str = "tradeify-bot"
    APP_VERSION: str = "0.1.0"

    # ---- Mode & instrument --------------------------------------------
    MODE: Mode = "PAPER"
    INSTRUMENT: str = "MES"
    MARKET_TYPE: MarketType = "futures"
    TIMEZONE: str = "America/New_York"

    # ---- Multi-symbol universe ----------------------------------------
    # Comma-separated symbols paper mode and backtest will scan. Single
    # entry == single-symbol behavior (back-compat default).
    # ``NoDecode`` keeps pydantic-settings from JSON-parsing the env var
    # before our validator can split on commas.
    ENABLED_SYMBOLS: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["MES"]
    )
    # Optional override; defaults to INSTRUMENT in code. Used by
    # reports + the model registry as the "headline" symbol.
    PRIMARY_SYMBOL: Optional[str] = None
    # Caps the orchestrator enforces across all per-symbol loops:
    MAX_ACTIVE_SYMBOLS: int = Field(default=1, ge=1)
    MAX_TRADES_PER_SYMBOL_PER_DAY: int = Field(default=4, ge=1)
    MAX_TOTAL_TRADES_PER_DAY: int = Field(default=8, ge=1)

    # ---- LIVE lockout --------------------------------------------------
    LIVE_ADAPTER_CONFIRMED: bool = False

    # ---- Trading windows (HH:MM in TIMEZONE) ---------------------------
    TRADING_WINDOW_START: str = "09:30"
    TRADING_WINDOW_END: str = "15:55"
    FORCE_FLAT_TIME: str = "15:55"

    # ---- Risk ----------------------------------------------------------
    MAX_TRADES_PER_DAY: int = Field(default=8, ge=1)
    MAX_DAILY_LOSS: float = Field(default=500.0, ge=0.0)
    MAX_DAILY_PROFIT: float = Field(default=1500.0, ge=0.0)
    MAX_POSITION_SIZE: int = Field(default=1, ge=1)
    RISK_PER_TRADE: float = Field(default=100.0, ge=0.0)
    MAX_OPEN_POSITIONS: int = Field(default=1, ge=1)
    COOLDOWN_AFTER_LOSS_MINUTES: int = Field(default=5, ge=0)
    COOLDOWN_AFTER_LARGE_WIN_MINUTES: int = Field(default=15, ge=0)
    LARGE_WIN_THRESHOLD: float = Field(default=200.0, ge=0.0)
    MAX_HOLD_BARS: int = Field(default=20, ge=1)

    # ---- Strategies -----------------------------------------------------
    # Comma-separated list of strategy names that paper mode runs. Single
    # strategy is the default; adding more enables multi-strategy paper
    # mode through the strategies.registry conflict resolver.
    #
    # ``NoDecode`` tells pydantic-settings *not* to JSON-parse this env
    # var before ``_split_enabled_strategies`` sees it, so operators can
    # write ``ENABLED_STRATEGIES=a,b`` instead of ``["a","b"]``.
    ENABLED_STRATEGIES: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["vwap_ema_pullback"]
    )

    # ---- Model ---------------------------------------------------------
    CONFIDENCE_THRESHOLD: float = 0.60

    # ---- Retrain-from-feedback ----------------------------------------
    # Minimum number of closed paper trades (with feature snapshots)
    # before the candidate trainer will run. Below this, retraining
    # exits with a clear error rather than fitting on noise.
    FEEDBACK_MIN_ROWS: int = Field(default=100, ge=10)
    # If true, derive labels from MistakeClassifier tags (label=0 for
    # any trade tagged with a mistake) instead of raw PnL > 0. Mistake
    # tags are otherwise stored as metadata only — never as the label.
    FEEDBACK_USE_MISTAKE_TAGS_AS_LABEL: bool = False

    # ---- Costs ---------------------------------------------------------
    # Futures: tick-based slippage + flat per-contract commission.
    SLIPPAGE_TICKS: float = Field(default=1.0, ge=0.0)
    COMMISSION_PER_CONTRACT: float = Field(default=1.50, ge=0.0)
    # Crypto: basis-point slippage + basis-point fee. Independent knobs so
    # ``MARKET_TYPE=crypto`` users don't silently inherit MES defaults.
    CRYPTO_SLIPPAGE_BPS: float = Field(default=1.0, ge=0.0)
    CRYPTO_FEE_BPS: float = Field(default=5.0, ge=0.0)

    # ---- Compliance ----------------------------------------------------
    CONSISTENCY_LIMIT_PERCENT: float = Field(default=30.0, gt=0.0, le=100.0)

    # ---- LLM agents ----------------------------------------------------
    ENABLE_LLM_AGENTS: bool = False
    OPENAI_API_KEY: Optional[SecretStr] = None
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TIMEOUT_SECONDS: int = Field(default=30, ge=1, le=300)
    AGENTS_RUN_AT_EOD: bool = True
    NEWS_CHECK_LOCAL_TIME: str = "09:25"

    # ---- Multi-provider LLM keys --------------------------------------
    # Each provider is optional; a missing key disables every agent
    # routed to that provider (the rest still run). The bot never
    # auto-routes a webhook signal or a trade through these — they are
    # advisory-only inputs to the existing agent layer.
    PERPLEXITY_API_KEY: Optional[SecretStr] = None
    ANTHROPIC_API_KEY: Optional[SecretStr] = None
    GEMINI_API_KEY: Optional[SecretStr] = None

    # ---- Per-provider default models ---------------------------------
    # Plain "default for the whole provider" — used when an agent has
    # no per-agent override. ``OPENAI_MODEL`` (legacy) and
    # ``OPENAI_DEFAULT_MODEL`` are aliases; the router prefers
    # ``OPENAI_DEFAULT_MODEL`` when set.
    OPENAI_MODEL: Optional[str] = None
    OPENAI_DEFAULT_MODEL: str = "gpt-4o-mini"
    # Stronger model used by review/audit agents (model_review,
    # backtest_critic). Operators can change this string when the
    # available model catalog changes.
    OPENAI_REVIEW_MODEL: str = "gpt-4o"
    PERPLEXITY_MODEL: str = "sonar"
    PERPLEXITY_DEFAULT_MODEL: str = "sonar-pro"
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-latest"
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # ---- Per-agent provider routing ----------------------------------
    # The router reads these to decide which provider executes each
    # advisory agent. Web-grounded research (news, macro, strategy)
    # defaults to Perplexity; reasoning/summarization defaults to
    # OpenAI. Set to "none" / "disabled" / "off" to switch an agent
    # off without removing it from the orchestrator. Validators below
    # reject any other value with a clear error so a typo cannot
    # silently disable a critical agent.
    NEWS_AGENT_PROVIDER: str = "perplexity"
    MACRO_NEWS_AGENT_PROVIDER: str = "perplexity"
    STRATEGY_RESEARCH_AGENT_PROVIDER: str = "perplexity"
    TRADE_ANALYSIS_AGENT_PROVIDER: str = "openai"
    MODEL_REVIEW_AGENT_PROVIDER: str = "openai"
    REPORT_AGENT_PROVIDER: str = "openai"
    RISK_EXPLAINER_AGENT_PROVIDER: str = "openai"
    TRADE_JOURNAL_AGENT_PROVIDER: str = "openai"
    # ``BacktestCriticAgent`` always runs an LLM (default OpenAI).
    # ``ModelDriftAgent`` runs deterministic stats and uses the LLM
    # only for an optional narrative — default ``"none"`` keeps the
    # bot offline-clean. ``DataQualityAgent`` is fully deterministic.
    BACKTEST_CRITIC_AGENT_PROVIDER: str = "openai"
    MODEL_DRIFT_AGENT_PROVIDER: str = "none"
    DATA_QUALITY_AGENT_PROVIDER: str = "none"

    # ---- Per-agent model overrides -----------------------------------
    # Operator can override which exact model each agent uses. ``None``
    # / empty / ``${OPENAI_DEFAULT_MODEL}`` / ``${OPENAI_REVIEW_MODEL}``
    # / ``${PERPLEXITY_DEFAULT_MODEL}`` are all interpreted as "inherit
    # the right default for this agent" — see ``model_for_agent``.
    NEWS_AGENT_MODEL: Optional[str] = None
    MACRO_NEWS_AGENT_MODEL: Optional[str] = None
    # Heavier research model; default ``sonar-deep-research`` is
    # spec'd for weekly use (slow / expensive). Operators may want to
    # downgrade to ``sonar-pro`` for daily runs.
    STRATEGY_RESEARCH_AGENT_MODEL: Optional[str] = "sonar-deep-research"
    TRADE_ANALYSIS_AGENT_MODEL: Optional[str] = None
    MODEL_REVIEW_AGENT_MODEL: Optional[str] = None
    REPORT_AGENT_MODEL: Optional[str] = None
    RISK_EXPLAINER_AGENT_MODEL: Optional[str] = None
    TRADE_JOURNAL_AGENT_MODEL: Optional[str] = None
    BACKTEST_CRITIC_AGENT_MODEL: Optional[str] = None
    # Default ``"none"`` — deterministic stats only. Set to a real
    # model name AND set ``MODEL_DRIFT_AGENT_PROVIDER=openai`` to
    # enable the optional narrative polish.
    MODEL_DRIFT_AGENT_MODEL: Optional[str] = "none"
    # ``DataQualityAgent`` never calls an LLM by default.
    DATA_QUALITY_AGENT_MODEL: Optional[str] = "none"

    # ---- Autonomous workflows (paper-first orchestration) --------------
    # Separate from ``MODE`` — workflows can run in DRY_RUN while the
    # core bot is in PAPER. LIVE is always refused by the workflow runner.
    WORKFLOW_EXECUTION_MODE: WorkflowExecutionMode = "DRY_RUN"
    WORKFLOW_TIMEZONE: str = "America/New_York"
    WORKFLOW_WEEKDAYS_ONLY: bool = True
    WORKFLOW_GIT_COMMIT: bool = False
    WORKFLOW_GIT_PUSH: bool = False
    # When true, pre-market may call MacroNewsAgent (requires LLM keys).
    PERPLEXITY_ENABLED: bool = False
    # Master switch for workflow-driven paper order placement.
    AUTONOMOUS_TRADING_ENABLED: bool = False
    # Safety rail: autonomous execution only when execution mode is PAPER.
    AUTONOMOUS_PAPER_ONLY: bool = True
    # Memory directory for strategy/research/trade logs.
    WORKFLOW_MEMORY_DIR: Path = Path("./memory")
    # Optional model that gates workflow entries. When set, the signal
    # engine scores each strategy setup with this model and only trades
    # approved setups. When unset, the workflow trades raw strategy
    # signals (strategy-only mode).
    WORKFLOW_MODEL_NAME: Optional[str] = None
    WORKFLOW_MODEL_VERSION: str = "latest"
    # When true, a symbol with no usable OHLCV data / no setup is skipped
    # (safe). This is always the behavior; the flag documents intent.
    WORKFLOW_REQUIRE_SIGNAL: bool = True

    # ---- Equity position sizing ---------------------------------------
    # Workflow equity trades size by risk: shares = RISK_PER_TRADE /
    # (stop distance per share), then capped by share count and notional
    # dollars so a $50 stock and a $900 stock risk the same and never
    # exceed buying power. Set USE_RISK_BASED_SIZING=false to fall back
    # to the fixed MAX_POSITION_SIZE share count.
    USE_RISK_BASED_SIZING: bool = True
    MAX_SHARES_PER_TRADE: int = Field(default=100, ge=1)
    MAX_NOTIONAL_PER_TRADE: float = Field(default=10_000.0, gt=0.0)

    # ---- Strategy time-of-day filter ----------------------------------
    # Skip VWAP/EMA setups within N minutes of the RTH open/close. The
    # open/close are the noisiest periods; skipping them often helps the
    # tight-stop config. 0 = no filter (default).
    STRATEGY_SKIP_OPEN_MINUTES: float = Field(default=0.0, ge=0)
    STRATEGY_SKIP_CLOSE_MINUTES: float = Field(default=0.0, ge=0)

    # ---- Continuous intraday loop + dynamic universe -------------------
    # The intraday loop re-scans the universe every N minutes during the
    # trading window and places bracket orders on approved setups.
    WORKFLOW_SCAN_INTERVAL_MINUTES: int = Field(default=5, ge=1, le=60)
    # When true, the loop refreshes bars from Alpaca before each scan.
    WORKFLOW_REFRESH_DATA_EACH_SCAN: bool = True
    # Long-only ignores short setups (avoids short-borrow complications).
    WORKFLOW_LONG_ONLY: bool = True
    # Dynamic universe: expand the scan set beyond ENABLED_SYMBOLS using
    # the vetted liquid allowlist (and optional research-agent ranking).
    # The LLM can only prioritize allowlist names — never invent tickers.
    WORKFLOW_DYNAMIC_UNIVERSE: bool = False
    # Hard cap on how many symbols the loop scans per cycle.
    WORKFLOW_MAX_UNIVERSE: int = Field(default=15, ge=1, le=100)
    # When true, the research agent (if enabled) proposes which allowlist
    # names to prioritize today. Falls back to deterministic allowlist
    # order when agents are off.
    WORKFLOW_AGENT_WATCHLIST: bool = False

    # ---- Broker adapter (paper/demo only) ------------------------------
    # ``BROKER_PROVIDER`` selects the integration used by the workflow
    # layer for order placement. ``mock`` is the default — every method
    # returns simulated JSON without touching the network. ``tradovate``
    # is permitted ONLY when the demo URL + ``TRADOVATE_DEMO=true`` are
    # both set (enforced by :class:`TradovateDemoClient`).
    BROKER_PROVIDER: BrokerProvider = "mock"
    DEFAULT_ORDER_TYPE: DefaultOrderType = "limit"
    BROKER_REQUEST_TIMEOUT_SECONDS: float = Field(default=15.0, gt=0.0)

    # ---- Tradovate (demo / simulation only) ----------------------------
    # The bot refuses to call Tradovate unless ``TRADOVATE_DEMO=true``
    # AND the base URL points at ``demo.tradovateapi.com``. There is no
    # live path: live execution is locked at the workflow runner and
    # again inside the broker router.
    TRADOVATE_USERNAME: Optional[str] = None
    TRADOVATE_PASSWORD: Optional[SecretStr] = None
    TRADOVATE_APP_ID: Optional[str] = None
    TRADOVATE_APP_VERSION: str = "1.0.0"
    TRADOVATE_CLIENT_ID: Optional[str] = None
    TRADOVATE_CLIENT_SECRET: Optional[SecretStr] = None
    TRADOVATE_DEMO: bool = True
    TRADOVATE_BASE_URL: str = "https://demo.tradovateapi.com/v1"
    TRADOVATE_WS_URL: str = "wss://demo.tradovateapi.com/v1/websocket"

    # ---- Alpaca (PAPER / sandbox only) ---------------------------------
    # Alpaca is the current paper-automation broker because Tradovate
    # demo credentials require a funded account. The adapter refuses
    # to act unless ``ALPACA_PAPER=true`` AND the base URL points at
    # ``paper-api.alpaca.markets``. Live execution is locked.
    ALPACA_API_KEY: Optional[SecretStr] = None
    ALPACA_SECRET_KEY: Optional[SecretStr] = None
    ALPACA_PAPER: bool = True
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"

    # ---- Options trading (paper/simulation only) ----------------------
    # Directional/spread/condor options on top of the VWAP/EMA signal on
    # the underlying. Execution is Alpaca paper or mock only — never live.
    OPTIONS_ENABLED: bool = False
    # atm_directional | vertical_spread | iron_condor
    OPTIONS_STRATEGY: str = "atm_directional"
    OPTIONS_DEFAULT_DTE: int = Field(default=30, ge=0)
    OPTIONS_MIN_DTE: int = Field(default=7, ge=0)
    OPTIONS_MAX_DTE: int = Field(default=60, ge=1)
    OPTIONS_TARGET_DELTA: float = Field(default=0.50, gt=0.0, le=1.0)
    OPTIONS_SPREAD_WIDTH_STRIKES: int = Field(default=1, ge=1)
    OPTIONS_MAX_PREMIUM_PER_TRADE: float = Field(default=500.0, ge=0.0)
    OPTIONS_MAX_OPEN_POSITIONS: int = Field(default=5, ge=1)
    OPTIONS_AUTO_CLOSE_DTE: int = Field(default=5, ge=0)
    OPTIONS_AUTO_ROLL: bool = True
    OPTIONS_ROLL_DTE_TRIGGER: int = Field(default=7, ge=0)
    OPTIONS_PROFIT_TARGET_PCT: float = 50.0
    OPTIONS_STOP_LOSS_PCT: float = -50.0
    OPTIONS_QTY_PER_TRADE: int = Field(default=1, ge=1)
    OPTIONS_STATE_PATH: str = "data/options/positions.json"
    # Underlyings the options layer is allowed to trade.
    OPTIONS_ENABLED_UNDERLYINGS: str = "SPY,QQQ,AAPL,MSFT"

    # ---- Parallel paper evaluation -------------------------------------
    # When ``ENABLE_PARALLEL_PAPER=true`` and ``--start-parallel-paper``
    # is passed on the CLI, the runner launches one evaluation track per
    # entry in ``PARALLEL_BROKERS``. Each track gets its own broker,
    # symbol universe, state file, and report directory. Brokers do NOT
    # share positions or trade state.
    ENABLE_PARALLEL_PAPER: bool = False
    PARALLEL_BROKERS: str = "futures_sim,alpaca"
    ALPACA_ENABLED_SYMBOLS: str = "SPY,QQQ,AAPL,MSFT"
    FUTURES_SIM_ENABLED_SYMBOLS: str = "MES,MNQ,MGC,MCL"
    ALPACA_EVALUATION_ID: str = "alpaca_2week_test"
    FUTURES_SIM_EVALUATION_ID: str = "futures_sim_2week_test"
    ALPACA_STATE_PATH: str = "data/paper/alpaca_state.json"
    FUTURES_SIM_STATE_PATH: str = "data/paper/futures_sim_state.json"

    # ---- Notifications -------------------------------------------------
    DISCORD_WEBHOOK_URL: Optional[SecretStr] = None

    # ---- TradingView webhook ------------------------------------------
    # Optional shared secret for the ``POST /webhooks/tradingview`` endpoint.
    # When set, every incoming webhook MUST carry a matching ``secret``
    # field (or a same-value header) or the request is rejected before it
    # touches the trading pipeline.
    #
    # Operational note: TradingView alert messages travel over plain HTTP
    # to whatever URL you configure, so the secret is only useful when
    # the bot is behind HTTPS (ngrok / Cloudflare Tunnel / a real LB).
    # Treat it as "this signal is from my alert", not "this signal is
    # cryptographically authenticated."
    TRADINGVIEW_WEBHOOK_SECRET: Optional[SecretStr] = None
    # Default stop/target distance in instrument ticks when a webhook
    # payload omits ``stop`` / ``target``. The repo's risk engine
    # requires a stop, so we synthesize one rather than refusing the
    # signal outright. Operators that care about stop placement should
    # include explicit ``stop`` / ``target`` in the alert message.
    WEBHOOK_DEFAULT_STOP_TICKS: int = Field(default=20, ge=1)
    WEBHOOK_DEFAULT_TARGET_TICKS: int = Field(default=40, ge=1)

    # ---- Paper service (Day 5) ----------------------------------------
    BAR_INTERVAL_SECONDS: int = Field(default=60, ge=1)
    ROLLING_WINDOW_BARS: int = Field(default=500, ge=50)
    PAPER_CSV_PATH: Optional[str] = None
    HEARTBEAT_LOCAL_TIME: str = "08:00"

    # ---- Storage -------------------------------------------------------
    DATABASE_URL: str = "sqlite:///./data/bot.db"

    # ---- Logging -------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False

    # ---- Filesystem layout --------------------------------------------
    DATA_DIR: Path = Path("./data")
    HISTORICAL_DATA_DIR: Path = Path("./data/historical")
    MODELS_DIR: Path = Path("./data/models")
    REPORTS_DIR: Path = Path("./data/reports")
    LOGS_DIR: Path = Path("./logs")

    # ----------------------------------------------------------------------
    # Validators
    # ----------------------------------------------------------------------
    @field_validator(
        "TRADING_WINDOW_START",
        "TRADING_WINDOW_END",
        "FORCE_FLAT_TIME",
        "HEARTBEAT_LOCAL_TIME",
        "NEWS_CHECK_LOCAL_TIME",
    )
    @classmethod
    def _validate_time(cls, v: str) -> str:
        try:
            time.fromisoformat(v)
        except ValueError as e:
            raise ValueError(f"Invalid time '{v}', expected HH:MM") from e
        return v

    @field_validator("TIMEZONE", "WORKFLOW_TIMEZONE")
    @classmethod
    def _validate_tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except Exception as e:
            raise ValueError(f"Invalid timezone '{v}'") from e
        return v

    @field_validator("BROKER_PROVIDER", mode="before")
    @classmethod
    def _normalize_broker_provider(cls, v: object) -> str:
        s = str(v or "mock").strip().lower()
        allowed = {"mock", "alpaca", "tradovate"}
        if s not in allowed:
            raise ValueError(
                f"Invalid BROKER_PROVIDER '{v}'. Allowed: {sorted(allowed)}."
            )
        return s

    @field_validator("DEFAULT_ORDER_TYPE", mode="before")
    @classmethod
    def _normalize_default_order_type(cls, v: object) -> str:
        s = str(v or "limit").strip().lower()
        if s not in {"limit", "market"}:
            raise ValueError(
                f"Invalid DEFAULT_ORDER_TYPE '{v}'. Allowed: ['limit', 'market']."
            )
        return s

    @field_validator("OPTIONS_STRATEGY", mode="before")
    @classmethod
    def _normalize_options_strategy(cls, v: object) -> str:
        s = str(v or "atm_directional").strip().lower()
        allowed = {"atm_directional", "vertical_spread", "iron_condor"}
        if s not in allowed:
            raise ValueError(
                f"Invalid OPTIONS_STRATEGY '{v}'. Allowed: {sorted(allowed)}."
            )
        return s

    @field_validator("WORKFLOW_EXECUTION_MODE", mode="before")
    @classmethod
    def _normalize_workflow_execution_mode(cls, v: object) -> str:
        s = str(v or "DRY_RUN").strip().upper()
        allowed = {"DRY_RUN", "PAPER", "LIVE"}
        if s not in allowed:
            raise ValueError(
                f"Invalid WORKFLOW_EXECUTION_MODE '{v}'. "
                f"Allowed: {sorted(allowed)}."
            )
        return s

    @field_validator("ENABLED_STRATEGIES", mode="before")
    @classmethod
    def _split_enabled_strategies(cls, v):
        # Accept both JSON-style ``["a","b"]`` (pydantic-settings default)
        # and the operator-friendly comma-separated env form ``"a,b"``.
        if isinstance(v, str):
            v = [s.strip() for s in v.split(",") if s.strip()]
        if not v:
            v = ["vwap_ema_pullback"]
        return v

    @field_validator("ENABLED_SYMBOLS", mode="before")
    @classmethod
    def _split_enabled_symbols(cls, v):
        if isinstance(v, str):
            v = [s.strip() for s in v.split(",") if s.strip()]
        if not v:
            return v
        return [str(s).strip().upper() for s in v if str(s).strip()]

    @field_validator("CONFIDENCE_THRESHOLD")
    @classmethod
    def _validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("CONFIDENCE_THRESHOLD must be in [0, 1]")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid LOG_LEVEL '{v}'")
        return v

    @field_validator(
        "NEWS_AGENT_PROVIDER",
        "MACRO_NEWS_AGENT_PROVIDER",
        "STRATEGY_RESEARCH_AGENT_PROVIDER",
        "TRADE_ANALYSIS_AGENT_PROVIDER",
        "MODEL_REVIEW_AGENT_PROVIDER",
        "REPORT_AGENT_PROVIDER",
        "RISK_EXPLAINER_AGENT_PROVIDER",
        "TRADE_JOURNAL_AGENT_PROVIDER",
        "BACKTEST_CRITIC_AGENT_PROVIDER",
        "MODEL_DRIFT_AGENT_PROVIDER",
        "DATA_QUALITY_AGENT_PROVIDER",
    )
    @classmethod
    def _normalize_agent_provider(cls, v: str) -> str:
        # Lowercase + validate. ``""`` / unset is treated as ``"none"``
        # (agent disabled) so a typo like ``MODEL_DRIFT_AGENT_PROVIDER=``
        # never falls through to "openai" silently.
        s = (v or "").strip().lower()
        if s == "":
            s = "none"
        allowed = {
            "openai",
            "perplexity",
            "anthropic",
            "gemini",
            "none",
            "off",
            "disabled",
            "false",
        }
        if s not in allowed:
            raise ValueError(
                f"Invalid agent provider '{v}'. Allowed: "
                f"{sorted(allowed)}. Use 'none' to disable an agent."
            )
        return s

    @model_validator(mode="after")
    def _refuse_live_without_adapter(self) -> "Settings":
        if self.MODE == "LIVE" and not self.LIVE_ADAPTER_CONFIRMED:
            raise ValueError(
                "LIVE mode is locked. To enable real-money trading you must:\n"
                "  1. Implement a real broker/exchange adapter "
                "(execution/futures_adapter_placeholder.py or crypto_adapter_placeholder.py).\n"
                "  2. Set LIVE_ADAPTER_CONFIRMED=true in your environment.\n"
                "Until both are done the bot refuses to boot in LIVE mode."
            )
        return self

    @model_validator(mode="after")
    def _trading_window_order(self) -> "Settings":
        start = time.fromisoformat(self.TRADING_WINDOW_START)
        end = time.fromisoformat(self.TRADING_WINDOW_END)
        if start >= end:
            raise ValueError("TRADING_WINDOW_START must be earlier than TRADING_WINDOW_END")
        flat = time.fromisoformat(self.FORCE_FLAT_TIME)
        if flat < start:
            raise ValueError("FORCE_FLAT_TIME must not be before TRADING_WINDOW_START")
        return self

    # ----------------------------------------------------------------------
    # Convenience accessors
    # ----------------------------------------------------------------------
    def trading_window_start_time(self) -> time:
        return time.fromisoformat(self.TRADING_WINDOW_START)

    def trading_window_end_time(self) -> time:
        return time.fromisoformat(self.TRADING_WINDOW_END)

    def force_flat_time(self) -> time:
        return time.fromisoformat(self.FORCE_FLAT_TIME)

    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.TIMEZONE)

    def workflow_tz(self) -> ZoneInfo:
        return ZoneInfo(self.WORKFLOW_TIMEZONE)

    # ----------------------------------------------------------------------
    # Agent provider/model accessors
    # ----------------------------------------------------------------------
    # Tokens we treat as "use the per-provider default" inside per-agent
    # model fields. ``${...}`` shorthand exists so ``.env.example`` can
    # show that a per-agent model "ties to" a default without actually
    # requiring pydantic-settings env-var interpolation (which does not
    # exist by default). Plain empty / "none" / None all behave the
    # same way.
    _MODEL_DEFAULT_TOKENS = {
        "",
        "none",
        "off",
        "disabled",
        "false",
        "${openai_default_model}",
        "${openai_review_model}",
        "${perplexity_default_model}",
    }

    # Agents that ride the heavier review / audit OpenAI model when no
    # explicit per-agent override is set. Everything else (writing,
    # narrative summaries) takes the default OpenAI model.
    _OPENAI_REVIEW_AGENTS = frozenset({"model_review", "backtest_critic"})

    def _resolve_default_model(self, agent_name: str, provider: str) -> Optional[str]:
        """Return the provider-default model for an agent, or ``None``
        when the agent is disabled / deterministic."""
        if provider in {"none", "off", "disabled", "false", ""}:
            return None
        if provider == "openai":
            if agent_name in self._OPENAI_REVIEW_AGENTS:
                return self.OPENAI_REVIEW_MODEL
            # Honor the legacy ``OPENAI_MODEL`` if explicitly set,
            # otherwise the new ``OPENAI_DEFAULT_MODEL``.
            return self.OPENAI_MODEL or self.OPENAI_DEFAULT_MODEL
        if provider == "perplexity":
            return self.PERPLEXITY_DEFAULT_MODEL or self.PERPLEXITY_MODEL
        if provider == "anthropic":
            return self.ANTHROPIC_MODEL
        if provider == "gemini":
            return self.GEMINI_MODEL
        return None

    def model_for_agent(self, agent_name: str) -> Optional[str]:
        """Resolve the model an agent should run with.

        - If the per-agent ``*_AGENT_MODEL`` override is set to a real
          string, use it.
        - If unset / empty / ``"none"`` / a ``${...}`` shorthand,
          inherit the right default for that agent's provider.
        - If the agent's provider is ``"none"`` (disabled or
          deterministic), return ``None`` so the router builds nothing.
        """
        provider = (
            getattr(self, f"{agent_name.upper()}_AGENT_PROVIDER", "none")
            or "none"
        ).lower()
        raw = getattr(self, f"{agent_name.upper()}_AGENT_MODEL", None)
        token = (raw or "").strip().lower()
        if token not in self._MODEL_DEFAULT_TOKENS:
            return raw
        return self._resolve_default_model(agent_name, provider)

    def provider_for_agent(self, agent_name: str) -> str:
        """Lowercased provider name for an agent. Always returns
        a string ('none' when off)."""
        raw = getattr(self, f"{agent_name.upper()}_AGENT_PROVIDER", "none")
        return (raw or "none").strip().lower() or "none"


# A simple module-level cache. ``reload_settings`` is provided so tests and
# the CLI can re-read environment variables after mutating ``os.environ``.
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings
