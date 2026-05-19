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
MarketType = Literal["futures", "crypto"]


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

    # ---- Per-provider model overrides --------------------------------
    # Each provider has a sensible default; operators can override
    # without recompiling. ``LLM_MODEL`` is kept for back-compat and
    # acts as the default for the OpenAI provider when
    # ``OPENAI_MODEL`` is not set.
    OPENAI_MODEL: Optional[str] = None
    PERPLEXITY_MODEL: str = "sonar"
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-latest"
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # ---- Per-agent provider routing ----------------------------------
    # The router reads these to decide which provider executes each
    # advisory agent. Web-grounded research (news, macro, strategy)
    # defaults to Perplexity; reasoning/summarization defaults to
    # OpenAI. Set to "none" / "disabled" to switch an agent off
    # without removing it from the orchestrator.
    NEWS_AGENT_PROVIDER: str = "perplexity"
    MACRO_NEWS_AGENT_PROVIDER: str = "perplexity"
    STRATEGY_RESEARCH_AGENT_PROVIDER: str = "perplexity"
    TRADE_ANALYSIS_AGENT_PROVIDER: str = "openai"
    MODEL_REVIEW_AGENT_PROVIDER: str = "openai"
    REPORT_AGENT_PROVIDER: str = "openai"
    RISK_EXPLAINER_AGENT_PROVIDER: str = "openai"
    TRADE_JOURNAL_AGENT_PROVIDER: str = "openai"

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

    @field_validator("TIMEZONE")
    @classmethod
    def _validate_tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except Exception as e:
            raise ValueError(f"Invalid timezone '{v}'") from e
        return v

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
