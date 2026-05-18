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
from typing import Literal, Optional
from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    # ---- Model ---------------------------------------------------------
    CONFIDENCE_THRESHOLD: float = 0.60

    # ---- Costs ---------------------------------------------------------
    SLIPPAGE_TICKS: float = Field(default=1.0, ge=0.0)
    COMMISSION_PER_CONTRACT: float = Field(default=1.50, ge=0.0)

    # ---- Compliance ----------------------------------------------------
    CONSISTENCY_LIMIT_PERCENT: float = Field(default=30.0, gt=0.0, le=100.0)

    # ---- LLM agents ----------------------------------------------------
    ENABLE_LLM_AGENTS: bool = False
    OPENAI_API_KEY: Optional[SecretStr] = None

    # ---- Notifications -------------------------------------------------
    DISCORD_WEBHOOK_URL: Optional[SecretStr] = None

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
    @field_validator("TRADING_WINDOW_START", "TRADING_WINDOW_END", "FORCE_FLAT_TIME")
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
