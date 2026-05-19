"""Shared pytest fixtures.

We deliberately make every test load settings from a clean environment so
that a stray ``.env`` in the working directory cannot influence results.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


_ENV_KEYS_TO_RESET = {
    "MODE",
    "LIVE_ADAPTER_CONFIRMED",
    "INSTRUMENT",
    "MARKET_TYPE",
    "TIMEZONE",
    "DATABASE_URL",
    "TRADING_WINDOW_START",
    "TRADING_WINDOW_END",
    "FORCE_FLAT_TIME",
    "MAX_DAILY_LOSS",
    "MAX_DAILY_PROFIT",
    "MAX_TRADES_PER_DAY",
    "MAX_POSITION_SIZE",
    "RISK_PER_TRADE",
    "CONFIDENCE_THRESHOLD",
    "SLIPPAGE_TICKS",
    "COMMISSION_PER_CONTRACT",
    "CRYPTO_SLIPPAGE_BPS",
    "CRYPTO_FEE_BPS",
    "CONSISTENCY_LIMIT_PERCENT",
    "ENABLE_LLM_AGENTS",
    "OPENAI_API_KEY",
    "LLM_MODEL",
    "LLM_TIMEOUT_SECONDS",
    "AGENTS_RUN_AT_EOD",
    "NEWS_CHECK_LOCAL_TIME",
    "LOG_LEVEL",
    "LOG_JSON",
    "ENABLED_STRATEGIES",
    "ENABLED_SYMBOLS",
    "PRIMARY_SYMBOL",
    "MAX_ACTIVE_SYMBOLS",
    "MAX_TRADES_PER_SYMBOL_PER_DAY",
    "MAX_TOTAL_TRADES_PER_DAY",
    "FEEDBACK_MIN_ROWS",
    "FEEDBACK_USE_MISTAKE_TAGS_AS_LABEL",
    "TRADINGVIEW_WEBHOOK_SECRET",
    "WEBHOOK_DEFAULT_STOP_TICKS",
    "WEBHOOK_DEFAULT_TARGET_TICKS",
    "PERPLEXITY_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_MODEL",
    "PERPLEXITY_MODEL",
    "ANTHROPIC_MODEL",
    "GEMINI_MODEL",
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
    # Per-agent model overrides + default-model knobs.
    "OPENAI_DEFAULT_MODEL",
    "OPENAI_REVIEW_MODEL",
    "PERPLEXITY_DEFAULT_MODEL",
    "NEWS_AGENT_MODEL",
    "MACRO_NEWS_AGENT_MODEL",
    "STRATEGY_RESEARCH_AGENT_MODEL",
    "TRADE_ANALYSIS_AGENT_MODEL",
    "MODEL_REVIEW_AGENT_MODEL",
    "REPORT_AGENT_MODEL",
    "RISK_EXPLAINER_AGENT_MODEL",
    "TRADE_JOURNAL_AGENT_MODEL",
    "BACKTEST_CRITIC_AGENT_MODEL",
    "MODEL_DRIFT_AGENT_MODEL",
    "DATA_QUALITY_AGENT_MODEL",
}


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    # Wipe any env vars that could pollute Settings construction.
    for key in _ENV_KEYS_TO_RESET:
        monkeypatch.delenv(key, raising=False)

    # Use a temp DB and prevent reading a real .env.
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    # Pydantic-settings still tries to read .env from cwd; redirect to /dev/null.
    monkeypatch.chdir(tmp_path)

    # Reset cached singletons in app modules.
    from config import settings as cfg
    from storage import db as db_mod

    cfg._settings = None
    db_mod.reset_engine_for_tests()

    yield

    cfg._settings = None
    db_mod.reset_engine_for_tests()
