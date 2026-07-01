"""Broker router: enforces DRY_RUN/PAPER/LIVE rails before any external call.

Workflows must call only this module to obtain a broker. Direct imports
of :class:`AlpacaPaperClient` or :class:`TradovateDemoClient` from the
``workflows/`` package are forbidden (enforced by tests).
"""

from __future__ import annotations

from typing import Optional

from app.logging_config import get_logger
from config.settings import Settings
from integrations.alpaca_paper_client import (
    AlpacaConfigurationError,
    AlpacaPaperClient,
)
from integrations.broker_base import (
    BaseBroker,
    BrokerError,
    LiveExecutionRefused,
)
from integrations.mock_broker import MockBroker
from integrations.tradovate_demo_client import (
    TradovateConfigurationError,
    TradovateDemoClient,
)


class InvalidBrokerProviderError(BrokerError):
    """Raised when ``BROKER_PROVIDER`` is set to an unknown value."""


_SUPPORTED_PROVIDERS = frozenset({"mock", "alpaca", "tradovate"})


class BrokerRouter:
    """Picks the right broker adapter for the active execution mode.

    Rules:

    - ``DRY_RUN``  → always returns :class:`MockBroker` (no network).
    - ``PAPER``    → returns the configured provider:
        * ``mock``      → :class:`MockBroker`
        * ``alpaca``    → :class:`AlpacaPaperClient` (paper-only rails)
        * ``tradovate`` → :class:`TradovateDemoClient` (demo-only rails)
      Each adapter raises a configuration error when its safety rails
      fail; the workflow runner catches the error and blocks new
      entries.
    - ``LIVE``     → :class:`LiveExecutionRefused` (always).
    - Unknown provider → :class:`InvalidBrokerProviderError`.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.log = get_logger("integrations.broker_router")

    def execution_mode(self) -> str:
        return str(self.settings.WORKFLOW_EXECUTION_MODE).upper()

    def _tradable_symbols(self) -> list[str]:
        """Symbols the broker is allowed to trade.

        Just ``ENABLED_SYMBOLS`` normally, but when the loop scans a dynamic
        universe the broker must also accept every vetted allowlist name —
        otherwise a valid NVDA/AMD/QCOM signal is rejected as "not enabled"
        even though the loop deliberately scanned it. The allowlist is the
        same safety boundary the scan universe uses.
        """
        symbols = list(self.settings.ENABLED_SYMBOLS)
        if getattr(self.settings, "WORKFLOW_DYNAMIC_UNIVERSE", False):
            from config.equity_allowlist import LIQUID_EQUITY_ALLOWLIST

            seen = {s.upper() for s in symbols}
            for s in sorted(LIQUID_EQUITY_ALLOWLIST):
                if s not in seen:
                    symbols.append(s)
                    seen.add(s)
        return symbols

    def is_live_locked(self) -> bool:
        return self.execution_mode() == "LIVE"

    def for_execution_mode(self) -> BaseBroker:
        mode = self.execution_mode()
        if mode == "LIVE":
            raise LiveExecutionRefused(
                "WORKFLOW_EXECUTION_MODE=LIVE is locked — no broker is built."
            )

        if mode == "DRY_RUN":
            self.log.info(
                "broker.router_select",
                mode=mode,
                provider="mock",
                reason="dry_run_forces_mock",
            )
            return MockBroker(enabled_symbols=self._tradable_symbols())

        provider = self.settings.BROKER_PROVIDER
        if provider not in _SUPPORTED_PROVIDERS:
            raise InvalidBrokerProviderError(
                f"BROKER_PROVIDER='{provider}' is not supported. "
                f"Allowed: {sorted(_SUPPORTED_PROVIDERS)}."
            )

        if provider == "mock":
            self.log.info("broker.router_select", mode=mode, provider="mock")
            return MockBroker(enabled_symbols=self._tradable_symbols())

        if provider == "alpaca":
            return self._build_alpaca(mode)

        return self._build_tradovate(mode)

    # ------------------------------------------------------------------
    # Per-provider builders
    # ------------------------------------------------------------------
    def _build_alpaca(
        self, mode: str, *, enabled_symbols_override: Optional[list[str]] = None
    ) -> BaseBroker:
        api_key = (
            self.settings.ALPACA_API_KEY.get_secret_value()
            if self.settings.ALPACA_API_KEY
            else None
        )
        secret = (
            self.settings.ALPACA_SECRET_KEY.get_secret_value()
            if self.settings.ALPACA_SECRET_KEY
            else None
        )
        symbols = enabled_symbols_override or self._tradable_symbols()
        try:
            client = AlpacaPaperClient(
                api_key=api_key,
                secret_key=secret,
                base_url=self.settings.ALPACA_BASE_URL,
                paper=self.settings.ALPACA_PAPER,
                enabled_symbols=symbols,
                timeout_seconds=float(self.settings.BROKER_REQUEST_TIMEOUT_SECONDS),
            )
        except AlpacaConfigurationError as e:
            self.log.error(
                "broker.router_refuse_alpaca",
                mode=mode,
                reason=str(e),
            )
            raise
        self.log.info(
            "broker.router_select",
            mode=mode,
            provider="alpaca",
            base_url=self.settings.ALPACA_BASE_URL,
        )
        return client

    def _build_tradovate(
        self, mode: str, *, enabled_symbols_override: Optional[list[str]] = None
    ) -> BaseBroker:
        symbols = enabled_symbols_override or self._tradable_symbols()
        try:
            client = TradovateDemoClient(
                base_url=self.settings.TRADOVATE_BASE_URL,
                username=self.settings.TRADOVATE_USERNAME,
                password=(
                    self.settings.TRADOVATE_PASSWORD.get_secret_value()
                    if self.settings.TRADOVATE_PASSWORD
                    else None
                ),
                app_id=self.settings.TRADOVATE_APP_ID,
                app_version=self.settings.TRADOVATE_APP_VERSION,
                client_id=self.settings.TRADOVATE_CLIENT_ID,
                client_secret=(
                    self.settings.TRADOVATE_CLIENT_SECRET.get_secret_value()
                    if self.settings.TRADOVATE_CLIENT_SECRET
                    else None
                ),
                demo=self.settings.TRADOVATE_DEMO,
                enabled_symbols=symbols,
                timeout_seconds=float(self.settings.BROKER_REQUEST_TIMEOUT_SECONDS),
            )
        except TradovateConfigurationError as e:
            self.log.error(
                "broker.router_refuse_tradovate",
                mode=mode,
                reason=str(e),
            )
            raise
        self.log.info(
            "broker.router_select",
            mode=mode,
            provider="tradovate",
            base_url=self.settings.TRADOVATE_BASE_URL,
        )
        return client


def build_broker(
    settings: Settings,
    *,
    override: Optional[BaseBroker] = None,
) -> BaseBroker:
    if override is not None:
        return override
    return BrokerRouter(settings).for_execution_mode()


def build_broker_for_provider(
    settings: Settings,
    *,
    provider: str,
    enabled_symbols: list[str],
) -> BaseBroker:
    """Build an isolated broker instance for a specific provider + symbol set.

    Used by parallel paper mode so each evaluation track gets its own
    broker without relying on the global ``BROKER_PROVIDER`` setting.
    """
    router = BrokerRouter(settings)
    mode = router.execution_mode()
    if mode == "LIVE":
        raise LiveExecutionRefused(
            "WORKFLOW_EXECUTION_MODE=LIVE is locked — no broker is built."
        )
    if mode == "DRY_RUN":
        return MockBroker(enabled_symbols=enabled_symbols)

    if provider == "mock" or provider == "futures_sim":
        return MockBroker(enabled_symbols=enabled_symbols)
    if provider == "alpaca":
        return router._build_alpaca(mode, enabled_symbols_override=enabled_symbols)
    if provider == "tradovate":
        return router._build_tradovate(mode, enabled_symbols_override=enabled_symbols)
    raise InvalidBrokerProviderError(
        f"Unknown provider '{provider}' for parallel paper."
    )


__all__ = [
    "BrokerError",
    "BrokerRouter",
    "InvalidBrokerProviderError",
    "LiveExecutionRefused",
    "build_broker",
    "build_broker_for_provider",
]
