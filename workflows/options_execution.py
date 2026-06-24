"""Bridge: route a workflow directional signal into the options layer.

Keeps the options wiring out of ``market_open`` so the workflow stays
broker/asset agnostic. When ``OPTIONS_ENABLED`` is true and the symbol
is an allowed options underlying, the signal becomes an options trade
(call/put/spread/condor) instead of an equity share order.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.logging_config import get_logger
from workflows.base import WorkflowContext

_log = get_logger("workflows.options_execution")


def options_underlyings(ctx: WorkflowContext) -> set[str]:
    return {
        s.strip().upper()
        for s in ctx.settings.OPTIONS_ENABLED_UNDERLYINGS.split(",")
        if s.strip()
    }


def options_enabled_for(ctx: WorkflowContext, symbol: str) -> bool:
    if not ctx.settings.OPTIONS_ENABLED:
        return False
    return symbol.upper() in options_underlyings(ctx)


def build_options_trader(ctx: WorkflowContext, *, now: Optional[datetime] = None):
    """Construct an OptionsTrader scoped to this workflow's execution mode.

    DRY_RUN → synthetic chain + mock executor (no network).
    PAPER + alpaca → Alpaca chain provider + Alpaca options executor.
    Anything else → mock (safe default).
    """
    # Imported lazily so the options package is only pulled in when used.
    from options.chain import AlpacaChainProvider, SyntheticChainProvider
    from options.execution import AlpacaOptionsExecutor, MockOptionsExecutor
    from options.position_manager import ManagerConfig, OptionsPositionManager
    from options.trader import OptionsTrader

    settings = ctx.settings

    if ctx.dry_run or settings.BROKER_PROVIDER != "alpaca":
        return OptionsTrader.for_dry_run(settings, now=now)

    api_key = (
        settings.ALPACA_API_KEY.get_secret_value() if settings.ALPACA_API_KEY else None
    )
    secret = (
        settings.ALPACA_SECRET_KEY.get_secret_value()
        if settings.ALPACA_SECRET_KEY
        else None
    )
    try:
        chain_provider = AlpacaChainProvider(api_key=api_key, secret_key=secret)
        executor = AlpacaOptionsExecutor(
            api_key=api_key,
            secret_key=secret,
            base_url=settings.ALPACA_BASE_URL,
            paper=settings.ALPACA_PAPER,
            timeout_seconds=float(settings.BROKER_REQUEST_TIMEOUT_SECONDS),
        )
    except Exception as e:  # noqa: BLE001
        _log.error("options.alpaca_build_failed", error=str(e))
        return None

    pm = OptionsPositionManager(
        executor=executor,
        chain_provider=chain_provider,
        state_path=Path(settings.OPTIONS_STATE_PATH),
        config=ManagerConfig(
            auto_close_dte=settings.OPTIONS_AUTO_CLOSE_DTE,
            auto_roll=settings.OPTIONS_AUTO_ROLL,
            roll_dte_trigger=settings.OPTIONS_ROLL_DTE_TRIGGER,
            profit_target_pct=settings.OPTIONS_PROFIT_TARGET_PCT,
            stop_loss_pct=settings.OPTIONS_STOP_LOSS_PCT,
        ),
    )
    return OptionsTrader(
        settings,
        chain_provider=chain_provider,
        executor=executor,
        notifier=ctx.notifier,
        position_manager=pm,
    )


def execute_options_signal(
    ctx: WorkflowContext,
    *,
    underlying: str,
    direction: str,
    thesis: str,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Route a directional signal to the options trader. Never raises."""
    trader = build_options_trader(ctx, now=now)
    if trader is None:
        return {"status": "error", "reason": "options_trader_unavailable"}
    try:
        return trader.handle_signal(
            underlying=underlying, direction=direction, now=now, thesis=thesis
        )
    except Exception as e:  # noqa: BLE001
        _log.error("options.execute_failed", underlying=underlying, error=str(e))
        return {"status": "error", "reason": str(e)}
