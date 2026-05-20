"""Workflow order execution via integrations broker (Tradovate demo / mock)."""

from __future__ import annotations

from typing import Any, Optional

from app.logging_config import get_logger
from config.instruments import get_instrument
from integrations.broker_base import BaseBroker, BrokerError, OrderResult
from integrations.broker_router import build_broker
from workflows.base import WorkflowContext

_log = get_logger("workflows.order_execution")


def resolve_order_broker(ctx: WorkflowContext) -> Optional[BaseBroker]:
    """Return the integrations broker when workflow PAPER execution is active."""
    if ctx.dry_run:
        return None
    if ctx.order_broker is not None:
        return ctx.order_broker
    if not ctx.gates.autonomous_execution_allowed(ctx.execution_mode):
        return None
    try:
        broker = build_broker(ctx.settings)
        ctx.order_broker = broker
        return broker
    except Exception as e:  # noqa: BLE001
        _log.error("workflow.broker_build_failed", error=str(e))
        _block_entries(ctx, reason=f"broker_build_failed: {e}")
        return None


def prepare_broker(ctx: WorkflowContext) -> dict[str, Any]:
    """Reconcile account/positions/orders before any new entry."""
    broker = resolve_order_broker(ctx)
    if broker is None:
        return {"reconciled": False, "reason": "no_broker"}
    try:
        snapshot = broker.reconcile()
        _log.info("workflow.broker_reconciled", **snapshot)
        return {"reconciled": True, **snapshot}
    except BrokerError as e:
        _block_entries(ctx, reason=f"reconcile_failed: {e}")
        return {"reconciled": False, "reason": str(e)}


def fetch_quote(ctx: WorkflowContext, symbol: str, *, csv_price: Optional[float]) -> Optional[float]:
    """Prefer broker quote; fall back to CSV-derived price."""
    sym = symbol.upper()
    broker = resolve_order_broker(ctx)
    if broker is not None:
        try:
            q = broker.get_latest_quote(sym)
            return float(q.last)
        except BrokerError:
            pass
    return csv_price


def execute_entry_with_stops(
    ctx: WorkflowContext,
    *,
    symbol: str,
    side: str,
    quantity: float,
    entry_price: float,
    stop_price: float,
    thesis: str,
) -> tuple[bool, Optional[OrderResult], Optional[OrderResult]]:
    """Place entry (limit preferred) then protective stop on the integrations broker."""
    if ctx.entries_blocked:
        return False, None, None

    broker = resolve_order_broker(ctx)
    if broker is None:
        return False, None, None

    order_side = "buy" if side == "long" else "sell"
    stop_side = "sell" if side == "long" else "buy"

    validation = broker.validate_order(symbol=symbol, qty=quantity, side=order_side)
    if not validation.valid:
        _log.warning(
            "workflow.order_validation_failed",
            symbol=symbol,
            reason=validation.reason,
        )
        return False, None, None

    try:
        if ctx.settings.DEFAULT_ORDER_TYPE == "market":
            entry = broker.place_market_order(
                symbol=symbol, qty=quantity, side=order_side
            )
        else:
            entry = broker.place_limit_order(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                limit_price=entry_price,
            )
    except BrokerError as e:
        _block_entries(ctx, reason=f"entry_failed: {e}")
        return False, None, None

    if not entry.success:
        _block_entries(ctx, reason=f"entry_rejected: {entry.reason}")
        return False, entry, None

    try:
        stop = broker.place_stop_order(
            symbol=symbol,
            qty=quantity,
            stop_price=stop_price,
            side=stop_side,
        )
    except BrokerError as e:
        _block_entries(ctx, reason=f"stop_failed: {e}")
        ctx.notifier.notify(
            "system.error",
            source="workflow.order_execution",
            symbol=symbol,
            error=str(e),
            detail="Stop placement failed — blocking new entries",
        )
        return False, entry, None

    if not stop.success:
        _block_entries(ctx, reason=f"stop_rejected: {stop.reason}")
        ctx.notifier.notify(
            "system.error",
            source="workflow.order_execution",
            symbol=symbol,
            error=stop.reason or "stop rejected",
            detail="Stop placement failed — blocking new entries",
        )
        return False, entry, stop

    _log.info(
        "workflow.entry_with_stop",
        symbol=symbol,
        entry_id=entry.order_id,
        stop_id=stop.order_id,
        thesis=thesis[:120],
    )
    return True, entry, stop


def execute_close(ctx: WorkflowContext, *, symbol: str) -> Optional[OrderResult]:
    """Close an open position on the integrations broker."""
    broker = resolve_order_broker(ctx)
    if broker is None:
        return None
    try:
        return broker.close_position(symbol=symbol)
    except BrokerError as e:
        _log.error("workflow.close_failed", symbol=symbol, error=str(e))
        return None


def _block_entries(ctx: WorkflowContext, *, reason: str) -> None:
    ctx.entries_blocked = True
    _log.error("workflow.entries_blocked", reason=reason)
    try:
        ctx.notifier.notify(
            "system.error",
            source="workflow.order_execution",
            reason=reason,
            detail="Broker failure — new entries blocked for this run",
        )
    except Exception:  # noqa: BLE001
        pass
