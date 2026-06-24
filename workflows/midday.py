"""Midday workflow — manage open positions (stop/trail/thesis break)."""

from __future__ import annotations

from typing import Any

from workflows.base import BaseWorkflow, WorkflowContext
from workflows.broker_interface import build_paper_executor, close_paper_position
from workflows.market_open import _fresh_quotes_from_csv
from workflows.order_execution import execute_close, fetch_quote, resolve_order_broker


class MiddayWorkflow(BaseWorkflow):
    name = "midday"

    def _execute(self, ctx: WorkflowContext) -> dict[str, Any]:
        csv_quotes = _fresh_quotes_from_csv(ctx)
        quotes = {
            sym: fetch_quote(ctx, sym.upper(), csv_price=csv_quotes.get(sym.upper()))
            for sym in (list(ctx.settings.ENABLED_SYMBOLS) or [ctx.settings.INSTRUMENT])
        }
        broker_state = ctx.broker.pull_state(
            now=ctx.now,
            quotes={k: v for k, v in quotes.items() if v is not None},
        )
        actions: list[dict[str, Any]] = []
        discord_sent = False
        can_execute = (
            ctx.gates.autonomous_execution_allowed(ctx.execution_mode)
            and not ctx.dry_run
        )
        use_integration = can_execute and resolve_order_broker(ctx) is not None
        executor = None
        if can_execute and not use_integration:
            executor = build_paper_executor(ctx.settings)

        for pos in broker_state.positions:
            pct = pos.unrealized_pnl_pct
            if pct is None:
                continue
            action: dict[str, Any] = {"symbol": pos.instrument, "pnl_pct": pct}

            if pct <= -7.0:
                action["kind"] = "close_loss"
                action["reason"] = "Unrealized PnL <= -7%"
                if can_execute and use_integration:
                    result = execute_close(ctx, symbol=pos.instrument)
                    action["executed"] = bool(result and result.success)
                    if result:
                        action["order"] = result.to_payload()
                elif can_execute and executor is not None and pos.current_price:
                    action["executed"] = close_paper_position(
                        executor,
                        now=ctx.now,
                        exit_price=pos.current_price,
                        reason="workflow_midday_loss",
                    )
                elif ctx.dry_run:
                    action["executed"] = False
                    action["simulated"] = True
                if action.get("executed") or ctx.dry_run:
                    discord_sent = self._notify_safe(
                        ctx,
                        "trade.closed",
                        instrument=pos.instrument,
                        reason=action["reason"],
                        pnl_pct=pct,
                        source="workflow.midday",
                    ) or discord_sent
                actions.append(action)
                continue

            if pct >= 20.0:
                action["kind"] = "tighten_trail"
                action["trail"] = "aggressive (+20%)"
            elif pct >= 15.0:
                action["kind"] = "tighten_trail"
                action["trail"] = "moderate (+15%)"
            else:
                action["kind"] = "monitor"

            if _thesis_broken(pos, ctx):
                action["kind"] = "thesis_break"
                action["reason"] = "Price through stop or high-risk news"
                if can_execute and use_integration:
                    result = execute_close(ctx, symbol=pos.instrument)
                    action["executed"] = bool(result and result.success)
                    if result:
                        action["order"] = result.to_payload()
                elif can_execute and executor is not None and pos.current_price:
                    action["executed"] = close_paper_position(
                        executor,
                        now=ctx.now,
                        exit_price=pos.current_price,
                        reason="workflow_midday_thesis",
                    )
                elif ctx.dry_run:
                    action["simulated"] = True
                if action.get("executed") or (
                    ctx.dry_run and action["kind"] == "thesis_break"
                ):
                    discord_sent = self._notify_safe(
                        ctx,
                        "trade.closed",
                        instrument=pos.instrument,
                        reason=action.get("reason", "thesis_break"),
                        source="workflow.midday",
                    ) or discord_sent

            actions.append(action)

        # Manage open options positions (profit target / stop / expiry / roll).
        options_actions: list[dict[str, Any]] = []
        if ctx.settings.OPTIONS_ENABLED and can_execute:
            try:
                from workflows.options_execution import build_options_trader

                trader = build_options_trader(ctx, now=ctx.now)
                if trader is not None:
                    options_actions = trader.manage(now=ctx.now)
            except Exception as e:  # noqa: BLE001
                self.log.warning("workflow.options_manage_failed", error=str(e))

        return {
            "positions_checked": len(broker_state.positions),
            "actions": actions,
            "options_actions": options_actions,
            "discord_sent": discord_sent,
            "dry_run": ctx.dry_run,
            "broker_provider": ctx.settings.BROKER_PROVIDER,
        }


def _thesis_broken(pos, ctx: WorkflowContext) -> bool:  # noqa: ANN001
    if ctx.orchestrator is not None and ctx.orchestrator.high_risk_news_active:
        return True
    if pos.current_price is None:
        return False
    if pos.direction == "long" and pos.current_price <= pos.stop_price:
        return True
    if pos.direction == "short" and pos.current_price >= pos.stop_price:
        return True
    return False
