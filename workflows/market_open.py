"""Market-open workflow — revalidate research and optional paper entries."""

from __future__ import annotations

import re
from typing import Any

from config.instruments import get_instrument
from execution.base import Order
from workflows.base import BaseWorkflow, WorkflowContext
from workflows.broker_interface import build_paper_executor, submit_paper_order
from workflows.memory import extract_research_section, has_research_for_date
from workflows.order_execution import (
    execute_entry_with_stops,
    fetch_quote,
    prepare_broker,
    resolve_order_broker,
)
from workflows.options_execution import execute_options_signal, options_enabled_for
from workflows.premarket import PremarketWorkflow
from workflows.schemas import SimulatedTradeDecision
from workflows.signal_engine import SignalEngine


class MarketOpenWorkflow(BaseWorkflow):
    name = "market-open"

    def _execute(self, ctx: WorkflowContext) -> dict[str, Any]:
        if not has_research_for_date(ctx.memory.research_log, ctx.session_date):
            PremarketWorkflow().run(ctx)
            if not has_research_for_date(ctx.memory.research_log, ctx.session_date):
                raise RuntimeError(
                    "Market-open refused: no dated research after inline pre-market."
                )

        research = extract_research_section(
            ctx.memory.research_log, ctx.session_date
        )
        csv_quotes = _fresh_quotes_from_csv(ctx)
        reconcile_info: dict[str, Any] = {}
        if resolve_order_broker(ctx) is not None:
            reconcile_info = prepare_broker(ctx)

        quotes = {
            sym: fetch_quote(ctx, sym, csv_price=csv_quotes.get(sym))
            for sym in (
                list(ctx.settings.ENABLED_SYMBOLS) or [ctx.settings.INSTRUMENT]
            )
        }
        broker_state = ctx.broker.pull_state(now=ctx.now, quotes={
            k: v for k, v in quotes.items() if v is not None
        })
        ideas = _parse_planned_symbols(research)
        if not ideas:
            ideas = [
                s.upper()
                for s in (
                    list(ctx.settings.ENABLED_SYMBOLS)
                    or [ctx.settings.INSTRUMENT]
                )
            ]
        decisions: list[SimulatedTradeDecision] = []
        actions_taken = 0
        discord_sent = False

        risk_gate = ctx.gates.high_risk_news_blocks_trading(None, False)
        if ctx.orchestrator is not None and ctx.orchestrator.high_risk_news_active:
            risk_gate = ctx.gates.high_risk_news_blocks_trading("high", True)

        autonomous = ctx.gates.autonomous_execution_allowed(ctx.execution_mode)
        use_integration = autonomous and not ctx.dry_run and resolve_order_broker(ctx) is not None
        executor = None
        if autonomous and not ctx.dry_run and not use_integration:
            executor = build_paper_executor(ctx.settings)

        signal_engine = SignalEngine(
            ctx.settings,
            model_name=ctx.settings.WORKFLOW_MODEL_NAME,
            model_version=ctx.settings.WORKFLOW_MODEL_VERSION,
        )

        for sym in ideas[: ctx.settings.MAX_ACTIVE_SYMBOLS]:
            price = quotes.get(sym.upper())
            if price is None:
                decisions.append(
                    SimulatedTradeDecision(
                        symbol=sym,
                        side="flat",
                        decision="skip",
                        reason="No fresh quote available",
                    )
                )
                continue

            if ctx.entries_blocked:
                decisions.append(
                    SimulatedTradeDecision(
                        symbol=sym,
                        side="long",
                        decision="skip",
                        reason="Entries blocked after broker failure",
                    )
                )
                continue

            if not risk_gate.passed:
                decisions.append(
                    SimulatedTradeDecision(
                        symbol=sym,
                        side="flat",
                        decision="hold",
                        reason=risk_gate.reason,
                    )
                )
                continue

            # Real signal: run the VWAP/EMA strategy (+ optional model)
            # instead of assuming a direction. No setup / no data / model
            # rejection → skip the symbol.
            signal = signal_engine.generate_signal(sym)
            if signal is None:
                decisions.append(
                    SimulatedTradeDecision(
                        symbol=sym,
                        side="flat",
                        decision="skip",
                        reason="No strategy setup / no data for symbol",
                    )
                )
                continue
            if not signal.approved:
                decisions.append(
                    SimulatedTradeDecision(
                        symbol=sym,
                        side=signal.direction,
                        decision="skip",
                        entry_price=signal.entry_price,
                        stop_price=signal.stop_price,
                        target_price=signal.target_price,
                        reason=f"Signal not approved ({signal.reason})",
                    )
                )
                continue

            side = signal.direction
            entry_signal_price = signal.entry_price
            stop_price = signal.stop_price
            target_price = signal.target_price
            # Use the live quote for execution pricing when available,
            # else the strategy's entry price.
            price = price or entry_signal_price
            rr = abs(target_price - entry_signal_price) / max(
                abs(entry_signal_price - stop_price), 1e-9
            )
            conf_txt = (
                f" conf={signal.confidence:.2f}" if signal.confidence is not None else ""
            )
            thesis = (
                f"{side} VWAP/EMA setup for {sym} on {ctx.session_date}"
                f"{conf_txt} (reason={signal.reason})."
            )

            if ctx.dry_run or not autonomous:
                decisions.append(
                    SimulatedTradeDecision(
                        symbol=sym,
                        side=side,
                        decision="hold",
                        entry_price=price,
                        stop_price=stop_price,
                        target_price=target_price,
                        thesis=thesis,
                        risk_reward=f"1:{rr:.1f}",
                        reason="DRY_RUN — simulated hold unless gates pass later",
                    )
                )
                continue

            if broker_state.account.open_positions >= ctx.settings.MAX_OPEN_POSITIONS:
                decisions.append(
                    SimulatedTradeDecision(
                        symbol=sym,
                        side=side,
                        decision="skip",
                        reason="Max open positions reached",
                    )
                )
                continue

            qty = float(ctx.settings.MAX_POSITION_SIZE)

            # Options routing: if enabled for this underlying, express the
            # directional signal as an options trade instead of shares.
            if options_enabled_for(ctx, sym):
                opt_result = execute_options_signal(
                    ctx,
                    underlying=sym,
                    direction=side,
                    thesis=thesis,
                    now=ctx.now,
                )
                status = opt_result.get("status")
                if status == "opened":
                    actions_taken += 1
                    decisions.append(
                        SimulatedTradeDecision(
                            symbol=sym,
                            side=side,
                            decision="enter",
                            entry_price=price,
                            stop_price=stop_price,
                            target_price=target_price,
                            thesis=thesis,
                            reason=f"Options entry ({ctx.settings.OPTIONS_STRATEGY})",
                        )
                    )
                    discord_sent = self._notify_safe(
                        ctx,
                        "options.opened",
                        instrument=sym,
                        direction=side,
                        strategy=ctx.settings.OPTIONS_STRATEGY,
                        broker_provider=ctx.settings.BROKER_PROVIDER,
                        source="workflow.market_open",
                        thesis=thesis,
                        detail=opt_result,
                    ) or discord_sent
                else:
                    decisions.append(
                        SimulatedTradeDecision(
                            symbol=sym,
                            side=side,
                            decision="skip",
                            reason=f"Options {status}: {opt_result.get('reason', '')}",
                        )
                    )
                continue

            if use_integration:
                ok, entry_res, stop_res = execute_entry_with_stops(
                    ctx,
                    symbol=sym,
                    side=side,
                    quantity=qty,
                    entry_price=price,
                    stop_price=stop_price,
                    target_price=target_price,
                    thesis=thesis,
                )
                if ok:
                    actions_taken += 1
                    decisions.append(
                        SimulatedTradeDecision(
                            symbol=sym,
                            side=side,
                            decision="enter",
                            entry_price=price,
                            stop_price=stop_price,
                            target_price=target_price,
                            thesis=thesis,
                            risk_reward=f"1:{rr:.1f}",
                            reason=(
                                f"Broker entry ({ctx.settings.BROKER_PROVIDER}) "
                                f"order={entry_res.order_id if entry_res else 'n/a'}"
                            ),
                        )
                    )
                    discord_sent = self._notify_safe(
                        ctx,
                        "trade.opened",
                        instrument=sym,
                        direction=side,
                        entry_price=price,
                        stop_price=stop_price,
                        target_price=target_price,
                        source="workflow.market_open",
                        broker=ctx.settings.BROKER_PROVIDER,
                        thesis=thesis,
                        entry_order=entry_res.to_payload() if entry_res else {},
                        stop_order=stop_res.to_payload() if stop_res else {},
                    ) or discord_sent
                else:
                    decisions.append(
                        SimulatedTradeDecision(
                            symbol=sym,
                            side=side,
                            decision="skip",
                            reason="Broker entry/stop failed",
                        )
                    )
                continue

            assert executor is not None
            order = Order(
                instrument=sym,
                direction=side,
                quantity=qty,
                entry_price=price,
                stop_price=stop_price,
                target_price=target_price,
                setup_id=f"workflow-{ctx.session_date}",
            )
            ok = submit_paper_order(executor, order=order)
            if ok:
                actions_taken += 1
                decisions.append(
                    SimulatedTradeDecision(
                        symbol=sym,
                        side=side,
                        decision="enter",
                        entry_price=price,
                        stop_price=stop_price,
                        target_price=target_price,
                        thesis=thesis,
                        risk_reward=f"1:{rr:.1f}",
                        reason="Autonomous local paper entry",
                    )
                )
                discord_sent = self._notify_safe(
                    ctx,
                    "trade.opened",
                    instrument=sym,
                    direction=side,
                    entry_price=price,
                    stop_price=stop_price,
                    target_price=target_price,
                    source="workflow.market_open",
                    thesis=thesis,
                ) or discord_sent
            else:
                decisions.append(
                    SimulatedTradeDecision(
                        symbol=sym,
                        side=side,
                        decision="skip",
                        reason="Paper submit failed",
                    )
                )

        return {
            "research_present": True,
            "decisions": [d.model_dump() for d in decisions],
            "actions_taken": actions_taken,
            "dry_run": ctx.dry_run,
            "autonomous": autonomous,
            "broker_provider": ctx.settings.BROKER_PROVIDER,
            "reconcile": reconcile_info,
            "entries_blocked": ctx.entries_blocked,
            "discord_sent": discord_sent,
        }


def _fresh_quotes_from_csv(ctx: WorkflowContext) -> dict[str, float]:
    """Last close from per-symbol historical CSV (fallback pricing)."""
    quotes: dict[str, float] = {}
    hist = ctx.settings.HISTORICAL_DATA_DIR
    for sym in ctx.settings.ENABLED_SYMBOLS:
        path = hist / sym / "1m.csv"
        if path.exists():
            try:
                import pandas as pd

                df = pd.read_csv(path)
                if "close" in df.columns and len(df):
                    quotes[sym.upper()] = float(df["close"].iloc[-1])
                    continue
            except Exception:  # noqa: BLE001
                pass
        spec = get_instrument(sym)
        quotes[sym.upper()] = 5000.0 if spec.market_type == "futures" else 100.0
    return quotes


def _parse_planned_symbols(research_md: str) -> list[str]:
    symbols: list[str] = []
    for line in research_md.splitlines():
        m = re.match(r"#### Idea \d+:\s+(\w+)", line)
        if m:
            symbols.append(m.group(1).upper())
    return symbols
