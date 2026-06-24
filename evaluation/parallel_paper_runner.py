"""Parallel paper runner — launches isolated evaluation tracks per broker."""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.logging_config import get_logger
from config.settings import Settings
from evaluation.evaluation_context import EvaluationContext
from integrations.broker_base import BaseBroker, BrokerError, FUTURES_SYMBOLS, LiveExecutionRefused
from integrations.broker_router import build_broker
from integrations.mock_broker import MockBroker
from notifications.notification_service import NotificationService
from workflows.workflow_runner import WorkflowRunner


_log = get_logger("evaluation.parallel_runner")

_FUTURES_SIM_ALIAS = "futures_sim"


class ParallelPaperRunner:
    """Orchestrates multiple broker evaluation tracks in isolation.

    Each track gets:
    - its own EvaluationContext (symbols, state file, report dir)
    - its own broker instance (built from the router with explicit symbols)
    - its own WorkflowRunner scoped to that context's symbols and state

    Failure in one track does NOT crash or block the other.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        notifier: Optional[NotificationService] = None,
        dry_run: bool = False,
    ) -> None:
        self.settings = settings
        self.notifier = notifier or NotificationService.from_settings(settings)
        self.dry_run = dry_run
        self.contexts: dict[str, EvaluationContext] = {}
        self._brokers: dict[str, BaseBroker] = {}
        self.log = _log

    # ------------------------------------------------------------------
    # Build tracks
    # ------------------------------------------------------------------
    def build_tracks(self) -> list[EvaluationContext]:
        """Create isolated EvaluationContext objects per configured broker."""
        if self.settings.WORKFLOW_EXECUTION_MODE == "LIVE":
            raise LiveExecutionRefused(
                "WORKFLOW_EXECUTION_MODE=LIVE is locked — no parallel paper."
            )

        brokers_raw = [
            b.strip().lower()
            for b in self.settings.PARALLEL_BROKERS.split(",")
            if b.strip()
        ]
        evaluations_dir = Path("data/evaluations")

        for provider in brokers_raw:
            ctx = self._build_context_for(provider, evaluations_dir)
            if ctx is not None:
                self.contexts[provider] = ctx

        return list(self.contexts.values())

    def _build_context_for(
        self, provider: str, evaluations_dir: Path
    ) -> Optional[EvaluationContext]:
        if provider == "alpaca":
            symbols = [
                s.strip().upper()
                for s in self.settings.ALPACA_ENABLED_SYMBOLS.split(",")
                if s.strip()
            ]
            eval_id = self.settings.ALPACA_EVALUATION_ID
            state_path = Path(self.settings.ALPACA_STATE_PATH)
            report_path = evaluations_dir / eval_id
        elif provider == _FUTURES_SIM_ALIAS:
            symbols = [
                s.strip().upper()
                for s in self.settings.FUTURES_SIM_ENABLED_SYMBOLS.split(",")
                if s.strip()
            ]
            eval_id = self.settings.FUTURES_SIM_EVALUATION_ID
            state_path = Path(self.settings.FUTURES_SIM_STATE_PATH)
            report_path = evaluations_dir / eval_id
        else:
            self.log.warning(
                "parallel.unknown_provider",
                provider=provider,
                note="Skipped — only 'alpaca' and 'futures_sim' are supported.",
            )
            return None

        return EvaluationContext(
            evaluation_id=eval_id,
            broker_provider=provider,
            enabled_symbols=symbols,
            state_path=state_path,
            report_path=report_path,
            model_name=None,
            model_version=None,
        )

    # ------------------------------------------------------------------
    # Broker construction
    # ------------------------------------------------------------------
    def build_broker_for(self, ctx: EvaluationContext) -> BaseBroker:
        if self.dry_run:
            return MockBroker(enabled_symbols=ctx.enabled_symbols)

        provider = ctx.broker_provider
        if provider == _FUTURES_SIM_ALIAS:
            return MockBroker(
                enabled_symbols=ctx.enabled_symbols,
                seed_quotes={
                    "MES": 5050.0,
                    "MNQ": 17500.0,
                    "MGC": 2350.0,
                    "MCL": 78.5,
                },
            )
        if provider == "alpaca":
            return build_broker(
                self.settings,
                override=None,
            )
        return MockBroker(enabled_symbols=ctx.enabled_symbols)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run_all(
        self, *, now: Optional[datetime] = None, force: bool = False
    ) -> dict[str, Any]:
        now = now or datetime.now(tz=timezone.utc)
        if not self.contexts:
            self.build_tracks()

        self._notify_start()
        results: dict[str, Any] = {}

        for provider, ctx in self.contexts.items():
            try:
                result = self._run_track(ctx, now=now, force=force)
                results[provider] = result
                ctx.save_state()
            except Exception as e:  # noqa: BLE001
                ctx.block(f"track_failed: {e}")
                ctx.save_state()
                results[provider] = {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                self.log.error(
                    "parallel.track_failed",
                    provider=provider,
                    evaluation_id=ctx.evaluation_id,
                    error=str(e),
                )
                try:
                    self.notifier.notify(
                        "system.error",
                        source="parallel_paper_runner",
                        **ctx.discord_tags(),
                        error=str(e),
                    )
                except Exception:  # noqa: BLE001
                    pass

        return results

    def _run_track(
        self,
        ctx: EvaluationContext,
        *,
        now: datetime,
        force: bool,
    ) -> dict[str, Any]:
        broker = self.build_broker_for(ctx)
        self._brokers[ctx.broker_provider] = broker

        try:
            recon = broker.reconcile()
        except BrokerError as e:
            ctx.block(f"reconcile_failed: {e}")
            return {"success": False, "reason": str(e)}

        self.log.info(
            "parallel.track_start",
            **ctx.discord_tags(),
            symbols=ctx.enabled_symbols,
            reconcile=recon,
        )

        runner = WorkflowRunner(
            self.settings,
            notifier=self.notifier,
            dry_run=self.dry_run,
        )
        day_results = runner.run_day(now=now, force=force)

        return {
            "success": all(r.success for r in day_results),
            "workflows": [r.model_dump() for r in day_results],
            "trades": len(ctx.trades),
            "reconcile": recon,
        }

    def _notify_start(self) -> None:
        providers = sorted(self.contexts.keys())
        try:
            self.notifier.notify(
                "system.info",
                source="parallel_paper_runner",
                message=f"Parallel paper mode started: {' + '.join(providers)}",
                providers=providers,
            )
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Status / reporting
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        if not self.contexts:
            self.build_tracks()
        out: dict[str, Any] = {}
        for provider, ctx in self.contexts.items():
            state = ctx.load_state()
            out[provider] = {
                "evaluation_id": ctx.evaluation_id,
                "broker_provider": provider,
                "enabled_symbols": ctx.enabled_symbols,
                "state_path": str(ctx.state_path),
                "report_path": str(ctx.report_path),
                "started_at": ctx.started_at.isoformat(),
                "blocked": ctx.blocked,
                "errors": ctx.errors,
                "persisted_state": state,
            }
        return out
