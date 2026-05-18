"""Orchestrate per-trade analysis after every close.

Single public entry point :meth:`PostTradeAnalysisService.on_trade_closed`,
called by ``paper/loop.py`` (and optionally by ``backtesting/engine.py``).

Pipeline:

1. ``TradeAnalyzer.analyze_closed_trade(...)``
2. ``MistakeClassifier.classify(...)``
3. Persist :class:`storage.tables.TradeAnalysis` + per-tag rows.
4. Render the per-trade Markdown via ``reports/post_trade_report.py``.
5. Optionally run :class:`agents.trade_analysis_agent.TradeAnalysisAgent`
   against the LLM and persist the (possibly-invalid) AgentResult to
   ``agent_outputs``. Failures NEVER raise.
6. Notify Discord with kind ``trade.analysis``.

Safety:

- Wrapped at every step; an exception in any one step is logged and the
  pipeline continues with whatever data it has so far.
- Notification payload is sanitized — the analyzer is the only authority
  on tags and PnL fields.
- The service does not modify settings, risk caps, model thresholds, or
  registry pointers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from sqlalchemy import select

from agents.base_agent import AgentContext
from agents.llm_client import LLMClient
from agents.schemas import AgentResult, TradeAnalysisSummary
from agents.trade_analysis_agent import TradeAnalysisAgent
from analysis.mistake_classifier import MistakeClassifier, MistakeTagging
from analysis.trade_analyzer import TradeAnalyzer
from analysis.types import PostTradeAnalysis
from app.logging_config import get_logger
from config.settings import Settings
from reports.post_trade_report import write_post_trade_report
from storage.db import session_scope
from storage.tables import (
    AgentOutput as AgentOutputRow,
    TradeAnalysis as TradeAnalysisRow,
    TradeMistakeTag as TradeMistakeTagRow,
)


class _NotifierLike(Protocol):
    def notify(self, kind: str, /, **payload: Any) -> None: ...


@dataclass
class TradeAnalysisOutcome:
    """Aggregated return from :meth:`PostTradeAnalysisService.on_trade_closed`."""

    closed_trade_id: str
    analysis: Optional[PostTradeAnalysis]
    tagging: Optional[MistakeTagging]
    md_path: Optional[Path]
    agent_summary: Optional[TradeAnalysisSummary]
    persisted: bool
    errors: list[str]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class PostTradeAnalysisService:
    """Per-trade analysis pipeline. Reused by paper, backtest, and the smoke CLI."""

    def __init__(
        self,
        settings: Settings,
        *,
        notifier: _NotifierLike,
        llm: Optional[LLMClient] = None,
        write_reports: bool = True,
    ) -> None:
        self.settings = settings
        self.notifier = notifier
        self.llm = llm
        self.write_reports = bool(write_reports)
        self.analyzer = TradeAnalyzer(settings)
        self.classifier = MistakeClassifier()
        self.log = get_logger("analysis.service")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def on_trade_closed(
        self,
        closed_trade_id: str,
        *,
        mfe: Optional[float] = None,
        mae: Optional[float] = None,
        news_risk_at_entry: Optional[bool] = None,
        confidence_override: Optional[float] = None,
    ) -> TradeAnalysisOutcome:
        """Run the pipeline for a single trade. Never raises."""
        outcome = TradeAnalysisOutcome(
            closed_trade_id=closed_trade_id,
            analysis=None,
            tagging=None,
            md_path=None,
            agent_summary=None,
            persisted=False,
            errors=[],
        )

        # Step 1: deterministic analysis.
        try:
            analysis = self.analyzer.analyze_closed_trade(
                closed_trade_id,
                mfe=mfe,
                mae=mae,
                news_risk_at_entry=news_risk_at_entry,
                confidence_override=confidence_override,
            )
        except Exception as e:  # pragma: no cover - analyzer is supposed to swallow
            self._record_error(outcome, "analyzer", e)
            return outcome
        if analysis is None:
            outcome.errors.append("analyzer.no_result")
            return outcome
        outcome.analysis = analysis

        # Step 2: deterministic tags.
        try:
            tagging = self.classifier.classify(analysis)
        except Exception as e:
            self._record_error(outcome, "classifier", e)
            return outcome
        outcome.tagging = tagging

        # Step 3: persist.
        try:
            self._persist(analysis, tagging)
            outcome.persisted = True
        except Exception as e:
            self._record_error(outcome, "persist", e)

        # Step 4: per-trade markdown.
        if self.write_reports:
            try:
                artifacts = write_post_trade_report(
                    analysis,
                    tagging,
                    self.settings,
                )
                outcome.md_path = artifacts.md_path
            except Exception as e:
                self._record_error(outcome, "report", e)

        # Step 5: optional LLM narration.
        if self.llm is not None:
            try:
                outcome.agent_summary = self._run_agent(analysis, tagging)
            except Exception as e:  # pragma: no cover - agent run is supposed to swallow
                self._record_error(outcome, "agent", e)

        # Step 6: Discord.
        try:
            self._notify(outcome)
        except Exception as e:
            self._record_error(outcome, "notify", e)

        return outcome

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _persist(self, analysis: PostTradeAnalysis, tagging: MistakeTagging) -> None:
        with session_scope() as session:
            row = TradeAnalysisRow(
                closed_trade_id=analysis.trade_id,
                setup_id=analysis.setup_id,
                instrument=analysis.instrument,
                strategy_name=analysis.strategy,
                direction=analysis.direction,
                entry_ts=analysis.entry_ts,
                exit_ts=analysis.exit_ts,
                result=analysis.result,
                net_pnl=float(analysis.net_pnl),
                r_multiple=float(analysis.r_multiple),
                model_confidence=analysis.model_confidence,
                risk_approved=bool(analysis.risk_approved),
                followed_plan=bool(analysis.followed_plan),
                exit_reason=str(analysis.exit_reason),
                time_of_day_bucket=analysis.time_of_day_bucket,
                volatility_regime=analysis.volatility_regime,
                market_regime=analysis.market_regime,
                news_risk_level=analysis.news_risk_level,
                mfe=analysis.mfe,
                mae=analysis.mae,
                analysis=analysis.model_dump(mode="json"),
            )
            session.add(row)
            session.flush()
            for tag in tagging.tags:
                session.add(
                    TradeMistakeTagRow(
                        trade_analysis_id=row.id,
                        closed_trade_id=analysis.trade_id,
                        tag=tag.value,
                        detail=tagging.details.get(tag),
                    )
                )

    def _run_agent(
        self, analysis: PostTradeAnalysis, tagging: MistakeTagging
    ) -> Optional[TradeAnalysisSummary]:
        if self.llm is None:
            return None
        agent = TradeAnalysisAgent(self.llm)
        ctx = AgentContext(
            settings_snapshot={
                "INSTRUMENT": self.settings.INSTRUMENT,
                "MARKET_TYPE": self.settings.MARKET_TYPE,
            },
            session_date=analysis.entry_ts.date().isoformat(),
            instrument=analysis.instrument,
            daily_report={
                "post_trade_analysis": analysis.model_dump(mode="json"),
                "mistake_tags": tagging.as_strings(),
            },
        )
        result: AgentResult = agent.run(ctx)
        # Persist every attempt — same audit pattern as Day 7 orchestrator.
        try:
            with session_scope() as session:
                session.add(
                    AgentOutputRow(
                        agent_name=result.agent_name,
                        schema_valid=bool(result.schema_valid),
                        payload=result.payload or {},
                        raw_text=result.raw_text,
                        error=result.error,
                    )
                )
        except Exception as e:
            self.log.error("analysis.agent_persist_failed", error=str(e))

        if result.schema_valid and result.payload is not None:
            try:
                return TradeAnalysisSummary(**result.payload)
            except Exception as e:  # pragma: no cover - already validated
                self.log.warning(
                    "analysis.agent_payload_unwrap_failed", error=str(e)
                )
        return None

    def _notify(self, outcome: TradeAnalysisOutcome) -> None:
        a = outcome.analysis
        if a is None:
            return
        tags = outcome.tagging.as_strings() if outcome.tagging is not None else []
        agent = outcome.agent_summary
        payload: dict[str, Any] = {
            "trade_id": a.trade_id,
            "instrument": a.instrument,
            "direction": a.direction,
            "result": a.result,
            "net_pnl": round(a.net_pnl, 2),
            "exit_reason": a.exit_reason,
            "model_confidence": (
                round(a.model_confidence, 3) if a.model_confidence is not None else None
            ),
            "mistake_tags": tags,
            "followed_plan": a.followed_plan,
            "r_multiple": round(a.r_multiple, 2),
        }
        if outcome.md_path is not None:
            payload["report_path"] = str(outcome.md_path)
        if agent is not None:
            payload["headline"] = agent.headline
            if agent.review_notes:
                payload["review_notes"] = agent.review_notes
        self.notifier.notify("trade.analysis", **payload)

    def _record_error(
        self, outcome: TradeAnalysisOutcome, kind: str, exc: Exception
    ) -> None:
        msg = f"{kind}: {exc}"
        outcome.errors.append(msg)
        self.log.error("analysis.error", kind=kind, error=str(exc))
