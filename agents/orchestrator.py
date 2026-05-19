"""End-of-day + pre-session orchestrator.

Responsibilities:

1. Build a single :class:`AgentContext` from the daily report payload + DB
   rows (model metadata, today's predictions).
2. Run each enabled agent sequentially. One agent's failure must not
   block the others.
3. Persist every :class:`AgentResult` to ``agent_outputs`` for the audit
   trail (whether ``schema_valid=True`` or not).
4. Optionally append an ``## Agent commentary`` section to the daily
   Markdown using the :class:`ReportAgent` headline.
5. Maintain an in-memory ``high_risk_news_active`` flag set by the
   :class:`NewsAgent`. The paper loop reads the flag via a callback and
   passes it into ``risk_engine.evaluate(..., high_risk_news_window=…)``
   — block-only.
6. Notify Discord with a single low-priority ``agent.summary`` digest;
   notification failures must never raise.

The orchestrator imports nothing from ``execution`` or ``risk`` —
``tests/test_agent_isolation.py`` enforces this.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select

from agents.base_agent import AgentContext, BaseAgent
from agents.llm_client import LLMClient
from agents.model_review_agent import ModelReviewAgent
from agents.news_agent import NewsAgent
from agents.providers.router import ProviderRouter
from agents.report_agent import ReportAgent
from agents.risk_explainer_agent import RiskExplainerAgent
from agents.schemas import AgentResult, NewsAssessment
from agents.trade_journal_agent import TradeJournalAgent
from app.logging_config import get_logger
from config.settings import Settings
from notifications.notification_service import NotificationService
from reports.daily_report import build_daily_report_payload
from scheduler.market_hours import session_date
from storage.db import session_scope
from storage.tables import AgentOutput as AgentOutputRow
from storage.tables import ModelMetadata as ModelMetadataRow
from storage.tables import ModelPrediction as ModelPredictionRow


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class OrchestratorResult:
    """Return value of :meth:`AgentOrchestrator.run_end_of_day`."""

    session_date: str
    results: dict[str, AgentResult] = field(default_factory=dict)
    high_risk_news: bool = False
    appended_md_path: Optional[Path] = None

    def n_valid(self) -> int:
        return sum(1 for r in self.results.values() if r.schema_valid)

    def n_total(self) -> int:
        return len(self.results)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class AgentOrchestrator:
    """Runs the agent layer at end-of-day and (optionally) pre-session.

    Constructed once per process. ``llm=None`` is a hard short-circuit —
    every public method returns an empty/no-op result without touching
    the network or the DB.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        llm: Optional[LLMClient] = None,
        notifier: NotificationService,
        provider_router: Optional[ProviderRouter] = None,
    ) -> None:
        # Two ways to wire LLM access (mutually exclusive in spirit but
        # we prefer the router when both are passed):
        # - ``provider_router``: per-agent dispatch (the new path).
        # - ``llm``: single legacy client shared by every agent. Tests
        #   that pass :class:`MockLLMClient` rely on this and must keep
        #   working unchanged.
        # When neither is given the orchestrator runs as a no-op.
        self.settings = settings
        self.llm = llm
        self.provider_router = provider_router
        self.notifier = notifier
        self.log = get_logger("agents.orchestrator")
        self._high_risk_news_active: bool = False
        if provider_router is not None:
            self.log.info(
                "agents.router_active",
                routing=provider_router.routing_table(),
            )

    # ------------------------------------------------------------------
    # Public read-only state
    # ------------------------------------------------------------------
    def high_risk_news_active(self) -> bool:
        """Latest NewsAgent verdict; default False on init or LLM disabled."""
        return self._high_risk_news_active

    # ------------------------------------------------------------------
    # Internal: per-agent client selection
    # ------------------------------------------------------------------
    def _client_for(self, agent_name: str) -> Optional[LLMClient]:
        """Pick the LLM client for a given agent.

        Resolution order:

        1. If a :class:`ProviderRouter` is wired, ask it. Returns
           ``None`` if the agent is disabled or its API key is missing
           — the orchestrator skips such agents and persists a clear
           ``AgentResult(schema_valid=False, error="agent.disabled")``
           row for the audit trail.
        2. Otherwise fall back to the single-client legacy path
           (``self.llm``), which preserves test-suite behavior with
           ``MockLLMClient``.
        """
        if self.provider_router is not None:
            return self.provider_router.client_for(agent_name)
        return self.llm

    def _has_any_llm(self) -> bool:
        if self.provider_router is not None:
            return self.provider_router.has_any_enabled()
        return self.llm is not None

    # ------------------------------------------------------------------
    # Pre-session news
    # ------------------------------------------------------------------
    def run_pre_session_news(
        self, *, now: Optional[datetime] = None
    ) -> Optional[NewsAssessment]:
        """Run only the NewsAgent and update the in-memory flag.

        Returns the validated :class:`NewsAssessment` on success, ``None``
        if disabled or the agent failed schema validation. Failures
        leave the previous flag unchanged so a transient LLM hiccup
        does not silently *unblock* trading mid-day.
        """
        client = self._client_for("news")
        if client is None:
            self.log.info("agents.disabled", phase="pre_session_news")
            return None
        now = now or datetime.now(tz=timezone.utc)
        sd = session_date(now, self.settings).isoformat()
        ctx = AgentContext(
            settings_snapshot=self._settings_snapshot(),
            session_date=sd,
            instrument=self.settings.INSTRUMENT,
        )
        agent = NewsAgent(client)
        result = agent.run(ctx)
        self._persist_result(result)
        if result.schema_valid and result.payload is not None:
            high_risk = bool(result.payload.get("high_risk_window", False))
            self._high_risk_news_active = high_risk
            self.log.info(
                "agents.news.flag_set",
                high_risk_news_active=high_risk,
                severity=result.payload.get("severity"),
            )
            if high_risk:
                self._safe_notify(
                    "high_risk_news",
                    severity=result.payload.get("severity"),
                    summary=result.payload.get("summary"),
                )
            return NewsAssessment(**result.payload)
        self.log.warning(
            "agents.news.failed",
            error=result.error,
            keep_previous_flag=self._high_risk_news_active,
        )
        return None

    # ------------------------------------------------------------------
    # End of day
    # ------------------------------------------------------------------
    def run_end_of_day(
        self,
        *,
        now: Optional[datetime] = None,
        daily_report_payload: Optional[dict[str, Any]] = None,
        daily_md_path: Optional[Path] = None,
    ) -> OrchestratorResult:
        """Run the full agent suite for today's session.

        ``daily_report_payload`` may be passed in if the caller already
        ran ``write_daily_report`` and wants to avoid re-querying the DB
        (the scheduler does this). Otherwise we rebuild it.
        """
        now = now or datetime.now(tz=timezone.utc)
        sd = session_date(now, self.settings).isoformat()
        result = OrchestratorResult(session_date=sd)
        if not self._has_any_llm():
            self.log.info("agents.disabled", phase="end_of_day")
            return result

        if daily_report_payload is None:
            daily_report_payload = build_daily_report_payload(self.settings, now=now)

        context = AgentContext(
            settings_snapshot=self._settings_snapshot(),
            session_date=sd,
            instrument=self.settings.INSTRUMENT,
            trades=daily_report_payload.get("trades", []) or [],
            risk_blocks=daily_report_payload.get("risk_blocks", []) or [],
            daily_report=daily_report_payload,
            model_metadata=self._load_latest_model_metadata(now=now),
        )

        # Each agent is constructed with its *own* LLM client (resolved
        # via the router). Agents whose provider is disabled or whose
        # API key is missing are skipped with a clear ``AgentResult``
        # row so the audit trail still records the absence.
        agent_specs: list[tuple[str, type[BaseAgent]]] = [
            ("news", NewsAgent),
            ("risk_explainer", RiskExplainerAgent),
            ("trade_journal", TradeJournalAgent),
            ("report", ReportAgent),
            ("model_review", ModelReviewAgent),
        ]
        for agent_name, agent_cls in agent_specs:
            client = self._client_for(agent_name)
            if client is None:
                ar = AgentResult(
                    agent_name=agent_name,
                    schema_valid=False,
                    payload=None,
                    raw_text=None,
                    error="agent.disabled: no LLM client (missing key or off)",
                )
                self.log.info(
                    "agents.skipped_no_provider",
                    agent=agent_name,
                )
                result.results[agent_name] = ar
                self._persist_result(ar)
                continue
            agent = agent_cls(client)
            try:
                ar = agent.run(context)
            except Exception as e:  # pragma: no cover - run() is supposed to catch
                self.log.error("agents.run_unexpected", agent=agent.name, error=str(e))
                ar = AgentResult(
                    agent_name=agent.name,
                    schema_valid=False,
                    payload=None,
                    raw_text=None,
                    error=f"unexpected: {e}",
                )
            result.results[agent.name] = ar
            self._persist_result(ar)

        # Update high-risk-news flag from the EOD News output.
        news = result.results.get("news")
        if news is not None and news.schema_valid and news.payload is not None:
            self._high_risk_news_active = bool(news.payload.get("high_risk_window", False))
        result.high_risk_news = self._high_risk_news_active

        # Optional polish: append commentary to the daily Markdown.
        if daily_md_path is not None:
            try:
                appended = self._append_commentary_to_md(result, daily_md_path)
                if appended is not None:
                    result.appended_md_path = appended
            except Exception as e:
                self.log.warning("agents.md_append_failed", error=str(e))

        # Optional digest notification.
        report = result.results.get("report")
        if report is not None and report.schema_valid and report.payload is not None:
            self._safe_notify(
                "agent.summary",
                headline=report.payload.get("headline"),
                tomorrow_focus=report.payload.get("tomorrow_focus"),
                high_risk_news=result.high_risk_news,
                n_valid=result.n_valid(),
                n_total=result.n_total(),
            )
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _settings_snapshot(self) -> dict[str, Any]:
        s = self.settings
        return {
            "INSTRUMENT": s.INSTRUMENT,
            "MARKET_TYPE": s.MARKET_TYPE,
            "TIMEZONE": s.TIMEZONE,
            "MODE": s.MODE,
            "trading_window_start": str(s.trading_window_start_time()),
            "trading_window_end": str(s.trading_window_end_time()),
            "force_flat_time": str(s.force_flat_time()),
            "max_trades_per_day": s.MAX_TRADES_PER_DAY,
            "max_daily_loss": s.MAX_DAILY_LOSS,
            "max_daily_profit": s.MAX_DAILY_PROFIT,
            "risk_per_trade": s.RISK_PER_TRADE,
            "confidence_threshold": s.CONFIDENCE_THRESHOLD,
        }

    def _load_latest_model_metadata(
        self, *, now: datetime
    ) -> Optional[dict[str, Any]]:
        try:
            with session_scope() as session:
                row = session.execute(
                    select(ModelMetadataRow)
                    .order_by(ModelMetadataRow.trained_at.desc())
                    .limit(1)
                ).scalar_one_or_none()
                if row is None:
                    return None
                meta: dict[str, Any] = {
                    "name": row.model_name,
                    "version": row.version,
                    "metrics": dict(row.metrics or {}),
                    "feature_columns": list(row.features or []),
                }
                # Today's predictions (all instruments — model is global per name).
                start_utc = datetime.fromisoformat(now.date().isoformat()).replace(
                    tzinfo=timezone.utc
                )
                preds = session.execute(
                    select(ModelPredictionRow)
                    .where(ModelPredictionRow.created_at >= start_utc)
                    .order_by(ModelPredictionRow.created_at.asc())
                    .limit(200)
                ).scalars().all()
                meta["today_predictions"] = [
                    {
                        "probability": float(p.probability),
                        "threshold": float(p.threshold),
                        "approved": bool(p.approved),
                    }
                    for p in preds
                ]
                return meta
        except Exception as e:
            self.log.warning("agents.model_metadata_failed", error=str(e))
            return None

    def _persist_result(self, result: AgentResult) -> None:
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
            # Persistence failures must never raise — they are recorded
            # only in the structured log.
            self.log.error("agents.persist_failed", agent=result.agent_name, error=str(e))

    def _append_commentary_to_md(
        self, result: OrchestratorResult, md_path: Path
    ) -> Optional[Path]:
        report = result.results.get("report")
        if report is None or not report.schema_valid or report.payload is None:
            return None
        if not md_path.exists():
            self.log.warning("agents.md_missing", path=str(md_path))
            return None

        bullets = report.payload.get("bullets") or []
        compliance_notes = report.payload.get("compliance_notes") or []
        section_lines = [
            "",
            "## Agent commentary",
            "",
            f"_{report.payload.get('headline', '')}_",
            "",
        ]
        if bullets:
            section_lines.extend(f"- {b}" for b in bullets)
            section_lines.append("")
        if compliance_notes:
            section_lines.append("**Compliance notes:**")
            section_lines.extend(f"- {c}" for c in compliance_notes)
            section_lines.append("")
        focus = report.payload.get("tomorrow_focus")
        if focus:
            section_lines.append(f"**Tomorrow focus:** {focus}")
            section_lines.append("")

        with md_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(section_lines))
        return md_path

    def _safe_notify(self, kind: str, /, **payload: Any) -> None:
        try:
            self.notifier.notify(kind, **payload)
        except Exception as e:  # pragma: no cover - notify is supposed to catch
            self.log.warning("agents.notify_failed", kind=kind, error=str(e))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_orchestrator(
    settings: Settings,
    *,
    notifier: NotificationService,
    llm: Optional[LLMClient] = None,
    provider_router: Optional[ProviderRouter] = None,
) -> AgentOrchestrator:
    """Build an :class:`AgentOrchestrator`.

    Three resolution paths:

    1. ``llm`` is passed in explicitly: legacy single-client mode.
       Used by tests that wire :class:`agents.llm_client.MockLLMClient`.
    2. ``provider_router`` is passed in: per-agent dispatch.
    3. Neither is passed and ``ENABLE_LLM_AGENTS=true``: build a
       :class:`ProviderRouter` from settings. If at least one
       configured agent has a working provider key, we use it. If
       nothing is wired (no keys at all) we fall back to the legacy
       :func:`agents.llm_client.build_llm_client` path so single-key
       OpenAI deployments keep working unchanged.

    The orchestrator returns no-ops when neither ``llm`` nor a
    populated router is available, so callers can always construct
    one safely.
    """
    if llm is not None:
        return AgentOrchestrator(settings, llm=llm, notifier=notifier)

    if provider_router is not None:
        return AgentOrchestrator(
            settings, provider_router=provider_router, notifier=notifier
        )

    log = get_logger("agents.orchestrator")

    if not settings.ENABLE_LLM_AGENTS:
        log.info("agents.disabled", reason="ENABLE_LLM_AGENTS=false")
        return AgentOrchestrator(settings, llm=None, notifier=notifier)

    router = ProviderRouter.from_settings(settings)
    if router.has_any_enabled():
        return AgentOrchestrator(
            settings, provider_router=router, notifier=notifier
        )

    # Backward compatibility: no per-provider keys configured but the
    # legacy ``OPENAI_API_KEY`` path may still be wired.
    from agents.llm_client import build_llm_client

    legacy = build_llm_client(settings)
    if legacy is None:
        log.info(
            "agents.disabled", reason="no_provider_keys_configured"
        )
    return AgentOrchestrator(settings, llm=legacy, notifier=notifier)
