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
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select

from agents.backtest_critic_agent import BacktestCriticAgent
from agents.base_agent import AgentContext, BaseAgent
from agents.data_quality_agent import DataQualityAgent
from agents.llm_client import LLMClient, LLMClientError
from agents.macro_news_agent import MacroNewsAgent
from agents.model_drift_agent import ModelDriftAgent
from agents.model_review_agent import ModelReviewAgent
from agents.news_agent import NewsAgent
from agents.providers.router import ProviderRouter
from agents.report_agent import ReportAgent
from agents.risk_explainer_agent import RiskExplainerAgent
from agents.schemas import (
    AgentResult,
    DataQualityReport,
    MacroNewsAssessment,
    ModelDriftReport,
    NewsAssessment,
    StrategyResearchReport,
)
from agents.strategy_research_agent import StrategyResearchAgent
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
    # On-demand agent entry points (called by workflows / scheduler)
    # ------------------------------------------------------------------
    def run_macro_news(
        self,
        *,
        now: Optional[datetime] = None,
        enabled_symbols: Optional[list[str]] = None,
        news_headlines: Optional[list[str]] = None,
    ) -> Optional[MacroNewsAssessment]:
        """Run :class:`MacroNewsAgent` and update the high-risk-news flag.

        Returns the validated :class:`MacroNewsAssessment` on success,
        ``None`` if the agent is disabled / failed schema. ``high``
        risk_level OR any populated ``blocked_windows`` flips the
        ``high_risk_news_active`` flag — block-only.
        """
        client = self._client_for("macro_news")
        if client is None:
            self.log.info("agents.disabled", phase="macro_news")
            return None
        now = now or datetime.now(tz=timezone.utc)
        sd = session_date(now, self.settings).isoformat()
        symbols = list(enabled_symbols or [])
        if not symbols:
            symbols = list(getattr(self.settings, "ENABLED_SYMBOLS", []) or [])
        if not symbols:
            symbols = [self.settings.INSTRUMENT]
        ctx = AgentContext(
            settings_snapshot=self._settings_snapshot(),
            session_date=sd,
            instrument=self.settings.INSTRUMENT,
            news_headlines=list(news_headlines or []),
            enabled_symbols=symbols,
        )
        agent = MacroNewsAgent(client)
        result = agent.run(ctx)
        self._persist_result(result)
        if not (result.schema_valid and result.payload):
            self.log.warning(
                "agents.macro_news.failed",
                error=result.error,
                keep_previous_flag=self._high_risk_news_active,
            )
            return None
        try:
            assessment = MacroNewsAssessment.model_validate(result.payload)
        except Exception as e:  # noqa: BLE001
            self.log.warning("agents.macro_news.bad_payload", error=str(e))
            return None
        # Filter affected_symbols to the operator-configured universe;
        # the LLM is forbidden from inventing tickers and we enforce
        # that here before the value reaches anything operational.
        allowed = {s.upper() for s in symbols}
        scoped = [s for s in assessment.affected_symbols if s.upper() in allowed]
        flag = (
            assessment.risk_level == "high"
            or bool(assessment.blocked_windows)
        )
        self._high_risk_news_active = bool(flag)
        self.log.info(
            "agents.macro_news.flag_set",
            high_risk_news_active=self._high_risk_news_active,
            risk_level=assessment.risk_level,
            affected_symbols=scoped,
        )
        if flag:
            self._safe_notify(
                "high_risk_news",
                source="macro_news",
                risk_level=assessment.risk_level,
                affected_symbols=scoped,
                summary=assessment.summary,
            )
        return assessment

    def run_data_quality_check(
        self,
        feeds_by_symbol: Optional[dict[str, Any]] = None,
        *,
        precomputed: Optional[dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> Optional[DataQualityReport]:
        """Run the deterministic :class:`DataQualityAgent` pre-paper hook.

        Two paths:

        - ``feeds_by_symbol`` (mapping symbol -> OHLCV DataFrame) is
          the typical paper-mode pre-flight call.
        - ``precomputed`` is used when the scan has already been
          performed elsewhere (e.g. by the scheduler) and we just
          want to persist + notify.

        Returns the validated :class:`DataQualityReport` on success,
        ``None`` on failure. The orchestrator does not raise — paper
        mode reads ``report.blocked_symbols`` from the return value.
        """
        agent = DataQualityAgent()
        if feeds_by_symbol is not None:
            now_utc = (now or datetime.now(tz=timezone.utc))
            sd = session_date(now_utc, self.settings).isoformat()
            result = agent.run_with_feeds(
                feeds_by_symbol,
                now=now_utc,
                session_date=sd,
            )
        else:
            ctx = AgentContext(
                settings_snapshot=self._settings_snapshot(),
                session_date=(
                    session_date(
                        now or datetime.now(tz=timezone.utc), self.settings
                    ).isoformat()
                ),
                instrument=self.settings.INSTRUMENT,
                data_quality=precomputed,
            )
            result = agent.run(ctx)

        self._persist_result(result)
        if not (result.schema_valid and result.payload):
            return None
        try:
            report = DataQualityReport.model_validate(result.payload)
        except Exception as e:  # noqa: BLE001
            self.log.warning(
                "agents.data_quality.bad_payload", error=str(e)
            )
            return None
        if report.blocked_symbols:
            self._safe_notify(
                "data_quality.blocked",
                blocked=list(report.blocked_symbols),
                checked=list(report.checked_symbols),
                summary=report.summary,
            )
        return report

    def run_model_drift_review(
        self,
        *,
        now: Optional[datetime] = None,
        paper_metrics: Optional[dict[str, Any]] = None,
    ) -> Optional[ModelDriftReport]:
        """Run :class:`ModelDriftAgent` (stats-first, optional LLM narrative).

        Used by daily / weekly reviews. The deterministic stats path
        runs even when no LLM is configured for ``model_drift``.
        """
        now = now or datetime.now(tz=timezone.utc)
        sd = session_date(now, self.settings).isoformat()
        observed = (
            paper_metrics
            if paper_metrics is not None
            else build_daily_report_payload(self.settings, now=now).get(
                "metrics"
            )
        )
        ctx = AgentContext(
            settings_snapshot=self._settings_snapshot(),
            session_date=sd,
            instrument=self.settings.INSTRUMENT,
            paper_metrics=observed,
            model_metadata=self._load_latest_model_metadata(now=now),
        )
        # LLM polish is optional; ``client_for`` returning ``None`` is
        # the normal "stats-only" path and not an error.
        client = self._client_for("model_drift")
        agent = ModelDriftAgent(client)
        result = agent.run(ctx)
        self._persist_result(result)
        if not (result.schema_valid and result.payload):
            return None
        try:
            report = ModelDriftReport.model_validate(result.payload)
        except Exception as e:  # noqa: BLE001
            self.log.warning(
                "agents.model_drift.bad_payload", error=str(e)
            )
            return None
        if report.severity in ("warn", "alert"):
            self._safe_notify(
                "model_drift",
                severity=report.severity,
                retrain_recommended=report.retrain_recommended,
                reason=report.reason,
            )
        return report

    def run_strategy_research(
        self,
        *,
        now: Optional[datetime] = None,
        backtest_summary: Optional[dict[str, Any]] = None,
        enabled_symbols: Optional[list[str]] = None,
    ) -> Optional[StrategyResearchReport]:
        """Run :class:`StrategyResearchAgent` for off-cycle ideation."""
        client = self._client_for("strategy_research")
        if client is None:
            self.log.info("agents.disabled", phase="strategy_research")
            return None
        now = now or datetime.now(tz=timezone.utc)
        sd = session_date(now, self.settings).isoformat()
        symbols = list(enabled_symbols or []) or list(
            getattr(self.settings, "ENABLED_SYMBOLS", [])
            or [self.settings.INSTRUMENT]
        )
        ctx = AgentContext(
            settings_snapshot=self._settings_snapshot(),
            session_date=sd,
            instrument=self.settings.INSTRUMENT,
            backtest_summary=backtest_summary or {},
            enabled_symbols=symbols,
        )
        agent = StrategyResearchAgent(client)
        result = agent.run(ctx)
        self._persist_result(result)
        if not (result.schema_valid and result.payload):
            return None
        try:
            return StrategyResearchReport.model_validate(result.payload)
        except Exception as e:  # noqa: BLE001
            self.log.warning(
                "agents.strategy_research.bad_payload", error=str(e)
            )
            return None

    def propose_watchlist_symbols(
        self,
        *,
        allowlist: list[str],
        max_symbols: int = 12,
        now: Optional[datetime] = None,
    ) -> list[str]:
        """Advisory watchlist: rank which *allowlist* names to prioritize.

        Uses the web-grounded research provider (Perplexity by default via
        the ``strategy_research`` route) to surface names with notable
        catalysts/relative strength today. This is strictly advisory:

        - It can only return symbols that are already on ``allowlist`` —
          the caller (and this method) filter anything else out, so the
          LLM cannot inject an arbitrary ticker.
        - It never places a trade, changes risk, or promotes a model.
        - Any failure returns ``[]`` and the loop falls back to the
          deterministic allowlist order.
        """
        # Use the fast web-grounded route (macro_news → sonar-pro) rather
        # than the heavy strategy_research model (sonar-deep-research),
        # which is too slow for a per-session watchlist and would time out.
        client = self._client_for("macro_news") or self._client_for("news")
        if client is None:
            self.log.info("agents.disabled", phase="watchlist")
            return []
        allowed = {s.upper() for s in allowlist}
        if not allowed:
            return []
        system = (
            "You are an advisory market scanner for a paper-trading bot. "
            "From the provided allowlist of liquid US equities/ETFs, pick "
            "the names most worth scanning intraday today based on notable "
            "catalysts, news, or relative strength. You do NOT place trades "
            "or give buy/sell advice — you only narrow a scan list. "
            "Respond with STRICT JSON: {\"symbols\": [\"TICKER\", ...]}. "
            "Only use tickers from the allowlist. No prose."
        )
        user = (
            f"Allowlist: {sorted(allowed)}\n"
            f"Return at most {max_symbols} tickers, highest priority first."
        )
        try:
            raw = client.complete(system=system, user=user)
        except LLMClientError as e:
            self.log.warning("agents.watchlist.llm_failed", error=str(e))
            return []
        except Exception as e:  # noqa: BLE001
            self.log.warning("agents.watchlist.unexpected", error=str(e))
            return []

        tickers = _extract_ticker_list(raw)
        picked = [t for t in tickers if t.upper() in allowed][:max_symbols]
        self.log.info(
            "agents.watchlist.proposed",
            n=len(picked),
            picked=picked,
        )
        return picked

    def run_backtest_critic(
        self,
        backtest_summary: dict[str, Any],
        *,
        now: Optional[datetime] = None,
    ) -> Optional[AgentResult]:
        """Run :class:`BacktestCriticAgent` on a freshly-completed backtest.

        Returns the raw :class:`AgentResult` so callers can inspect
        the validated payload (with weak spots + experiment list) or
        the validation error. None when the agent is disabled.
        """
        client = self._client_for("backtest_critic")
        if client is None:
            self.log.info("agents.disabled", phase="backtest_critic")
            return None
        now = now or datetime.now(tz=timezone.utc)
        sd = session_date(now, self.settings).isoformat()
        daily = build_daily_report_payload(self.settings, now=now) or {}
        ctx = AgentContext(
            settings_snapshot=self._settings_snapshot(),
            session_date=sd,
            instrument=self.settings.INSTRUMENT,
            backtest_summary=backtest_summary or {},
            paper_metrics=daily.get("metrics"),
            daily_report=daily,
        )
        agent = BacktestCriticAgent(client)
        result = agent.run(ctx)
        self._persist_result(result)
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


_TICKER_RE = re.compile(r"\b[A-Z]{1,6}\b")


def _extract_ticker_list(raw: str) -> list[str]:
    """Parse tickers from an LLM watchlist response, defensively.

    Accepts strict JSON ``{"symbols": [...]}``, a bare JSON array, or
    (last resort) a regex sweep of uppercase tokens. Never raises.
    """
    if not raw:
        return []
    text = raw.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("symbols"), list):
            return [str(s).strip().upper() for s in data["symbols"] if str(s).strip()]
        if isinstance(data, list):
            return [str(s).strip().upper() for s in data if str(s).strip()]
    except (ValueError, TypeError):
        pass
    # Last resort: sweep already-uppercase tokens (likely tickers),
    # filtered against the allowlist by the caller anyway.
    return list(dict.fromkeys(_TICKER_RE.findall(text)))


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
