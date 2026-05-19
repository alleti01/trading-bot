"""Strict-validation tests for the Day 7 agent output schemas."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agents.schemas import (
    AGENT_SCHEMAS,
    AgentResult,
    ModelReviewOutput,
    NewsAssessment,
    ReportCommentary,
    RiskBlockExplanation,
    RiskExplainerOutput,
    TradeJournalNarrative,
)


# ---------------------------------------------------------------------------
# NewsAssessment
# ---------------------------------------------------------------------------
def test_news_assessment_accepts_valid_json() -> None:
    payload = {
        "high_risk_window": True,
        "severity": "high",
        "events": ["FOMC"],
        "summary": "FOMC at 14:00 ET",
        "recommendation": "Stand aside through 14:30 ET.",
    }
    parsed = NewsAssessment.model_validate(payload)
    assert parsed.high_risk_window is True
    assert parsed.severity == "high"


def test_news_assessment_rejects_invalid_severity() -> None:
    with pytest.raises(ValidationError):
        NewsAssessment.model_validate(
            {
                "high_risk_window": False,
                "severity": "extreme",  # not in Literal["low","medium","high"]
                "events": [],
                "summary": "x",
                "recommendation": "y",
            }
        )


def test_news_assessment_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        NewsAssessment.model_validate(
            {
                "high_risk_window": False,
                "severity": "low",
                "events": [],
                "summary": "x",
                "recommendation": "y",
                "unexpected_field": 1,
            }
        )


def test_news_assessment_is_frozen() -> None:
    parsed = NewsAssessment(
        high_risk_window=False,
        severity="low",
        events=[],
        summary="x",
        recommendation="y",
    )
    with pytest.raises(ValidationError):
        parsed.high_risk_window = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RiskExplainerOutput
# ---------------------------------------------------------------------------
def test_risk_explainer_accepts_nested_blocks() -> None:
    parsed = RiskExplainerOutput.model_validate(
        {
            "session_date": "2026-05-18",
            "blocks": [
                {"rule": "max_trades_per_day", "count": 1, "explanation": "Hit cap."}
            ],
            "overall_assessment": "ok",
            "operator_actions": [],
        }
    )
    assert isinstance(parsed.blocks[0], RiskBlockExplanation)
    assert parsed.blocks[0].rule == "max_trades_per_day"


def test_risk_explainer_rejects_negative_count() -> None:
    with pytest.raises(ValidationError):
        RiskExplainerOutput.model_validate(
            {
                "session_date": "2026-05-18",
                "blocks": [{"rule": "x", "count": -1, "explanation": "y"}],
                "overall_assessment": "z",
                "operator_actions": [],
            }
        )


# ---------------------------------------------------------------------------
# Trade journal / report / model review
# ---------------------------------------------------------------------------
def test_trade_journal_allows_optional_setup_ids() -> None:
    parsed = TradeJournalNarrative.model_validate(
        {
            "session_date": "2026-05-18",
            "highlights": ["x"],
            "mistakes": [],
            "lessons": [],
            "best_trade_setup_id": None,
            "worst_trade_setup_id": "abc",
        }
    )
    assert parsed.best_trade_setup_id is None
    assert parsed.worst_trade_setup_id == "abc"


def test_report_commentary_required_fields() -> None:
    with pytest.raises(ValidationError):
        ReportCommentary.model_validate(
            {
                "session_date": "2026-05-18",
                # missing headline
                "bullets": [],
                "compliance_notes": [],
                "tomorrow_focus": "z",
            }
        )


def test_model_review_retrain_required_bool() -> None:
    parsed = ModelReviewOutput.model_validate(
        {
            "model_name": "vwap",
            "model_version": "v1",
            "calibration_comment": "ok",
            "drift_warnings": [],
            "retrain_recommended": False,
            "reason": "no evidence",
        }
    )
    assert parsed.retrain_recommended is False


def test_agent_schemas_map_covers_known_agents() -> None:
    """Day 7 (5 agents) + Day 8 trade_analysis + the autonomous-paper
    additions (macro_news / backtest_critic / model_drift /
    strategy_research / data_quality)."""
    assert set(AGENT_SCHEMAS.keys()) == {
        "news",
        "risk_explainer",
        "trade_journal",
        "report",
        "model_review",
        "trade_analysis",
        "macro_news",
        "backtest_critic",
        "model_drift",
        "strategy_research",
        "data_quality",
    }


# ---------------------------------------------------------------------------
# AgentResult wrapper
# ---------------------------------------------------------------------------
def test_agent_result_valid_with_payload() -> None:
    r = AgentResult(
        agent_name="news",
        schema_valid=True,
        payload={"x": 1},
        raw_text="{...}",
        error=None,
    )
    assert r.schema_valid is True
    assert r.payload == {"x": 1}


def test_agent_result_invalid_with_error() -> None:
    r = AgentResult(
        agent_name="news",
        schema_valid=False,
        payload=None,
        raw_text="bad",
        error="schema_invalid: ...",
    )
    assert r.schema_valid is False
    assert r.payload is None
    assert "schema_invalid" in (r.error or "")


def test_agent_result_serializes_to_json() -> None:
    r = AgentResult(agent_name="news", schema_valid=True, payload={"x": 1})
    blob = r.model_dump_json()
    assert "news" in blob
    assert json.loads(blob)["payload"] == {"x": 1}
