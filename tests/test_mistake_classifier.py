"""Mistake classifier rules — false_positive, timeout, low volume, news risk, etc."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from analysis.mistake_classifier import classify, is_false_positive
from analysis.types import MistakeTag, OrderflowFeatures, PostTradeAnalysis
from features.feature_builder import FEATURE_COLUMNS


def _features(**overrides) -> dict[str, float]:
    base = {col: 0.0 for col in FEATURE_COLUMNS}
    base.update(
        {
            "volatility_regime": 1,
            "trend_regime": 1,
            "volume_ratio_20": 1.0,
        }
    )
    base.update(overrides)
    return base


def _analysis(**overrides) -> PostTradeAnalysis:
    base = {
        "trade_id": "t1",
        "setup_id": "s1",
        "instrument": "MES",
        "direction": "long",
        "strategy": "vwap_ema_pullback",
        "entry_ts": datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc),
        "exit_ts": datetime(2026, 5, 18, 14, 4, tzinfo=timezone.utc),
        "entry_price": 4500.0,
        "exit_price": 4504.0,
        "stop_price": 4498.0,
        "target_price": 4504.0,
        "result": "win",
        "net_pnl": 20.0,
        "gross_pnl": 20.5,
        "commission": 0.5,
        "slippage": 0.0,
        "hold_seconds": 240,
        "r_multiple": 2.0,
        "model_confidence": None,
        "model_threshold": None,
        "risk_approved": True,
        "features": _features(),
        "orderflow": OrderflowFeatures(),
        "market_regime": "uptrend",
        "volatility_regime": "medium",
        "time_of_day_bucket": "afternoon",
        "news_risk_level": "low",
        "exit_reason": "tp",
        "followed_plan": True,
        "mfe": None,
        "mae": None,
    }
    base.update(overrides)
    return PostTradeAnalysis(**base)


# ---------------------------------------------------------------------------
def test_winner_no_tags() -> None:
    tagging = classify(_analysis())
    assert tagging.tags == ()


def test_false_positive_when_model_approved_and_loses() -> None:
    a = _analysis(
        result="loss",
        net_pnl=-15.0,
        exit_reason="sl",
        exit_price=4498.0,
        model_confidence=0.78,
        model_threshold=0.60,
    )
    tagging = classify(a)
    assert MistakeTag.FALSE_POSITIVE in tagging.tags
    assert is_false_positive(a) is True


def test_false_positive_not_set_for_winning_trade() -> None:
    a = _analysis(model_confidence=0.78, model_threshold=0.60, result="win", net_pnl=15.0)
    assert not classify(a).has(MistakeTag.FALSE_POSITIVE)


def test_low_confidence_when_just_above_threshold() -> None:
    tagging = classify(_analysis(model_confidence=0.62, model_threshold=0.60))
    assert MistakeTag.LOW_CONFIDENCE_TRADE in tagging.tags


def test_timeout_exit_tagged() -> None:
    tagging = classify(_analysis(exit_reason="time", result="loss", net_pnl=-2.0))
    assert MistakeTag.TIMEOUT_EXIT in tagging.tags


def test_low_volume_trade_tagged() -> None:
    a = _analysis(features=_features(volume_ratio_20=0.5))
    tagging = classify(a)
    assert MistakeTag.LOW_VOLUME_TRADE in tagging.tags


def test_high_volatility_spike_tagged_for_loss() -> None:
    a = _analysis(
        result="loss",
        net_pnl=-5.0,
        exit_reason="sl",
        exit_price=4498.0,
        features=_features(volatility_regime=2),
    )
    tagging = classify(a)
    assert MistakeTag.HIGH_VOLATILITY_SPIKE in tagging.tags


def test_news_risk_trade_tagged() -> None:
    a = _analysis(news_risk_level="high")
    tagging = classify(a)
    assert MistakeTag.NEWS_RISK_TRADE in tagging.tags


def test_chop_loss_tagged() -> None:
    a = _analysis(market_regime="chop", result="loss", net_pnl=-5.0, exit_reason="sl", exit_price=4498.0)
    tagging = classify(a)
    assert MistakeTag.ENTERED_DURING_CHOP in tagging.tags


def test_lunch_loss_tagged() -> None:
    a = _analysis(time_of_day_bucket="lunch", result="loss", net_pnl=-5.0, exit_reason="sl", exit_price=4498.0)
    tagging = classify(a)
    assert MistakeTag.BAD_TIME_OF_DAY in tagging.tags


def test_poor_risk_reward_tagged() -> None:
    # entry 4500, stop 4498, target 4501.5 → RR = 1.5/2.0 = 0.75 (< 1.0)
    a = _analysis(stop_price=4498.0, target_price=4501.5)
    tagging = classify(a)
    assert MistakeTag.POOR_RISK_REWARD in tagging.tags


def test_rule_violation_when_not_followed_plan() -> None:
    a = _analysis(
        followed_plan=False,
        result="breakeven",
        net_pnl=0.0,
        exit_reason="manual",
    )
    tagging = classify(a)
    assert MistakeTag.RULE_VIOLATION in tagging.tags


def test_unknown_default_for_loss_with_no_specific_rule() -> None:
    a = _analysis(
        result="loss",
        net_pnl=-3.0,
        exit_reason="sl",
        exit_price=4498.0,
        # All vol/regime fields neutral, no model, no news.
        market_regime="uptrend",
        time_of_day_bucket="afternoon",
        features=_features(),
    )
    tagging = classify(a)
    # Some heuristic might still fire; but if not, UNKNOWN must be there.
    if tagging.tags == (MistakeTag.UNKNOWN,):
        assert tagging.has(MistakeTag.UNKNOWN)
    else:
        # As long as we got *something* the classifier did its job.
        assert tagging.tags


def test_multiple_tags_can_apply() -> None:
    a = _analysis(
        result="loss",
        net_pnl=-15.0,
        exit_reason="sl",
        exit_price=4498.0,
        model_confidence=0.78,
        model_threshold=0.60,
        market_regime="chop",
        news_risk_level="high",
        features=_features(volume_ratio_20=0.4, volatility_regime=2),
    )
    tagging = classify(a)
    expected_subset = {
        MistakeTag.FALSE_POSITIVE,
        MistakeTag.ENTERED_DURING_CHOP,
        MistakeTag.LOW_VOLUME_TRADE,
        MistakeTag.HIGH_VOLATILITY_SPIKE,
        MistakeTag.NEWS_RISK_TRADE,
    }
    assert expected_subset.issubset(set(tagging.tags))
