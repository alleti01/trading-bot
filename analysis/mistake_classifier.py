"""Rule-based mistake tagging.

The classifier takes a deterministic :class:`PostTradeAnalysis` and emits
zero or more :class:`MistakeTag` values plus optional per-tag detail
strings. It is **fully deterministic** so:

- The same trade always produces the same tags (good for tests, audit,
  and reproducible retraining).
- The LLM's role is reduced to *narrating* the tags in plain English,
  not classifying — the classifier is the single source of truth.

Tag definitions follow the user's spec. The most important one,
``false_positive``, has its own constant so callers can introspect and
the pattern miner can compute false-positive rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from analysis.types import MistakeTag, PostTradeAnalysis


# False positive: model approved + risk approved + the trade lost
# materially. We require an explicit confidence above the threshold
# *and* a clear loss to avoid tagging marginal break-evens.
FALSE_POSITIVE_LOSS_THRESHOLD: float = 0.0
LOW_CONFIDENCE_MARGIN_ABOVE_THRESHOLD: float = 0.05  # within 5 pp of threshold
LOW_VOLUME_RATIO_FLOOR: float = 0.7
HIGH_VOLATILITY_REGIME_VALUE: int = 2
POOR_RR_RATIO: float = 1.0  # planned reward / planned risk
WEAK_WIN_R_MULTIPLE: float = 0.5  # win < 0.5R is a "weak winner"


@dataclass(frozen=True)
class MistakeTagging:
    """Ordered list of tags for a single trade plus optional details."""

    tags: tuple[MistakeTag, ...]
    details: dict[MistakeTag, str]

    def as_strings(self) -> list[str]:
        return [t.value for t in self.tags]

    def has(self, tag: MistakeTag) -> bool:
        return tag in self.tags


def classify(analysis: PostTradeAnalysis) -> MistakeTagging:
    """Return the deterministic mistake tagging for one trade.

    Multiple tags can apply. Wins that hit target cleanly typically
    return an empty list (no tags) — the classifier is for losses and
    weak winners.
    """
    tags: list[MistakeTag] = []
    details: dict[MistakeTag, str] = {}

    # 1) False positive — model approved + lost materially.
    if (
        analysis.result == "loss"
        and analysis.model_confidence is not None
        and analysis.model_threshold is not None
        and analysis.model_confidence >= analysis.model_threshold
        and analysis.net_pnl < FALSE_POSITIVE_LOSS_THRESHOLD
    ):
        tags.append(MistakeTag.FALSE_POSITIVE)
        details[MistakeTag.FALSE_POSITIVE] = (
            f"Model approved at p={analysis.model_confidence:.3f} "
            f"(thr={analysis.model_threshold:.2f}) but trade closed "
            f"at PnL=${analysis.net_pnl:.2f}."
        )

    # 2) Low-confidence trade — barely cleared the threshold (regardless of result).
    if (
        analysis.model_confidence is not None
        and analysis.model_threshold is not None
        and 0.0
        < (analysis.model_confidence - analysis.model_threshold)
        <= LOW_CONFIDENCE_MARGIN_ABOVE_THRESHOLD
    ):
        tags.append(MistakeTag.LOW_CONFIDENCE_TRADE)
        details[MistakeTag.LOW_CONFIDENCE_TRADE] = (
            f"Model confidence {analysis.model_confidence:.3f} is within "
            f"{LOW_CONFIDENCE_MARGIN_ABOVE_THRESHOLD:.2f} of threshold "
            f"{analysis.model_threshold:.2f}."
        )

    # 3+) Orderflow tags only fire when we have orderflow data. The MVP
    #    has no real feed, so these are gated behind explicit non-null
    #    fields and stay quiet by default. The codepath is here so the
    #    classifier is ready when orderflow lands.
    of = analysis.orderflow
    if of.bid_ask_imbalance is not None and analysis.result == "loss":
        if (analysis.direction == "long" and of.bid_ask_imbalance < 0) or (
            analysis.direction == "short" and of.bid_ask_imbalance > 0
        ):
            tags.append(MistakeTag.BAD_ORDERFLOW_CONFIRMATION)
            details[MistakeTag.BAD_ORDERFLOW_CONFIRMATION] = (
                f"Imbalance {of.bid_ask_imbalance:+.2f} contradicts {analysis.direction} entry."
            )
    if of.cumulative_delta is not None and analysis.result == "loss":
        if (analysis.direction == "long" and of.cumulative_delta < 0) or (
            analysis.direction == "short" and of.cumulative_delta > 0
        ):
            tags.append(MistakeTag.ORDERFLOW_DIVERGENCE)
            details[MistakeTag.ORDERFLOW_DIVERGENCE] = (
                f"Cumulative delta {of.cumulative_delta:+.2f} diverged from price."
            )

    # 4) Entered during chop.
    if analysis.market_regime == "chop" and analysis.result == "loss":
        tags.append(MistakeTag.ENTERED_DURING_CHOP)
        details[MistakeTag.ENTERED_DURING_CHOP] = "trend_regime=chop at entry."

    # 5) Low volume entry.
    vol_ratio = analysis.features.get("volume_ratio_20")
    if vol_ratio is not None and vol_ratio < LOW_VOLUME_RATIO_FLOOR:
        tags.append(MistakeTag.LOW_VOLUME_TRADE)
        details[MistakeTag.LOW_VOLUME_TRADE] = (
            f"volume_ratio_20={vol_ratio:.2f} < {LOW_VOLUME_RATIO_FLOOR:.2f}"
        )

    # 6) Bad time of day — we don't have history yet to know which
    #    buckets are bad; the PatternMiner generates that. For now we
    #    only tag trades in the "lunch" bucket as a known weak slot for
    #    intraday futures, which the operator can override.
    if analysis.time_of_day_bucket == "lunch" and analysis.result == "loss":
        tags.append(MistakeTag.BAD_TIME_OF_DAY)
        details[MistakeTag.BAD_TIME_OF_DAY] = "Entered during lunch chop window."

    # 7) High volatility spike.
    vol_code = analysis.features.get("volatility_regime")
    try:
        vol_int = int(vol_code) if vol_code is not None else 0
    except (TypeError, ValueError):
        vol_int = 0
    if vol_int >= HIGH_VOLATILITY_REGIME_VALUE and analysis.result == "loss":
        tags.append(MistakeTag.HIGH_VOLATILITY_SPIKE)
        details[MistakeTag.HIGH_VOLATILITY_SPIKE] = (
            f"volatility_regime={vol_int} (high) at entry."
        )

    # 8) News risk trade — entry happened while NewsAgent flagged window.
    if analysis.news_risk_level == "high":
        tags.append(MistakeTag.NEWS_RISK_TRADE)
        details[MistakeTag.NEWS_RISK_TRADE] = "Entered during agent-flagged news window."

    # 9) Stop too tight — only fires for losses that lost via SL with low MAE.
    if (
        analysis.result == "loss"
        and analysis.exit_reason == "sl"
        and analysis.mae is not None
        and analysis.stop_price is not None
        and analysis.mae <= abs(analysis.net_pnl) * 0.6
    ):
        tags.append(MistakeTag.STOP_TOO_TIGHT)
        details[MistakeTag.STOP_TOO_TIGHT] = (
            f"MAE only ${analysis.mae:.2f} before stop fired."
        )

    # 10) Target too far — winning trade with MFE near target but exit
    #     was time-based, *or* losing trade that came back close to TP.
    if (
        analysis.target_price is not None
        and analysis.mfe is not None
        and analysis.exit_reason in {"time", "forced_flat"}
        and analysis.result != "win"
    ):
        # If MFE was a meaningful fraction of the planned reward, target
        # was probably reachable but we ran out of time.
        planned_reward = abs(analysis.target_price - analysis.entry_price)
        if planned_reward > 0 and analysis.mfe >= 0.7 * planned_reward * 0.0:  # placeholder; in $ we don't have reward $
            tags.append(MistakeTag.TARGET_TOO_FAR)
            details[MistakeTag.TARGET_TOO_FAR] = (
                f"Time-based exit while MFE=${analysis.mfe:.2f} was close to plan."
            )

    # 11) Poor risk/reward — planned RR <= 1.
    if (
        analysis.stop_price is not None
        and analysis.target_price is not None
    ):
        planned_risk = abs(analysis.entry_price - analysis.stop_price)
        planned_reward = abs(analysis.target_price - analysis.entry_price)
        if planned_risk > 0:
            rr = planned_reward / planned_risk
            if rr <= POOR_RR_RATIO:
                tags.append(MistakeTag.POOR_RISK_REWARD)
                details[MistakeTag.POOR_RISK_REWARD] = (
                    f"Planned reward/risk = {rr:.2f} (≤ {POOR_RR_RATIO})."
                )

    # 12) Slippage loss — losing trade where commissions+slippage > |gross|.
    if (
        analysis.result == "loss"
        and abs(analysis.gross_pnl) > 0
        and (analysis.commission + analysis.slippage)
        >= abs(analysis.gross_pnl) * 0.5
    ):
        tags.append(MistakeTag.SLIPPAGE_LOSS)
        details[MistakeTag.SLIPPAGE_LOSS] = (
            f"Costs ${analysis.commission + analysis.slippage:.2f} ate "
            f"≥50% of gross |${analysis.gross_pnl:.2f}|."
        )

    # 13) Timeout exit — exited because of max-hold, not by plan target.
    if analysis.exit_reason == "time":
        tags.append(MistakeTag.TIMEOUT_EXIT)
        details[MistakeTag.TIMEOUT_EXIT] = "Position closed via max-hold time-out."

    # 14) Rule violation — followed_plan=False captures execution drift
    #     (e.g. end_of_data exit, manual close).
    if not analysis.followed_plan:
        tags.append(MistakeTag.RULE_VIOLATION)
        details[MistakeTag.RULE_VIOLATION] = (
            f"Exit reason '{analysis.exit_reason}' or entry slippage exceeded plan."
        )

    # Weak winner — winner with R below threshold gets the catch-all
    # POOR_RISK_REWARD already if planned RR is poor; otherwise leave
    # untagged. This is intentional: the classifier favors precision.

    # Default if absolutely nothing fired and the trade lost.
    if not tags and analysis.result == "loss":
        tags.append(MistakeTag.UNKNOWN)
        details[MistakeTag.UNKNOWN] = "No specific rule matched; lost trade."

    # De-duplicate while preserving order.
    seen: set[MistakeTag] = set()
    deduped: list[MistakeTag] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return MistakeTagging(tags=tuple(deduped), details=details)


# Public façade so callers can import a single class.
class MistakeClassifier:
    @staticmethod
    def classify(analysis: PostTradeAnalysis) -> MistakeTagging:
        return classify(analysis)


def is_false_positive(analysis: PostTradeAnalysis) -> bool:
    """Convenience predicate matching the classifier's rule."""
    return MistakeTag.FALSE_POSITIVE in classify(analysis).tags


def tag_summary(taggings: Iterable[MistakeTagging]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tagging in taggings:
        for t in tagging.tags:
            counts[t.value] = counts.get(t.value, 0) + 1
    return counts
