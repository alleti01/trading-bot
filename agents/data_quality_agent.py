"""DataQualityAgent: deterministic pre-flight check on per-symbol feeds.

This is the only "agent" in the package that does **not** require an
LLM. It exists so the same orchestrator hook the other agents use can
also enforce data-quality rules before paper mode starts on a given
symbol. Operationally it answers the question "is this symbol's data
healthy enough to trust for trading today?"

Checks performed per symbol:

- ``empty_feed``        — no rows.
- ``missing_candles``   — >= ``max_gap_bars`` consecutive bar
  intervals missing inside RTH.
- ``duplicate_timestamps`` — repeated timestamps after sort.
- ``bad_ohlcv``         — geometrically impossible candles
  (high < low, NaN/Inf, negative volume, etc.).
- ``stale_feed``        — most recent bar older than
  ``stale_after_seconds``.
- ``symbol_data_gap``   — *aggregate* gap percentage above
  ``max_gap_pct``.

The deterministic checks decide :class:`DataQualityReport`'s
``blocked_symbols`` directly: ``empty_feed``, ``stale_feed``, or any
``severity="high"`` bad-OHLCV / data-gap finding pushes the symbol
onto the blocked list. Operators / orchestrators downstream (paper
loop, scheduler) read ``blocked_symbols`` and refuse to start that
symbol's loop.

Architectural property: this module imports nothing from
``execution/`` or ``risk/``. It is pure pandas + stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Iterable, Mapping, Optional

import math

import pandas as pd

from agents.base_agent import AgentContext, BaseAgent
from agents.schemas import (
    AgentResult,
    DataQualityIssue,
    DataQualityReport,
    RiskLevel,
)
from app.logging_config import get_logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataQualityConfig:
    """Operator-tunable thresholds.

    All times in seconds; ``max_gap_bars`` is in *bars* (so the same
    config works for 1m / 5m / 1h feeds).
    """

    bar_seconds: int = 60                # 1m default
    stale_after_seconds: int = 600       # >10m old -> stale
    max_gap_bars: int = 5                # >5 missing bars in a row -> issue
    max_gap_pct: float = 0.05            # >5% missing bars -> issue
    min_bars: int = 10                   # below this -> ``empty_feed``


# ---------------------------------------------------------------------------
# Per-symbol scan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _ScanOutcome:
    issues: list[DataQualityIssue]
    blocking: bool


def _is_finite_float(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _scan_dataframe(
    symbol: str,
    df: pd.DataFrame,
    *,
    config: DataQualityConfig,
    now_utc: datetime,
) -> _ScanOutcome:
    issues: list[DataQualityIssue] = []
    blocking = False
    sym = symbol.upper()

    # Empty feed: short-circuit. We can't check anything else.
    if df is None or len(df) < config.min_bars:
        issues.append(
            DataQualityIssue(
                symbol=sym,
                kind="empty_feed",
                severity="high",
                detail=(
                    f"Feed for {sym} has only "
                    f"{0 if df is None else len(df)} rows; below "
                    f"min_bars={config.min_bars}."
                ),
                sample_timestamps=[],
            )
        )
        return _ScanOutcome(issues=issues, blocking=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append(
            DataQualityIssue(
                symbol=sym,
                kind="other",
                severity="medium",
                detail=(
                    "Expected a DatetimeIndex; got "
                    f"{type(df.index).__name__}. Skipping further checks."
                ),
                sample_timestamps=[],
            )
        )
        return _ScanOutcome(issues=issues, blocking=False)

    # Duplicates (post-sort).
    sorted_index = df.index.sort_values()
    dup_mask = sorted_index.duplicated(keep="first")
    if dup_mask.any():
        sample = [str(t) for t in sorted_index[dup_mask][:5]]
        issues.append(
            DataQualityIssue(
                symbol=sym,
                kind="duplicate_timestamps",
                severity="medium",
                detail=(
                    f"{int(dup_mask.sum())} duplicate timestamps detected."
                ),
                sample_timestamps=sample,
            )
        )

    # Bad OHLCV: NaN / Inf / negative volume / impossible candles.
    needed = ("open", "high", "low", "close", "volume")
    if all(col in df.columns for col in needed):
        bad_mask = (
            df[list(needed)].isna().any(axis=1)
            | (df["high"] < df["low"])
            | (df["high"] < df[["open", "close"]].max(axis=1))
            | (df["low"] > df[["open", "close"]].min(axis=1))
            | (df["volume"] < 0)
            | (~df["close"].apply(_is_finite_float))
        )
        if bad_mask.any():
            sample = [str(t) for t in df.index[bad_mask][:5]]
            severity: RiskLevel = (
                "high" if int(bad_mask.sum()) > 0 else "medium"
            )
            issues.append(
                DataQualityIssue(
                    symbol=sym,
                    kind="bad_ohlcv",
                    severity=severity,
                    detail=(
                        f"{int(bad_mask.sum())} OHLCV rows are NaN / "
                        "Inf / geometrically impossible / negative-volume."
                    ),
                    sample_timestamps=sample,
                )
            )
            blocking = True
    else:
        issues.append(
            DataQualityIssue(
                symbol=sym,
                kind="other",
                severity="medium",
                detail=(
                    "Required OHLCV columns missing; cannot validate "
                    f"candle geometry. Got columns={list(df.columns)}."
                ),
                sample_timestamps=[],
            )
        )

    # Stale feed: most recent bar age vs ``now_utc``.
    most_recent = df.index.max()
    if isinstance(most_recent, pd.Timestamp):
        most_recent_dt = most_recent.to_pydatetime()
        if most_recent_dt.tzinfo is None:
            most_recent_dt = most_recent_dt.replace(tzinfo=timezone.utc)
        else:
            most_recent_dt = most_recent_dt.astimezone(timezone.utc)
        age = (now_utc - most_recent_dt).total_seconds()
        if age > config.stale_after_seconds:
            issues.append(
                DataQualityIssue(
                    symbol=sym,
                    kind="stale_feed",
                    severity="high",
                    detail=(
                        f"Most recent bar at {most_recent_dt.isoformat()} "
                        f"is {int(age)}s old (threshold "
                        f"{config.stale_after_seconds}s)."
                    ),
                    sample_timestamps=[str(most_recent_dt)],
                )
            )
            blocking = True

    # Gap detection: how many bars are missing relative to the
    # expected ``bar_seconds`` cadence between first and last seen bar.
    if config.bar_seconds > 0 and len(df) >= 2:
        diffs = df.index.to_series().sort_values().diff().dropna()
        bar_td = pd.Timedelta(seconds=config.bar_seconds)
        big_gaps = diffs[diffs > bar_td * config.max_gap_bars]
        if not big_gaps.empty:
            sample = [str(ts) for ts in big_gaps.index[:5]]
            issues.append(
                DataQualityIssue(
                    symbol=sym,
                    kind="missing_candles",
                    severity="medium",
                    detail=(
                        f"{len(big_gaps)} gap(s) larger than "
                        f"{config.max_gap_bars} bars detected."
                    ),
                    sample_timestamps=sample,
                )
            )

        total_span = (df.index.max() - df.index.min())
        try:
            expected_bars = max(1, int(total_span / bar_td) + 1)
        except (TypeError, ValueError):
            expected_bars = len(df)
        if expected_bars > 0:
            missing = max(0, expected_bars - len(df))
            gap_pct = missing / expected_bars
            if gap_pct > config.max_gap_pct:
                severity_g: RiskLevel = (
                    "high" if gap_pct >= 2 * config.max_gap_pct else "medium"
                )
                issues.append(
                    DataQualityIssue(
                        symbol=sym,
                        kind="symbol_data_gap",
                        severity=severity_g,
                        detail=(
                            f"{missing} of {expected_bars} bars missing "
                            f"({gap_pct:.2%}) — exceeds "
                            f"{config.max_gap_pct:.2%}."
                        ),
                        sample_timestamps=[],
                    )
                )
                if severity_g == "high":
                    blocking = True

    return _ScanOutcome(issues=issues, blocking=blocking)


# ---------------------------------------------------------------------------
# Public scan helper (used by tests and the orchestrator)
# ---------------------------------------------------------------------------
def scan_data_quality(
    feeds_by_symbol: Mapping[str, pd.DataFrame],
    *,
    config: Optional[DataQualityConfig] = None,
    now: Optional[datetime] = None,
    session_date: Optional[str] = None,
) -> DataQualityReport:
    """Run the deterministic data-quality pipeline.

    ``feeds_by_symbol`` maps a symbol to its OHLCV DataFrame. Symbols
    with severe issues (empty / stale / corrupt / large data gaps)
    end up in :attr:`DataQualityReport.blocked_symbols` so the
    orchestrator can refuse to start their paper loops.
    """
    cfg = config or DataQualityConfig()
    now_utc = (now or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
    sd = session_date or now_utc.date().isoformat()

    issues: list[DataQualityIssue] = []
    blocked: list[str] = []
    checked: list[str] = []
    for symbol, df in feeds_by_symbol.items():
        sym = str(symbol).upper()
        checked.append(sym)
        outcome = _scan_dataframe(sym, df, config=cfg, now_utc=now_utc)
        issues.extend(outcome.issues)
        if outcome.blocking:
            blocked.append(sym)

    if not feeds_by_symbol:
        summary = "No feeds supplied; nothing checked."
    elif blocked:
        summary = (
            f"{len(blocked)} of {len(checked)} symbols blocked by "
            "data-quality checks: " + ", ".join(sorted(set(blocked))) + "."
        )
    elif issues:
        summary = (
            f"{len(issues)} non-blocking issue(s) across "
            f"{len(checked)} symbol(s); paper mode may proceed."
        )
    else:
        summary = (
            f"All {len(checked)} symbol(s) clean; paper mode may proceed."
        )

    return DataQualityReport(
        session_date=sd,
        checked_symbols=sorted(set(checked)),
        issues=issues,
        blocked_symbols=sorted(set(blocked)),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Agent class — wraps ``scan_data_quality`` so the orchestrator can
# treat it the same way as every other agent.
# ---------------------------------------------------------------------------
class DataQualityAgent(BaseAgent):
    """No-LLM data-quality agent.

    Works in two modes:

    1. **Direct call**:
       :meth:`run_with_feeds(feeds_by_symbol)` — pass DataFrames
       in directly. The orchestrator pre-paper-loop hook uses this.
    2. **Standard ``run(context)``**: reads
       ``context.data_quality`` (a pre-built dict matching the
       :class:`DataQualityReport` schema) and validates it. This is
       useful when the scan has already been performed elsewhere
       and only the orchestrator's "persist + notify" pipeline is
       needed.
    """

    name: ClassVar[str] = "data_quality"
    schema_class = DataQualityReport
    system_prompt: ClassVar[str] = (
        "Deterministic agent — no LLM call performed."
    )

    def __init__(self, llm: Any = None) -> None:
        # ``llm`` accepted only so the orchestrator's existing
        # ``agent_cls(client)`` call doesn't have to special-case
        # this agent. We never use it.
        self.llm = None
        self.config = DataQualityConfig()
        self.log = get_logger(f"agents.{self.name}")

    # ---- BaseAgent contract ------------------------------------
    def build_user_prompt(self, context: AgentContext) -> str:  # pragma: no cover - unused
        return ""

    def run(self, context: AgentContext) -> AgentResult:  # type: ignore[override]
        provided = context.data_quality
        if provided is None:
            # No precomputed report and no feeds available via
            # ``AgentContext`` — emit an empty advisory result and
            # let the orchestrator decide whether to call
            # ``run_with_feeds`` instead.
            empty = DataQualityReport(
                session_date=context.session_date,
                checked_symbols=[],
                issues=[],
                blocked_symbols=[],
                summary=(
                    "DataQualityAgent invoked without feeds; "
                    "no symbols checked."
                ),
            )
            return AgentResult(
                agent_name=self.name,
                schema_valid=True,
                payload=empty.model_dump(),
                raw_text=None,
                error=None,
            )
        try:
            report = DataQualityReport.model_validate(provided)
        except Exception as e:  # noqa: BLE001
            self.log.warning(
                "data_quality.invalid_input", error=str(e)[:300]
            )
            return AgentResult(
                agent_name=self.name,
                schema_valid=False,
                payload=None,
                raw_text=None,
                error=f"invalid_input: {e}",
            )
        return AgentResult(
            agent_name=self.name,
            schema_valid=True,
            payload=report.model_dump(),
            raw_text=None,
            error=None,
        )

    # ---- Direct-call path used by the paper-mode pre-flight hook
    def run_with_feeds(
        self,
        feeds_by_symbol: Mapping[str, pd.DataFrame],
        *,
        now: Optional[datetime] = None,
        session_date: Optional[str] = None,
        config: Optional[DataQualityConfig] = None,
    ) -> AgentResult:
        try:
            report = scan_data_quality(
                feeds_by_symbol,
                config=config or self.config,
                now=now,
                session_date=session_date,
            )
        except Exception as e:  # noqa: BLE001
            self.log.error("data_quality.scan_failed", error=str(e))
            return AgentResult(
                agent_name=self.name,
                schema_valid=False,
                payload=None,
                raw_text=None,
                error=f"scan_failed: {e}",
            )
        return AgentResult(
            agent_name=self.name,
            schema_valid=True,
            payload=report.model_dump(),
            raw_text=None,
            error=None,
        )
