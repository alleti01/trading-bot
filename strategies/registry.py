"""Strategy plugin registry + multi-strategy conflict resolution.

Why this file exists
--------------------
Until now ``app/main.py``, ``backtesting/``, and ``paper/loop.py`` all
imported :class:`VWAPEMAPullback` directly. Adding a second strategy
meant editing every site. This module replaces that with:

- :class:`StrategyRegistry` — name → class lookup, with a small
  decorator (``@register_strategy``) for plug-in registration.
- :func:`get_strategy` / :func:`instantiate` — public lookups used by
  the CLI and the runners.
- :func:`instantiate_enabled` — reads ``settings.ENABLED_STRATEGIES``
  and returns one instance per enabled name.
- :func:`resolve_conflicts` — pure function used by the paper loop to
  prevent simultaneous long+short on the same symbol.

The registry is module-level. Importing this module always populates it
with the in-tree strategies (`vwap_ema_pullback`, `opening_range_breakout`)
so callers don't need any explicit "discover plugins" step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Type

from app.logging_config import get_logger
from strategies.base import Setup, Strategy


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class StrategyRegistry:
    """Name → Strategy class lookup.

    The registry is intentionally simple: it holds *classes*, not
    instances, so the same registry can be safely reused across runs
    with different ``instrument`` / ``params`` arguments.
    """

    def __init__(self) -> None:
        self._classes: dict[str, Type[Strategy]] = {}

    # ---- registration -------------------------------------------------
    def register(self, cls: Type[Strategy]) -> Type[Strategy]:
        """Register a strategy class. Idempotent for the same class.

        Returns the class unchanged so this method can be used as a
        decorator on a class definition.
        """
        name = getattr(cls, "name", None)
        if not name or name == "abstract":
            raise ValueError(
                f"Strategy class {cls.__name__} must define a non-empty "
                f"`name` class variable to be registered."
            )
        existing = self._classes.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Strategy name {name!r} already registered by "
                f"{existing.__name__}; refusing to overwrite with {cls.__name__}."
            )
        self._classes[name] = cls
        return cls

    # ---- lookup -------------------------------------------------------
    def get(self, name: str) -> Type[Strategy]:
        try:
            return self._classes[name]
        except KeyError as e:
            available = ", ".join(sorted(self._classes)) or "<none>"
            raise KeyError(
                f"Unknown strategy {name!r}. Available: {available}."
            ) from e

    def list_names(self) -> list[str]:
        return sorted(self._classes)

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial
        return name in self._classes

    # ---- instantiation -----------------------------------------------
    def instantiate(self, name: str, *, instrument: str) -> Strategy:
        cls = self.get(name)
        return cls(instrument=instrument)

    def instantiate_many(
        self, names: Iterable[str], *, instrument: str
    ) -> list[Strategy]:
        # Preserve caller order; deduplicate while keeping first occurrence.
        seen: set[str] = set()
        out: list[Strategy] = []
        for n in names:
            if n in seen:
                continue
            seen.add(n)
            out.append(self.instantiate(n, instrument=instrument))
        return out


# Module-level singleton. Importing this module is the discovery step.
STRATEGY_REGISTRY = StrategyRegistry()


def register_strategy(cls: Type[Strategy]) -> Type[Strategy]:
    """Decorator alias for ``STRATEGY_REGISTRY.register``."""
    return STRATEGY_REGISTRY.register(cls)


# ---------------------------------------------------------------------------
# Public lookup helpers (the CLI + runners use these, not the singleton)
# ---------------------------------------------------------------------------
def get_strategy_class(name: str) -> Type[Strategy]:
    return STRATEGY_REGISTRY.get(name)


def list_strategies() -> list[str]:
    return STRATEGY_REGISTRY.list_names()


def instantiate(name: str, *, instrument: str) -> Strategy:
    return STRATEGY_REGISTRY.instantiate(name, instrument=instrument)


def instantiate_enabled(
    settings,
    *,
    instrument: Optional[str] = None,
    cli_strategy: Optional[str] = None,
) -> list[Strategy]:
    """Build the list of enabled :class:`Strategy` instances.

    Resolution order:

    1. If ``cli_strategy`` is provided (from ``--strategy``), that
       single name wins — operators can override the env to focus on
       one strategy in TRAIN/BACKTEST without touching ``.env``.
    2. Otherwise read ``settings.ENABLED_STRATEGIES``. Empty / missing
       falls back to ``["vwap_ema_pullback"]`` so the bot never starts
       paper mode with zero strategies by accident.
    """
    instrument = instrument or settings.INSTRUMENT
    if cli_strategy:
        return [instantiate(cli_strategy, instrument=instrument)]
    enabled = list(getattr(settings, "ENABLED_STRATEGIES", None) or [])
    if not enabled:
        enabled = ["vwap_ema_pullback"]
    return STRATEGY_REGISTRY.instantiate_many(enabled, instrument=instrument)


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoredSetup:
    """A setup paired with the model gate's verdict.

    ``confidence`` is the calibrated probability the model assigned.
    ``approved`` mirrors ``Predictor.predict_setup(...).approved`` —
    True iff confidence >= threshold. Either may be ``None`` when the
    paper loop runs without a model gate, in which case the resolver
    treats all setups as "missing confidence".
    """

    setup: Setup
    confidence: Optional[float] = None
    approved: Optional[bool] = None


@dataclass(frozen=True)
class ConflictRecord:
    """One per (instrument, timestamp) pair that produced a conflict.

    Captured for logging + notifications. ``winner_setup_id`` is the
    setup we kept (if any); ``dropped_setup_ids`` are the ones the
    resolver threw away. ``reason`` is one of:

    - ``"highest_confidence"`` — both sides scored, kept the higher
      approved confidence.
    - ``"missing_confidence"`` — at least one side missing scoring,
      dropped both to be safe.
    - ``"none_approved"`` — both sides explicitly model-rejected.
    """

    instrument: str
    timestamp: object  # datetime, but we don't import to keep this file thin
    long_setup_id: Optional[str]
    short_setup_id: Optional[str]
    winner_setup_id: Optional[str]
    dropped_setup_ids: list[str]
    reason: str


@dataclass
class ResolutionResult:
    """Output of :func:`resolve_conflicts`."""

    survivors: list[ScoredSetup] = field(default_factory=list)
    conflicts: list[ConflictRecord] = field(default_factory=list)


def _pick_best(side_setups: list[ScoredSetup]) -> Optional[ScoredSetup]:
    """Pick the highest-confidence approved setup from a same-direction
    bucket. Returns ``None`` when no candidate is approved with a
    numeric confidence."""
    approved = [
        s for s in side_setups
        if s.confidence is not None and (s.approved is True or s.approved is None)
    ]
    if not approved:
        return None
    # Without a predictor (approved is None), confidence is also None,
    # so the filter above already rules them out — but be defensive.
    return max(approved, key=lambda s: float(s.confidence or 0.0))


def resolve_conflicts(scored: list[ScoredSetup]) -> ResolutionResult:
    """Drop opposing-direction setups on the same symbol.

    Rules (in this order):

    1. Group by ``setup.instrument``. Setups for different instruments
       never conflict.
    2. Within an instrument, if every setup shares the same direction,
       keep all of them — multiple strategies all going long is fine.
    3. If a long and a short are present:
       a. If both sides have at least one approved setup with a
          numeric confidence, keep the single highest-confidence
          approved setup; drop everything else for that instrument.
       b. Otherwise drop *both* sides — refuse to take an unscored
          conflicting trade. We log the skip.

    The function never mutates its inputs.
    """
    log = get_logger("strategies.registry")

    by_instrument: dict[str, list[ScoredSetup]] = {}
    for s in scored:
        by_instrument.setdefault(s.setup.instrument, []).append(s)

    survivors: list[ScoredSetup] = []
    conflicts: list[ConflictRecord] = []

    for instrument, bucket in by_instrument.items():
        longs = [s for s in bucket if s.setup.direction == "long"]
        shorts = [s for s in bucket if s.setup.direction == "short"]

        if not longs or not shorts:
            survivors.extend(bucket)
            continue

        # We have both sides. Pick the best of each.
        best_long = _pick_best(longs)
        best_short = _pick_best(shorts)
        long_ids = [s.setup.id for s in longs]
        short_ids = [s.setup.id for s in shorts]
        ts = bucket[0].setup.timestamp

        if best_long is None and best_short is None:
            reason = "none_approved"
            dropped = long_ids + short_ids
            log.warning(
                "strategy.conflict",
                instrument=instrument,
                reason=reason,
                long_setups=long_ids,
                short_setups=short_ids,
                note="No approved + scored setup on either side; skipping.",
            )
            conflicts.append(
                ConflictRecord(
                    instrument=instrument,
                    timestamp=ts,
                    long_setup_id=long_ids[0] if long_ids else None,
                    short_setup_id=short_ids[0] if short_ids else None,
                    winner_setup_id=None,
                    dropped_setup_ids=dropped,
                    reason=reason,
                )
            )
            continue

        if best_long is None or best_short is None:
            # One side has scoring, the other doesn't. Per spec, refuse
            # both rather than guess.
            reason = "missing_confidence"
            dropped = long_ids + short_ids
            log.warning(
                "strategy.conflict",
                instrument=instrument,
                reason=reason,
                long_setups=long_ids,
                short_setups=short_ids,
                note=(
                    "Long+short on same symbol but at least one side has no "
                    "model confidence; refusing both."
                ),
            )
            conflicts.append(
                ConflictRecord(
                    instrument=instrument,
                    timestamp=ts,
                    long_setup_id=long_ids[0] if long_ids else None,
                    short_setup_id=short_ids[0] if short_ids else None,
                    winner_setup_id=None,
                    dropped_setup_ids=dropped,
                    reason=reason,
                )
            )
            continue

        # Both sides scored — pick the higher confidence.
        if float(best_long.confidence or 0.0) >= float(best_short.confidence or 0.0):
            winner = best_long
            loser_side = shorts
        else:
            winner = best_short
            loser_side = longs

        survivors.append(winner)
        winner_id = winner.setup.id
        dropped = [s.setup.id for s in bucket if s.setup.id != winner_id]
        log.info(
            "strategy.conflict",
            instrument=instrument,
            reason="highest_confidence",
            winner_setup_id=winner_id,
            winner_direction=winner.setup.direction,
            winner_confidence=round(float(winner.confidence or 0.0), 4),
            dropped=dropped,
        )
        conflicts.append(
            ConflictRecord(
                instrument=instrument,
                timestamp=ts,
                long_setup_id=long_ids[0] if long_ids else None,
                short_setup_id=short_ids[0] if short_ids else None,
                winner_setup_id=winner_id,
                dropped_setup_ids=dropped,
                reason="highest_confidence",
            )
        )

    return ResolutionResult(survivors=survivors, conflicts=conflicts)


# ---------------------------------------------------------------------------
# Built-in registrations (module-import side effect)
# ---------------------------------------------------------------------------
# Imports are placed at the bottom so the decorator-style registration
# happens *after* the registry singleton is fully constructed.
from strategies.opening_range_breakout import OpeningRangeBreakout  # noqa: E402
from strategies.vwap_ema_pullback import VWAPEMAPullback  # noqa: E402

STRATEGY_REGISTRY.register(VWAPEMAPullback)
STRATEGY_REGISTRY.register(OpeningRangeBreakout)
