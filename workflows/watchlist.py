"""Build the scan universe for the intraday loop (allowlist-gated).

The universe is always ``ENABLED_SYMBOLS`` first, optionally expanded
with vetted liquid-allowlist names when ``WORKFLOW_DYNAMIC_UNIVERSE`` is
on. Research agents may *rank* allowlist names (advisory), but a symbol
is only ever included if it is on ``LIQUID_EQUITY_ALLOWLIST``. The LLM
can never inject an arbitrary ticker, and it never decides a trade — the
deterministic signal/risk pipeline does.
"""

from __future__ import annotations

from typing import Optional

from app.logging_config import get_logger
from config.equity_allowlist import (
    LIQUID_EQUITY_ALLOWLIST,
    filter_to_allowlist,
    is_allowed_equity,
)
from config.settings import Settings

_log = get_logger("workflows.watchlist")


def build_scan_universe(
    settings: Settings,
    *,
    agent_candidates: Optional[list[str]] = None,
) -> list[str]:
    """Return the ordered, de-duplicated, capped list of symbols to scan.

    Order of preference:
    1. ``ENABLED_SYMBOLS`` (operator-pinned, always included first).
    2. Agent-proposed allowlist names (advisory ranking), if provided.
    3. Remaining allowlist names (deterministic order) to fill the cap.

    Dynamic expansion only happens when ``WORKFLOW_DYNAMIC_UNIVERSE`` is
    true. Everything is filtered against the liquid allowlist except the
    operator's explicit ENABLED_SYMBOLS (which they chose deliberately).
    """
    cap = int(settings.WORKFLOW_MAX_UNIVERSE)
    universe: list[str] = []
    seen: set[str] = set()

    def _add(sym: str) -> None:
        u = sym.upper()
        if u not in seen and len(universe) < cap:
            seen.add(u)
            universe.append(u)

    # 1. Operator-pinned symbols always come first.
    for s in settings.ENABLED_SYMBOLS:
        _add(s)

    if not settings.WORKFLOW_DYNAMIC_UNIVERSE:
        return universe

    # 2. Agent-proposed names — allowlist-gated.
    if agent_candidates:
        rejected = [s for s in agent_candidates if not is_allowed_equity(s)]
        if rejected:
            _log.info("watchlist.agent_rejected_off_allowlist", rejected=rejected)
        for s in filter_to_allowlist(agent_candidates):
            _add(s)

    # 3. Fill remaining capacity from the allowlist (deterministic order).
    for s in sorted(LIQUID_EQUITY_ALLOWLIST):
        if len(universe) >= cap:
            break
        _add(s)

    _log.info(
        "watchlist.built",
        size=len(universe),
        dynamic=settings.WORKFLOW_DYNAMIC_UNIVERSE,
        cap=cap,
    )
    return universe


def propose_agent_watchlist(
    settings: Settings,
    *,
    orchestrator=None,  # noqa: ANN001 — optional AgentOrchestrator
) -> list[str]:
    """Ask the research agent to rank allowlist names (advisory only).

    Returns a list of allowlist tickers, or [] when agents are disabled
    or unavailable. Any symbol the agent returns that is not on the
    allowlist is dropped by ``build_scan_universe``. This function never
    places trades and never changes risk settings.
    """
    if not settings.WORKFLOW_AGENT_WATCHLIST:
        return []
    if orchestrator is None or not settings.ENABLE_LLM_AGENTS:
        return []
    # The strategy-research agent is the natural place to surface
    # candidate names. It is advisory and block-safe; if it fails or is
    # disabled we simply fall back to the deterministic allowlist order.
    proposer = getattr(orchestrator, "propose_watchlist_symbols", None)
    if proposer is None:
        return []
    try:
        candidates = proposer(allowlist=sorted(LIQUID_EQUITY_ALLOWLIST))
    except Exception as e:  # noqa: BLE001
        _log.warning("watchlist.agent_failed", error=str(e))
        return []
    return filter_to_allowlist(list(candidates or []))
