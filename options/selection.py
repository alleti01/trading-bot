"""Contract selection: directional signal → concrete option contract(s)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from app.logging_config import get_logger
from options.chain import BaseChainProvider
from options.contract import OptionContract, OptionType

_log = get_logger("options.selection")


@dataclass(frozen=True)
class SelectionConfig:
    target_dte: int = 30
    min_dte: int = 7
    max_dte: int = 60
    # Target absolute delta for the long leg (0.50 == ATM).
    target_delta: float = 0.50
    # Width (in strikes) for vertical spreads / wings.
    spread_width_strikes: int = 1


def _pick_expiry(
    expiries: list[date], *, config: SelectionConfig, now: datetime
) -> Optional[date]:
    today = now.date()
    candidates = [
        e for e in sorted(expiries) if config.min_dte <= (e - today).days <= config.max_dte
    ]
    if not candidates:
        # Relax to anything in the future if nothing in the window.
        candidates = [e for e in sorted(expiries) if (e - today).days >= config.min_dte]
    if not candidates:
        return None
    return min(candidates, key=lambda e: abs((e - today).days - config.target_dte))


def select_directional_contract(
    chain_provider: BaseChainProvider,
    *,
    underlying: str,
    direction: str,
    config: Optional[SelectionConfig] = None,
    now: Optional[datetime] = None,
) -> Optional[OptionContract]:
    """Long signal → call, short signal → put. Picks the contract whose
    absolute delta is closest to ``target_delta`` at the chosen expiry.
    """
    config = config or SelectionConfig()
    now = now or datetime.now()
    option_type: OptionType = "call" if direction == "long" else "put"

    expiries = chain_provider.list_expirations(underlying)
    expiry = _pick_expiry(expiries, config=config, now=now)
    if expiry is None:
        _log.warning("options.selection.no_expiry", underlying=underlying)
        return None

    chain = chain_provider.get_chain(
        underlying, expiry=expiry, option_type=option_type
    )
    if not chain:
        _log.warning("options.selection.empty_chain", underlying=underlying)
        return None

    def _delta_distance(c: OptionContract) -> float:
        if c.delta is None:
            return 999.0
        return abs(abs(c.delta) - config.target_delta)

    best = min(chain, key=_delta_distance)
    _log.info(
        "options.selection.picked",
        underlying=underlying,
        occ=best.occ_symbol,
        strike=best.strike,
        delta=best.delta,
        dte=best.days_to_expiry(now=now),
    )
    return best


def select_vertical_spread(
    chain_provider: BaseChainProvider,
    *,
    underlying: str,
    direction: str,
    config: Optional[SelectionConfig] = None,
    now: Optional[datetime] = None,
) -> Optional[tuple[OptionContract, OptionContract]]:
    """Returns (long_leg, short_leg) for a debit vertical in the signal
    direction. Bull call spread for long, bear put spread for short."""
    config = config or SelectionConfig()
    now = now or datetime.now()
    long_leg = select_directional_contract(
        chain_provider,
        underlying=underlying,
        direction=direction,
        config=config,
        now=now,
    )
    if long_leg is None:
        return None

    option_type: OptionType = long_leg.option_type
    chain = chain_provider.get_chain(
        underlying, expiry=long_leg.expiry, option_type=option_type
    )
    strikes = sorted({c.strike for c in chain})
    try:
        idx = strikes.index(long_leg.strike)
    except ValueError:
        return None

    # Bull call: short a higher strike. Bear put: short a lower strike.
    if direction == "long":
        target_idx = idx + config.spread_width_strikes
    else:
        target_idx = idx - config.spread_width_strikes
    if not 0 <= target_idx < len(strikes):
        _log.warning("options.selection.no_spread_strike", underlying=underlying)
        return None

    short_strike = strikes[target_idx]
    short_leg = next((c for c in chain if c.strike == short_strike), None)
    if short_leg is None:
        return None
    return long_leg, short_leg


def select_iron_condor(
    chain_provider: BaseChainProvider,
    *,
    underlying: str,
    config: Optional[SelectionConfig] = None,
    now: Optional[datetime] = None,
) -> Optional[dict[str, OptionContract]]:
    """Neutral 4-leg iron condor: short put + long put (lower wing),
    short call + long call (upper wing). Returns a dict of legs or None.
    """
    config = config or SelectionConfig()
    now = now or datetime.now()
    expiries = chain_provider.list_expirations(underlying)
    expiry = _pick_expiry(expiries, config=config, now=now)
    if expiry is None:
        return None

    calls = sorted(
        chain_provider.get_chain(underlying, expiry=expiry, option_type="call"),
        key=lambda c: c.strike,
    )
    puts = sorted(
        chain_provider.get_chain(underlying, expiry=expiry, option_type="put"),
        key=lambda c: c.strike,
    )
    if len(calls) < 2 or len(puts) < 2:
        return None

    # Short strikes near ~0.30 delta, long strikes one wing out.
    def _closest_delta(contracts: list[OptionContract], target: float) -> Optional[OptionContract]:
        scored = [c for c in contracts if c.delta is not None]
        if not scored:
            return None
        return min(scored, key=lambda c: abs(abs(c.delta) - target))

    short_call = _closest_delta(calls, 0.30)
    short_put = _closest_delta(puts, 0.30)
    if short_call is None or short_put is None:
        return None

    call_strikes = [c.strike for c in calls]
    put_strikes = [p.strike for p in puts]
    w = config.spread_width_strikes

    try:
        sc_idx = call_strikes.index(short_call.strike)
        sp_idx = put_strikes.index(short_put.strike)
    except ValueError:
        return None

    long_call = calls[sc_idx + w] if sc_idx + w < len(calls) else None
    long_put = puts[sp_idx - w] if sp_idx - w >= 0 else None
    if long_call is None or long_put is None:
        return None

    return {
        "short_put": short_put,
        "long_put": long_put,
        "short_call": short_call,
        "long_call": long_call,
    }
