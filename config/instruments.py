"""Instrument metadata + multi-symbol universe.

Two layers:

- :class:`InstrumentSpec` — per-symbol contract metadata (tick size,
  point value, session). Looked up via :func:`get_instrument`.
- :class:`SymbolUniverse` — the operator-configured set of *enabled*
  symbols. Built once at boot from settings and consumed by paper mode,
  backtests, reports, and the (optional) TradingView webhook input.

A symbol is considered supported if it has an entry in ``_REGISTRY``.
The MVP ships micro futures + a crypto sample; new symbols are added
by registering an :class:`InstrumentSpec` here and listing them in
``ENABLED_SYMBOLS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    market_type: str          # "futures" | "crypto"
    tick_size: float          # smallest price increment
    tick_value: float         # $ value per tick per contract / unit
    point_value: float        # $ value per 1.00 price move
    session_tz: str           # timezone for trading session
    session_open: str         # HH:MM
    session_close: str        # HH:MM


# Built-in registry. CME micro futures (RTH 09:30–16:00 ET) + a crypto
# stand-in. Tick/point values are the publicly-quoted contract specs.
_REGISTRY: dict[str, InstrumentSpec] = {
    "MES": InstrumentSpec(
        symbol="MES", market_type="futures",
        tick_size=0.25, tick_value=1.25, point_value=5.0,
        session_tz="America/New_York",
        session_open="09:30", session_close="16:00",
    ),
    "MNQ": InstrumentSpec(
        symbol="MNQ", market_type="futures",
        tick_size=0.25, tick_value=0.50, point_value=2.0,
        session_tz="America/New_York",
        session_open="09:30", session_close="16:00",
    ),
    "MGC": InstrumentSpec(
        symbol="MGC", market_type="futures",
        tick_size=0.10, tick_value=1.00, point_value=10.0,
        session_tz="America/New_York",
        session_open="09:30", session_close="16:00",
    ),
    "MCL": InstrumentSpec(
        symbol="MCL", market_type="futures",
        tick_size=0.01, tick_value=1.00, point_value=100.0,
        session_tz="America/New_York",
        session_open="09:30", session_close="16:00",
    ),
    "MYM": InstrumentSpec(
        symbol="MYM", market_type="futures",
        tick_size=1.0, tick_value=0.50, point_value=0.50,
        session_tz="America/New_York",
        session_open="09:30", session_close="16:00",
    ),
    "M2K": InstrumentSpec(
        symbol="M2K", market_type="futures",
        tick_size=0.10, tick_value=0.50, point_value=5.0,
        session_tz="America/New_York",
        session_open="09:30", session_close="16:00",
    ),
    "BTC": InstrumentSpec(
        symbol="BTC", market_type="crypto",
        tick_size=0.01, tick_value=0.01, point_value=1.0,
        session_tz="UTC",
        session_open="00:00", session_close="23:59",
    ),
    "ETH": InstrumentSpec(
        symbol="ETH", market_type="crypto",
        tick_size=0.01, tick_value=0.01, point_value=1.0,
        session_tz="UTC",
        session_open="00:00", session_close="23:59",
    ),
}


def get_instrument(symbol: str) -> InstrumentSpec:
    """Look up an instrument by symbol (case-insensitive)."""
    try:
        return _REGISTRY[symbol.upper()]
    except KeyError as e:
        raise KeyError(
            f"Unknown instrument '{symbol}'. Known: {sorted(_REGISTRY)}"
        ) from e


def is_supported_symbol(symbol: str) -> bool:
    return symbol.upper() in _REGISTRY


def supported_symbols() -> list[str]:
    """The full sorted list of registered symbols."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Multi-symbol universe
# ---------------------------------------------------------------------------
class SymbolUniverseError(ValueError):
    """Raised when ENABLED_SYMBOLS is malformed."""


@dataclass(frozen=True)
class SymbolUniverse:
    """The operator-configured set of symbols the bot is allowed to trade.

    Built once via :meth:`from_settings` (or :meth:`from_list`) at boot
    and threaded through paper mode / backtest / reports. Validation
    rules:

    - Each entry maps to a registered :class:`InstrumentSpec`.
    - Duplicates are rejected.
    - The list is non-empty after parsing (single-symbol bots default
      to ``[INSTRUMENT]`` upstream; an empty universe is a config bug
      we surface loudly rather than silently fall back).
    - All symbols share the same ``market_type`` — mixing
      futures+crypto in one universe complicates session/cost
      handling and is rejected for the MVP.

    The universe is *immutable* — to change it the operator restarts
    with a new env. This guarantees that runtime caps (per-symbol day
    counters, max-active-symbols) cannot be invalidated mid-session by
    a config reload.
    """

    symbols: tuple[str, ...]
    primary: str

    # ---- constructors -------------------------------------------------
    @classmethod
    def from_list(
        cls,
        raw: Iterable[str] | str,
        *,
        primary: Optional[str] = None,
        market_type: Optional[str] = None,
    ) -> "SymbolUniverse":
        symbols = cls._parse(raw)
        cls._validate_unique(symbols)
        cls._validate_known(symbols)
        if market_type is not None:
            cls._validate_market_type(symbols, market_type)
        prim = (primary or symbols[0]).upper()
        if prim not in symbols:
            raise SymbolUniverseError(
                f"PRIMARY_SYMBOL {prim!r} is not in ENABLED_SYMBOLS {list(symbols)}"
            )
        return cls(symbols=tuple(symbols), primary=prim)

    @classmethod
    def from_settings(cls, settings) -> "SymbolUniverse":
        """Build from a pydantic ``Settings`` object.

        Falls back to ``[INSTRUMENT]`` when ``ENABLED_SYMBOLS`` is empty
        so single-symbol bots keep working with no env changes.
        """
        raw = list(getattr(settings, "ENABLED_SYMBOLS", None) or [])
        if not raw:
            raw = [settings.INSTRUMENT]
        primary = (
            getattr(settings, "PRIMARY_SYMBOL", None) or settings.INSTRUMENT
        )
        return cls.from_list(
            raw, primary=primary, market_type=settings.MARKET_TYPE
        )

    # ---- accessors ----------------------------------------------------
    def __iter__(self):
        return iter(self.symbols)

    def __len__(self) -> int:
        return len(self.symbols)

    def __contains__(self, symbol: str) -> bool:
        return symbol.upper() in self.symbols

    def specs(self) -> list[InstrumentSpec]:
        return [get_instrument(s) for s in self.symbols]

    def market_type(self) -> str:
        # All symbols share this (validated at construction).
        return get_instrument(self.symbols[0]).market_type

    def as_list(self) -> list[str]:
        return list(self.symbols)

    # ---- validators ---------------------------------------------------
    @staticmethod
    def _parse(raw: Iterable[str] | str) -> list[str]:
        if isinstance(raw, str):
            parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
        else:
            parts = [str(p).strip().upper() for p in raw if str(p).strip()]
        if not parts:
            raise SymbolUniverseError(
                "ENABLED_SYMBOLS is empty; configure at least one symbol "
                "(e.g. ENABLED_SYMBOLS=MES)."
            )
        return parts

    @staticmethod
    def _validate_unique(symbols: list[str]) -> None:
        seen: set[str] = set()
        dupes: list[str] = []
        for s in symbols:
            if s in seen:
                dupes.append(s)
            seen.add(s)
        if dupes:
            raise SymbolUniverseError(
                f"ENABLED_SYMBOLS contains duplicates: {sorted(set(dupes))}"
            )

    @staticmethod
    def _validate_known(symbols: list[str]) -> None:
        unknown = [s for s in symbols if not is_supported_symbol(s)]
        if unknown:
            raise SymbolUniverseError(
                f"ENABLED_SYMBOLS contains unknown symbols {unknown}. "
                f"Supported: {supported_symbols()}. Add an InstrumentSpec "
                f"to config/instruments.py to enable a new symbol."
            )

    @staticmethod
    def _validate_market_type(symbols: list[str], market_type: str) -> None:
        for s in symbols:
            spec = get_instrument(s)
            if spec.market_type != market_type:
                raise SymbolUniverseError(
                    f"Symbol {s} is {spec.market_type!r} but MARKET_TYPE "
                    f"is {market_type!r}. Mixing market types in one "
                    f"universe is not supported."
                )
