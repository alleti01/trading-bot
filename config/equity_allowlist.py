"""Vetted liquid US equity/ETF allowlist for dynamic symbol discovery.

Research agents may *propose* a watchlist, but a proposed symbol is only
ever scanned if it appears here. This is a hard safety boundary: the LLM
cannot invent or trade an arbitrary ticker — it can only prioritize names
from this vetted, liquid set. The deterministic pipeline (strategy →
model → risk → broker validation) still decides every trade.

All entries are large-cap US equities or major ETFs with deep liquidity
and tight spreads, suitable for paper testing the VWAP/EMA strategy.
"""

from __future__ import annotations

# Major index / sector ETFs.
_ETFS: frozenset[str] = frozenset(
    {
        "SPY", "QQQ", "IWM", "DIA",  # broad index
        "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "XLRE",  # sectors
        "SMH", "SOXX",  # semis
        "GLD", "SLV",  # metals ETFs (equity-settled)
        "TLT",  # bonds
    }
)

# Large-cap, high-liquidity single names.
_LARGE_CAPS: frozenset[str] = frozenset(
    {
        "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "GOOGL", "GOOG",
        "META", "NFLX", "AVGO", "INTC", "MU", "QCOM", "ADBE", "CRM",
        "ORCL", "CSCO", "TXN", "AMAT",
        "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA",
        "XOM", "CVX", "COP",
        "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV",
        "WMT", "COST", "HD", "PG", "KO", "PEP", "MCD", "NKE", "DIS",
        "BA", "CAT", "GE", "UBER", "PYPL", "SHOP", "PLTR", "COIN", "SQ",
    }
)

LIQUID_EQUITY_ALLOWLIST: frozenset[str] = _ETFS | _LARGE_CAPS


def is_allowed_equity(symbol: str) -> bool:
    return symbol.upper() in LIQUID_EQUITY_ALLOWLIST


def filter_to_allowlist(symbols: list[str]) -> list[str]:
    """Return the subset of ``symbols`` that are on the liquid allowlist,
    de-duplicated and uppercased, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for s in symbols:
        u = s.upper()
        if u in LIQUID_EQUITY_ALLOWLIST and u not in seen:
            seen.add(u)
            out.append(u)
    return out
