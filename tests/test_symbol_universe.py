"""SymbolUniverse + per-symbol CSV resolver + webhook filter.

Coverage:

1. ``ENABLED_SYMBOLS`` parses comma-separated env values, uppercases,
   strips whitespace.
2. Duplicate symbols fail with a clear error.
3. Unknown symbols fail with a clear error listing supported symbols.
4. Mixing futures + crypto in one universe is refused.
5. ``PRIMARY_SYMBOL`` defaults to ``INSTRUMENT`` and must be in the
   universe; otherwise refused.
6. ``resolve_symbol_csv_paths`` finds existing CSVs and surfaces missing
   symbols separately so the orchestrator can disable just those.
7. ``build_per_symbol_feeds`` isolates per-symbol load failures.
8. TradingView webhook validator rejects disabled symbols, accepts
   enabled ones, and returns a normalized :class:`WebhookSignal`.
9. Daily report payload now carries ``by_symbol`` + ``best_symbol`` /
   ``worst_symbol`` (smoke check that the rendered Markdown table lands).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from config.instruments import (
    SymbolUniverse,
    SymbolUniverseError,
    is_supported_symbol,
    supported_symbols,
)
from data.market_data_service import (
    PerSymbolFeedPlan,
    build_per_symbol_feeds,
    resolve_symbol_csv_paths,
)
from webhook.tradingview import (
    InvalidWebhookSignal,
    validate_webhook_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _settings(tmp_path: Path, **overrides):
    from config.settings import reload_settings
    from storage.db import init_db

    defaults = {
        "MODE": "PAPER",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "REPORTS_DIR": str(tmp_path / "reports"),
        "MODELS_DIR": str(tmp_path / "models"),
        "DATABASE_URL": "sqlite:///:memory:",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    s = reload_settings()
    init_db()
    return s


def _write_per_symbol_csvs(
    base_dir: Path, symbols: list[str], *, n_bars: int = 200, tz: str = "UTC"
) -> None:
    """Create ``base_dir/<SYM>/1m.csv`` for each symbol using synthetic OHLCV."""
    from tests.fixtures.synthetic import synthetic_ohlcv

    for sym in symbols:
        sub = base_dir / sym.upper()
        sub.mkdir(parents=True, exist_ok=True)
        df = synthetic_ohlcv(n_bars=n_bars, tz=tz, seed=hash(sym) & 0xFFFF)
        out = df.reset_index()
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        out = out[["timestamp", "open", "high", "low", "close", "volume"]]
        out.to_csv(sub / "1m.csv", index=False)


# ---------------------------------------------------------------------------
# 1. Parsing
# ---------------------------------------------------------------------------
def test_universe_parses_comma_separated_string() -> None:
    u = SymbolUniverse.from_list("MES, MNQ ,  MGC")
    assert u.as_list() == ["MES", "MNQ", "MGC"]
    assert u.primary == "MES"


def test_universe_uppercases_and_strips() -> None:
    u = SymbolUniverse.from_list(["mes", " MNq ", "mgc"])
    assert u.as_list() == ["MES", "MNQ", "MGC"]


def test_universe_from_settings_defaults_to_instrument(tmp_path: Path) -> None:
    s = _settings(tmp_path, INSTRUMENT="MES")
    u = SymbolUniverse.from_settings(s)
    assert u.as_list() == ["MES"]
    assert u.primary == "MES"


def test_universe_reads_env_enabled_symbols(tmp_path: Path) -> None:
    s = _settings(tmp_path, ENABLED_SYMBOLS="MES,MNQ,MGC")
    u = SymbolUniverse.from_settings(s)
    assert u.as_list() == ["MES", "MNQ", "MGC"]


# ---------------------------------------------------------------------------
# 2-5. Validations
# ---------------------------------------------------------------------------
def test_duplicate_symbols_rejected() -> None:
    with pytest.raises(SymbolUniverseError) as exc:
        SymbolUniverse.from_list("MES,MES,MNQ")
    assert "duplicate" in str(exc.value).lower()


def test_unknown_symbol_rejected_with_supported_listing() -> None:
    with pytest.raises(SymbolUniverseError) as exc:
        SymbolUniverse.from_list("MES,FAKEFOO")
    msg = str(exc.value)
    assert "FAKEFOO" in msg
    # The error must list real supported symbols so the operator can fix.
    assert "MES" in msg


def test_mixing_market_types_rejected() -> None:
    with pytest.raises(SymbolUniverseError) as exc:
        SymbolUniverse.from_list("MES,BTC", market_type="futures")
    assert "BTC" in str(exc.value)


def test_primary_symbol_must_be_in_universe() -> None:
    with pytest.raises(SymbolUniverseError):
        SymbolUniverse.from_list("MES,MNQ", primary="MGC")


def test_empty_universe_rejected() -> None:
    with pytest.raises(SymbolUniverseError):
        SymbolUniverse.from_list("")


def test_supported_symbols_includes_ops_set() -> None:
    """The MVP's required-by-spec set is registered."""
    for sym in ("MES", "MNQ", "MGC", "MCL", "MYM", "M2K"):
        assert is_supported_symbol(sym), f"{sym} missing from registry"
    assert "BTC" in supported_symbols()


# ---------------------------------------------------------------------------
# 6-7. Per-symbol CSV resolver
# ---------------------------------------------------------------------------
def test_resolve_symbol_csv_paths_finds_present_and_missing(tmp_path: Path) -> None:
    base = tmp_path / "data" / "historical"
    _write_per_symbol_csvs(base, ["MES", "MNQ"])
    found, missing = resolve_symbol_csv_paths(
        ["MES", "MNQ", "MGC"], base_dir=base, timeframe="1m"
    )
    assert sorted(found) == ["MES", "MNQ"]
    assert missing == ["MGC"]
    for sym, path in found.items():
        assert path.is_file()
        assert sym in str(path)


def test_build_per_symbol_feeds_isolates_failures(tmp_path: Path) -> None:
    """If one CSV is corrupt the others must still load."""
    base = tmp_path / "data" / "historical"
    _write_per_symbol_csvs(base, ["MES", "MNQ"])
    # Corrupt MNQ's file: write nonsense.
    (base / "MNQ" / "1m.csv").write_text("not,a,valid,csv\n")

    plan = build_per_symbol_feeds(
        ["MES", "MNQ", "MGC"], base_dir=base, timeframe="1m", tz="UTC"
    )
    assert isinstance(plan, PerSymbolFeedPlan)
    assert "MES" in plan.feeds
    assert "MNQ" in plan.failed
    assert plan.missing == ["MGC"]


# ---------------------------------------------------------------------------
# 8. Webhook validator
# ---------------------------------------------------------------------------
def _universe(symbols: list[str]) -> SymbolUniverse:
    return SymbolUniverse.from_list(symbols, market_type="futures")


def test_webhook_accepts_enabled_symbol() -> None:
    universe = _universe(["MES", "MNQ"])
    sig = validate_webhook_signal(
        {
            "symbol": "MES",
            "direction": "long",
            "price": 4500.25,
            "stop": 4495.0,
            "target": 4510.0,
            "strategy": "external_breakout",
            "ts": "2026-05-19T14:00:00Z",
        },
        universe=universe,
    )
    assert sig.symbol == "MES"
    assert sig.direction == "long"
    assert sig.price == pytest.approx(4500.25)
    assert sig.stop_price == pytest.approx(4495.0)
    assert sig.target_price == pytest.approx(4510.0)
    assert sig.strategy == "external_breakout"
    assert sig.ts.tzinfo is not None


def test_webhook_rejects_disabled_symbol() -> None:
    universe = _universe(["MES"])
    with pytest.raises(InvalidWebhookSignal) as exc:
        validate_webhook_signal(
            {
                "symbol": "MNQ",
                "direction": "long",
                "price": 18_000.0,
            },
            universe=universe,
        )
    assert "MNQ" in str(exc.value)
    assert "ENABLED_SYMBOLS" in str(exc.value)


def test_webhook_rejects_unknown_direction() -> None:
    universe = _universe(["MES"])
    with pytest.raises(InvalidWebhookSignal):
        validate_webhook_signal(
            {"symbol": "MES", "direction": "diagonal", "price": 4500.0},
            universe=universe,
        )


def test_webhook_secret_enforced_when_configured() -> None:
    universe = _universe(["MES"])
    with pytest.raises(InvalidWebhookSignal):
        validate_webhook_signal(
            {"symbol": "MES", "direction": "long", "price": 4500.0},
            universe=universe,
            expected_secret="hunter2",
        )
    sig = validate_webhook_signal(
        {
            "symbol": "MES",
            "direction": "long",
            "price": 4500.0,
            "secret": "hunter2",
        },
        universe=universe,
        expected_secret="hunter2",
    )
    assert sig.symbol == "MES"


def test_webhook_rejects_empty_payload() -> None:
    universe = _universe(["MES"])
    with pytest.raises(InvalidWebhookSignal):
        validate_webhook_signal({}, universe=universe)
    with pytest.raises(InvalidWebhookSignal):
        validate_webhook_signal(None, universe=universe)  # type: ignore[arg-type]


def test_webhook_buy_sell_aliases_normalized() -> None:
    universe = _universe(["MES"])
    long_sig = validate_webhook_signal(
        {"symbol": "MES", "direction": "buy", "price": 4500.0},
        universe=universe,
    )
    short_sig = validate_webhook_signal(
        {"symbol": "MES", "direction": "sell", "price": 4500.0},
        universe=universe,
    )
    assert long_sig.direction == "long"
    assert short_sig.direction == "short"


# ---------------------------------------------------------------------------
# 9. Daily report carries per-symbol section
# ---------------------------------------------------------------------------
def _seed_closed_trade(*, instrument: str, pnl: float, base: datetime) -> None:
    from storage.db import session_scope
    from storage.tables import ClosedTrade

    with session_scope() as session:
        session.add(
            ClosedTrade(
                paper_trade_id=None,
                setup_id=None,
                instrument=instrument,
                direction="long",
                quantity=1.0,
                entry_ts=base,
                entry_price=4500.0,
                exit_ts=base + timedelta(minutes=4),
                exit_price=4500.0 + (4 if pnl > 0 else -4),
                exit_reason="tp" if pnl > 0 else "sl",
                pnl=pnl,
                commission=0.5,
                slippage=0.0,
            )
        )


def test_daily_report_payload_has_per_symbol_breakdown(tmp_path: Path) -> None:
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo

    s = _settings(
        tmp_path,
        ENABLED_SYMBOLS="MES,MNQ,MGC",
    )
    today = _dt.now(ZoneInfo(s.TIMEZONE)).date()
    base = _dt(
        today.year, today.month, today.day, 14, 0,
        tzinfo=ZoneInfo(s.TIMEZONE),
    ).astimezone(timezone.utc)
    _seed_closed_trade(instrument="MES", pnl=20.0, base=base)
    _seed_closed_trade(instrument="MES", pnl=-10.0, base=base + timedelta(minutes=5))
    _seed_closed_trade(instrument="MNQ", pnl=15.0, base=base + timedelta(minutes=10))
    _seed_closed_trade(instrument="MGC", pnl=-25.0, base=base + timedelta(minutes=15))

    from reports.daily_report import (
        build_daily_report_payload,
        render_daily_markdown,
    )

    payload = build_daily_report_payload(s, now=base)
    assert payload["enabled_symbols"] == ["MES", "MNQ", "MGC"]
    assert payload["by_symbol"]
    syms = {row["symbol"] for row in payload["by_symbol"]}
    assert syms == {"MES", "MNQ", "MGC"}

    by_symbol = {row["symbol"]: row for row in payload["by_symbol"]}
    assert by_symbol["MES"]["trades"] == 2
    assert by_symbol["MES"]["wins"] == 1
    assert by_symbol["MES"]["losses"] == 1
    assert by_symbol["MNQ"]["trades"] == 1
    assert by_symbol["MGC"]["trades"] == 1
    # Best/worst by net PnL (MNQ gains the most relatively, MGC loses).
    assert payload["best_symbol"] == "MNQ"
    assert payload["worst_symbol"] == "MGC"

    md = render_daily_markdown(payload)
    assert "Performance by symbol" in md
    assert "MES" in md and "MNQ" in md and "MGC" in md
    assert "Best symbol" in md and "Worst symbol" in md
