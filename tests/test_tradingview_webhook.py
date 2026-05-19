"""End-to-end tests for the TradingView webhook endpoint.

Coverage map (matches the ticket):

1. valid webhook         -> opens a paper trade
2. invalid secret        -> 401, no trade, no DB rows
3. missing field         -> 422, no trade
4. symbol normalization  -> "MNQ1!" -> "MNQ", crypto suffix stripping
5. approved paper trade  -> Order is submitted, ClosedTrade/PaperTrade rows exist
6. risk-blocked signal   -> 200 status="blocked", risk_block row written
7. Discord failure       -> notifier exception is swallowed, endpoint still 200

Tests bypass the model gate (no predictor) so the assertions don't
depend on a synthetic feature snapshot. The "model gate" path is
covered separately via a stub predictor.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from backtesting.fills import make_fills_model
from backtesting.portfolio import Portfolio
from config.instruments import SymbolUniverse, get_instrument
from config.settings import reload_settings
from execution.paper_executor import PaperExecutor
from risk.kill_switch import KillSwitch
from sqlalchemy import select
from storage.db import init_db, reset_engine_for_tests, session_scope
from storage.tables import ClosedTrade as ClosedTradeRow
from storage.tables import PaperTrade as PaperTradeRow
from storage.tables import RiskBlock as RiskBlockRow
from storage.tables import Setup as SetupRow
from webhooks import create_app
from webhooks.tradingview import normalize_symbol


NY = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _settings(**overrides) -> Any:
    """Build a settings instance suitable for webhook tests.

    The trading window is wide and crypto-style so requests-during-tests
    don't get blocked by clock-of-the-week issues. Daily caps are
    permissive — risk blocking is exercised by individual tests by
    overriding specific values.
    """
    defaults = {
        "MODE": "PAPER",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "TRADING_WINDOW_START": "00:00",
        "TRADING_WINDOW_END": "23:55",
        "FORCE_FLAT_TIME": "23:55",
        "MAX_TRADES_PER_DAY": "8",
        "MAX_TRADES_PER_SYMBOL_PER_DAY": "8",
        "MAX_TOTAL_TRADES_PER_DAY": "8",
        "MAX_ACTIVE_SYMBOLS": "4",
        "MAX_DAILY_LOSS": "10000",
        "MAX_DAILY_PROFIT": "10000",
        "MAX_POSITION_SIZE": "1",
        "RISK_PER_TRADE": "1000",
        "MAX_OPEN_POSITIONS": "1",
        "COOLDOWN_AFTER_LOSS_MINUTES": "0",
        "COOLDOWN_AFTER_LARGE_WIN_MINUTES": "0",
        "LARGE_WIN_THRESHOLD": "9999",
        "MAX_HOLD_BARS": "5",
        "SLIPPAGE_TICKS": "0",
        "COMMISSION_PER_CONTRACT": "0",
        "CRYPTO_SLIPPAGE_BPS": "0",
        "CRYPTO_FEE_BPS": "0",
        "ENABLED_SYMBOLS": "MES,MNQ",
        "PRIMARY_SYMBOL": "MES",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    # Conftest sets DATABASE_URL to a tmp file; we deliberately do NOT
    # override it here. SQLite ``:memory:`` databases don't share state
    # across SQLAlchemy connections, so a temp file is the only thing
    # that actually keeps the same DB across test_client requests.
    return reload_settings()


class _RecordingNotifier:
    """In-memory notifier whose ``notify`` records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def notify(self, kind: str, /, **payload: Any) -> None:
        self.calls.append((kind, dict(payload)))

    def kinds(self) -> list[str]:
        return [k for k, _ in self.calls]


class _ExplodingNotifier:
    """Notifier that raises every time — used to verify the endpoint's
    "Discord failure must not 500" guarantee."""

    def __init__(self) -> None:
        self.calls = 0

    def notify(self, kind: str, /, **payload: Any) -> None:
        self.calls += 1
        raise RuntimeError(f"discord exploded on kind={kind}")


def _build_app(
    *,
    settings,
    notifier,
    kill_switch=None,
    predictor=None,
):
    """Wire up a fresh FastAPI app with the in-memory dependencies."""
    # Each test gets a fresh engine pinned to the current Settings object
    # (the conftest tmp DB). Without the reset, a previously-cached engine
    # bound to a stale URL would leak across tests.
    reset_engine_for_tests()
    init_db()
    universe = SymbolUniverse.from_settings(settings)
    primary = settings.PRIMARY_SYMBOL or settings.INSTRUMENT
    spec = get_instrument(primary)
    portfolio = Portfolio(instrument_spec=spec)
    fills = make_fills_model(
        primary,
        slippage_ticks=settings.SLIPPAGE_TICKS,
        commission_per_contract=settings.COMMISSION_PER_CONTRACT,
        crypto_slippage_bps=settings.CRYPTO_SLIPPAGE_BPS,
        crypto_fee_bps=settings.CRYPTO_FEE_BPS,
    )
    executor = PaperExecutor(
        portfolio=portfolio,
        fills_model=fills,
        kill_switch=kill_switch,
    )
    app = create_app(
        settings=settings,
        executor=executor,
        notifier=notifier,
        universe=universe,
        predictor=predictor,
        kill_switch=kill_switch,
        init_database=False,
    )
    return app, executor, portfolio


def _valid_payload(**overrides) -> dict[str, Any]:
    payload = {
        "secret": "",
        "source": "tradingview",
        "symbol": "MES1!",
        "time": "2026-05-19T14:00:00Z",
        "price": 4500.25,
        "action": "long",
        "strategy": "vwap_pullback",
        "timeframe": "1m",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# 4. Symbol normalization (unit tests for the helper)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("MNQ1!", "MNQ"),
        ("MES1!", "MES"),
        ("NQ1!", "NQ"),
        ("ES1!", "ES"),
        ("MGC2!", "MGC"),
        ("MES", "MES"),
        ("mes1!", "MES"),
        ("MNQ!", "MNQ"),
    ],
)
def test_normalize_futures_symbols(raw: str, expected: str) -> None:
    assert normalize_symbol(raw, market_type="futures") == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BTCUSDT", "BTC"),
        ("BTCUSD", "BTC"),
        ("ETHUSDT", "ETH"),
        ("ETHUSDC", "ETH"),
        ("BINANCE:BTCUSDT", "BTC"),
        ("ETH/USDT", "ETH"),
        ("BTC", "BTC"),
    ],
)
def test_normalize_crypto_symbols(raw: str, expected: str) -> None:
    assert normalize_symbol(raw, market_type="crypto") == expected


def test_normalize_empty_symbol_raises() -> None:
    with pytest.raises(ValueError):
        normalize_symbol("", market_type="futures")


# ---------------------------------------------------------------------------
# 1. Valid webhook -> opens a paper trade  (also covers #5)
# ---------------------------------------------------------------------------
def test_valid_webhook_opens_paper_trade(tmp_path) -> None:
    settings = _settings(MARKET_TYPE="crypto", ENABLED_SYMBOLS="BTC", PRIMARY_SYMBOL="BTC", INSTRUMENT="BTC")
    notifier = _RecordingNotifier()
    app, executor, portfolio = _build_app(settings=settings, notifier=notifier)

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/tradingview",
            json={
                "source": "tradingview",
                "symbol": "BTCUSDT",
                "time": "2026-05-19T14:00:00Z",
                "price": "30000.50",
                "action": "long",
                "strategy": "external",
                "timeframe": "1m",
                "stop": 29950.0,
                "target": 30100.0,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "accepted"
    assert body["symbol"] == "BTC"
    assert body["direction"] == "long"
    assert body["detail"]["raw_symbol"] == "BTCUSDT"
    assert body["detail"]["fill_price"] == pytest.approx(30000.50)

    assert portfolio.open_position is not None
    assert portfolio.open_position.direction == "long"

    with session_scope() as session:
        paper_rows = session.execute(select(PaperTradeRow)).scalars().all()
        setup_rows = session.execute(select(SetupRow)).scalars().all()
    assert len(paper_rows) == 1
    assert paper_rows[0].direction == "long"
    assert paper_rows[0].instrument == "BTC"
    assert len(setup_rows) == 1
    assert setup_rows[0].strategy_name.startswith("webhook:")

    kinds = notifier.kinds()
    assert "webhook.received" in kinds
    assert "webhook.approved" in kinds
    assert "webhook.trade_opened" in kinds


def test_short_webhook_with_default_stop_target(tmp_path) -> None:
    settings = _settings(
        MARKET_TYPE="crypto",
        ENABLED_SYMBOLS="BTC",
        PRIMARY_SYMBOL="BTC",
        INSTRUMENT="BTC",
        WEBHOOK_DEFAULT_STOP_TICKS=10,
        WEBHOOK_DEFAULT_TARGET_TICKS=20,
    )
    notifier = _RecordingNotifier()
    app, executor, portfolio = _build_app(settings=settings, notifier=notifier)

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/tradingview",
            json={
                "source": "tradingview",
                "symbol": "BTC",
                "price": 30000.0,
                "action": "short",
                "strategy": "external",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "accepted"
    assert body["direction"] == "short"
    # Defaults: stop above entry, target below entry for short
    assert body["detail"]["stop_price"] > body["detail"]["fill_price"]
    assert body["detail"]["target_price"] < body["detail"]["fill_price"]


# ---------------------------------------------------------------------------
# 2. Invalid secret -> 401, no trade
# ---------------------------------------------------------------------------
def test_invalid_secret_returns_401_and_no_trade() -> None:
    settings = _settings(
        MARKET_TYPE="crypto",
        ENABLED_SYMBOLS="BTC",
        PRIMARY_SYMBOL="BTC",
        INSTRUMENT="BTC",
        TRADINGVIEW_WEBHOOK_SECRET="hunter2",
    )
    notifier = _RecordingNotifier()
    app, executor, portfolio = _build_app(settings=settings, notifier=notifier)

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/tradingview",
            json={
                "secret": "wrong",
                "source": "tradingview",
                "symbol": "BTC",
                "price": 30000.0,
                "action": "long",
                "stop": 29950.0,
                "target": 30100.0,
            },
        )

    assert response.status_code == 401
    body = response.json()
    assert "detail" in body
    assert "secret" in body["detail"].lower()

    assert portfolio.open_position is None
    with session_scope() as session:
        paper_rows = session.execute(select(PaperTradeRow)).scalars().all()
    assert paper_rows == []

    kinds = notifier.kinds()
    assert "webhook.received" in kinds
    assert "webhook.invalid" in kinds


def test_missing_secret_when_required() -> None:
    settings = _settings(
        MARKET_TYPE="crypto",
        ENABLED_SYMBOLS="BTC",
        PRIMARY_SYMBOL="BTC",
        INSTRUMENT="BTC",
        TRADINGVIEW_WEBHOOK_SECRET="hunter2",
    )
    notifier = _RecordingNotifier()
    app, _, portfolio = _build_app(settings=settings, notifier=notifier)

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/tradingview",
            json={
                "source": "tradingview",
                "symbol": "BTC",
                "price": 30000.0,
                "action": "long",
                "stop": 29950.0,
                "target": 30100.0,
            },
        )
    assert response.status_code == 401
    assert portfolio.open_position is None


def test_secret_can_be_provided_via_header() -> None:
    settings = _settings(
        MARKET_TYPE="crypto",
        ENABLED_SYMBOLS="BTC",
        PRIMARY_SYMBOL="BTC",
        INSTRUMENT="BTC",
        TRADINGVIEW_WEBHOOK_SECRET="hunter2",
    )
    notifier = _RecordingNotifier()
    app, _, portfolio = _build_app(settings=settings, notifier=notifier)

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/tradingview",
            json={
                "source": "tradingview",
                "symbol": "BTC",
                "price": 30000.0,
                "action": "long",
                "stop": 29950.0,
                "target": 30100.0,
            },
            headers={"X-Webhook-Secret": "hunter2"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "accepted"
    assert portfolio.open_position is not None


# ---------------------------------------------------------------------------
# 3. Missing field -> 422
# ---------------------------------------------------------------------------
def test_missing_field_returns_422() -> None:
    settings = _settings(MARKET_TYPE="crypto", ENABLED_SYMBOLS="BTC", PRIMARY_SYMBOL="BTC", INSTRUMENT="BTC")
    notifier = _RecordingNotifier()
    app, _, portfolio = _build_app(settings=settings, notifier=notifier)

    with TestClient(app) as client:
        # Missing required ``action``.
        response = client.post(
            "/webhooks/tradingview",
            json={"source": "tradingview", "symbol": "BTC", "price": 30000.0},
        )

    assert response.status_code == 422
    assert portfolio.open_position is None


def test_unknown_action_rejected() -> None:
    settings = _settings(MARKET_TYPE="crypto", ENABLED_SYMBOLS="BTC", PRIMARY_SYMBOL="BTC", INSTRUMENT="BTC")
    notifier = _RecordingNotifier()
    app, _, portfolio = _build_app(settings=settings, notifier=notifier)

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/tradingview",
            json={
                "source": "tradingview",
                "symbol": "BTC",
                "price": 30000.0,
                "action": "diagonal",
            },
        )
    assert response.status_code == 422


def test_disabled_symbol_rejected_in_pipeline() -> None:
    settings = _settings(
        MARKET_TYPE="futures",
        ENABLED_SYMBOLS="MES",
        PRIMARY_SYMBOL="MES",
        INSTRUMENT="MES",
    )
    notifier = _RecordingNotifier()
    app, _, portfolio = _build_app(settings=settings, notifier=notifier)

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/tradingview",
            json={
                "source": "tradingview",
                "symbol": "MNQ1!",
                "price": 18000.0,
                "action": "long",
                "stop": 17995.0,
                "target": 18010.0,
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "rejected"
    assert "MNQ" in (body.get("reason") or "")
    assert portfolio.open_position is None


# ---------------------------------------------------------------------------
# 6. Risk-blocked signal -> 200 status="blocked", risk_block row written
# ---------------------------------------------------------------------------
def test_risk_blocked_when_kill_switch_tripped() -> None:
    settings = _settings(
        MARKET_TYPE="crypto",
        ENABLED_SYMBOLS="BTC",
        PRIMARY_SYMBOL="BTC",
        INSTRUMENT="BTC",
    )
    notifier = _RecordingNotifier()
    kill_switch = KillSwitch()
    app, _, portfolio = _build_app(
        settings=settings, notifier=notifier, kill_switch=kill_switch
    )
    kill_switch.trip("test-kill")

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/tradingview",
            json={
                "source": "tradingview",
                "symbol": "BTC",
                "price": 30000.0,
                "action": "long",
                "stop": 29950.0,
                "target": 30100.0,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "blocked"
    assert body["reason"] == "kill_switch"
    assert portfolio.open_position is None

    with session_scope() as session:
        rb_rows = session.execute(select(RiskBlockRow)).scalars().all()
    assert len(rb_rows) == 1
    assert rb_rows[0].rule == "kill_switch"

    assert "webhook.blocked" in notifier.kinds()


def test_risk_blocked_when_max_trades_per_day_exhausted() -> None:
    settings = _settings(
        MARKET_TYPE="crypto",
        ENABLED_SYMBOLS="BTC",
        PRIMARY_SYMBOL="BTC",
        INSTRUMENT="BTC",
        MAX_TRADES_PER_DAY=1,
    )
    notifier = _RecordingNotifier()
    app, executor, portfolio = _build_app(settings=settings, notifier=notifier)

    with TestClient(app) as client:
        first = client.post(
            "/webhooks/tradingview",
            json={
                "source": "tradingview",
                "symbol": "BTC",
                "price": 30000.0,
                "action": "long",
                "stop": 29950.0,
                "target": 30100.0,
            },
        )
        assert first.status_code == 200, first.text
        assert first.json()["status"] == "accepted"

        # Close so the next entry isn't blocked by max_open_positions; it
        # should be blocked by max_trades_per_day instead.
        executor.close_position(
            ts=datetime.now(timezone.utc),
            exit_raw_price=30050.0,
            exit_reason="manual_test",
        )

        second = client.post(
            "/webhooks/tradingview",
            json={
                "source": "tradingview",
                "symbol": "BTC",
                "price": 30000.0,
                "action": "long",
                "stop": 29950.0,
                "target": 30100.0,
            },
        )

    body = second.json()
    assert second.status_code == 200, second.text
    assert body["status"] == "blocked"
    assert body["reason"] == "max_trades_per_day"

    with session_scope() as session:
        closed_rows = session.execute(select(ClosedTradeRow)).scalars().all()
        rb_rows = session.execute(select(RiskBlockRow)).scalars().all()
    assert len(closed_rows) == 1
    assert len(rb_rows) == 1


# ---------------------------------------------------------------------------
# Close action handling
# ---------------------------------------------------------------------------
def test_close_action_flattens_open_position() -> None:
    settings = _settings(
        MARKET_TYPE="crypto",
        ENABLED_SYMBOLS="BTC",
        PRIMARY_SYMBOL="BTC",
        INSTRUMENT="BTC",
    )
    notifier = _RecordingNotifier()
    app, executor, portfolio = _build_app(settings=settings, notifier=notifier)

    with TestClient(app) as client:
        client.post(
            "/webhooks/tradingview",
            json={
                "source": "tradingview",
                "symbol": "BTC",
                "price": 30000.0,
                "action": "long",
                "stop": 29950.0,
                "target": 30100.0,
            },
        )
        assert portfolio.open_position is not None

        close_response = client.post(
            "/webhooks/tradingview",
            json={
                "source": "tradingview",
                "symbol": "BTC",
                "price": 30050.0,
                "action": "close",
            },
        )

    assert close_response.status_code == 200
    body = close_response.json()
    assert body["status"] == "closed"
    assert portfolio.open_position is None

    kinds = notifier.kinds()
    assert "webhook.closed" in kinds


def test_close_action_no_position_returns_noop() -> None:
    settings = _settings(
        MARKET_TYPE="crypto",
        ENABLED_SYMBOLS="BTC",
        PRIMARY_SYMBOL="BTC",
        INSTRUMENT="BTC",
    )
    notifier = _RecordingNotifier()
    app, _, _ = _build_app(settings=settings, notifier=notifier)

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/tradingview",
            json={
                "source": "tradingview",
                "symbol": "BTC",
                "price": 30000.0,
                "action": "close",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "noop"


# ---------------------------------------------------------------------------
# 7. Discord failure must not crash the endpoint
# ---------------------------------------------------------------------------
def test_notifier_failure_does_not_crash_endpoint() -> None:
    settings = _settings(
        MARKET_TYPE="crypto",
        ENABLED_SYMBOLS="BTC",
        PRIMARY_SYMBOL="BTC",
        INSTRUMENT="BTC",
    )
    notifier = _ExplodingNotifier()
    app, _, portfolio = _build_app(settings=settings, notifier=notifier)

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/tradingview",
            json={
                "source": "tradingview",
                "symbol": "BTC",
                "price": 30000.0,
                "action": "long",
                "stop": 29950.0,
                "target": 30100.0,
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    assert portfolio.open_position is not None
    # We *did* call notify multiple times despite every call raising —
    # the safety wrapper has to ride out every notify in the pipeline.
    assert notifier.calls >= 2


# ---------------------------------------------------------------------------
# Optional model-gate behavior
# ---------------------------------------------------------------------------
class _AlwaysRejectPredictor:
    """Stub predictor whose ``predict_setup`` always returns ``approved=False``."""

    def predict_setup(self, setup) -> Any:  # noqa: ARG002 - signature matches Predictor
        from models.predictor import Prediction

        return Prediction(
            probability=0.10,
            approved=False,
            threshold=0.50,
            model_name="stub",
            model_version="0",
        )


def test_predictor_rejection_blocks_trade() -> None:
    settings = _settings(
        MARKET_TYPE="crypto",
        ENABLED_SYMBOLS="BTC",
        PRIMARY_SYMBOL="BTC",
        INSTRUMENT="BTC",
    )
    notifier = _RecordingNotifier()
    app, _, portfolio = _build_app(
        settings=settings,
        notifier=notifier,
        predictor=_AlwaysRejectPredictor(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/tradingview",
            json={
                "source": "tradingview",
                "symbol": "BTC",
                "price": 30000.0,
                "action": "long",
                "stop": 29950.0,
                "target": 30100.0,
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "blocked"
    assert body["reason"] == "model_gate"
    assert portfolio.open_position is None


# ---------------------------------------------------------------------------
# Healthz
# ---------------------------------------------------------------------------
def test_healthz_endpoint() -> None:
    settings = _settings(MARKET_TYPE="crypto", ENABLED_SYMBOLS="BTC", PRIMARY_SYMBOL="BTC", INSTRUMENT="BTC")
    notifier = _RecordingNotifier()
    app, _, _ = _build_app(settings=settings, notifier=notifier)

    with TestClient(app) as client:
        response = client.get("/webhooks/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
