"""Options layer tests — contract, chain, selection, greeks, execution, risk."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from config.settings import reload_settings
from options.chain import SyntheticChainProvider
from options.contract import build_occ_symbol, parse_occ_symbol
from options.execution import MockOptionsExecutor, OptionLeg
from options.greeks import black_scholes_greeks
from options.position_manager import OptionsPositionManager
from options.risk_rules import OptionsRiskConfig, OptionsRiskEngine
from options.selection import (
    SelectionConfig,
    select_directional_contract,
    select_iron_condor,
    select_vertical_spread,
)
from options.trader import OptionsTrader

_NOW = datetime(2026, 6, 16, 14, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# OCC symbol encoding
# ---------------------------------------------------------------------------
def test_occ_symbol_roundtrip() -> None:
    occ = build_occ_symbol("SPY", date(2026, 7, 18), 550.0, "call")
    assert occ == "SPY260718C00550000"
    parsed = parse_occ_symbol(occ)
    assert parsed.underlying == "SPY"
    assert parsed.strike == 550.0
    assert parsed.option_type == "call"
    assert parsed.expiry == date(2026, 7, 18)


def test_parse_invalid_occ_raises() -> None:
    with pytest.raises(ValueError):
        parse_occ_symbol("NOT_AN_OPTION")


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------
def test_black_scholes_atm_call_delta_near_half() -> None:
    g = black_scholes_greeks(
        spot=100, strike=100, days_to_expiry=30, volatility=0.2, option_type="call"
    )
    assert g is not None
    assert 0.45 < g.delta < 0.65
    assert g.price > 0
    assert g.theta < 0  # long option bleeds theta


def test_black_scholes_put_delta_negative() -> None:
    g = black_scholes_greeks(
        spot=100, strike=100, days_to_expiry=30, volatility=0.2, option_type="put"
    )
    assert g is not None
    assert g.delta < 0


def test_greeks_degenerate_inputs_return_none() -> None:
    assert black_scholes_greeks(
        spot=0, strike=100, days_to_expiry=30, volatility=0.2, option_type="call"
    ) is None


# ---------------------------------------------------------------------------
# Synthetic chain + selection
# ---------------------------------------------------------------------------
def _provider() -> SyntheticChainProvider:
    return SyntheticChainProvider(
        spot_by_symbol={"SPY": 550.0}, strike_step=5.0, now=_NOW
    )


def test_synthetic_chain_has_calls_and_puts() -> None:
    chain = _provider().get_chain("SPY")
    assert any(c.option_type == "call" for c in chain)
    assert any(c.option_type == "put" for c in chain)
    assert all(c.underlying == "SPY" for c in chain)


def test_select_directional_long_picks_call() -> None:
    contract = select_directional_contract(
        _provider(), underlying="SPY", direction="long",
        config=SelectionConfig(target_dte=30), now=_NOW,
    )
    assert contract is not None
    assert contract.option_type == "call"
    assert 7 <= contract.days_to_expiry(now=_NOW) <= 60


def test_select_directional_short_picks_put() -> None:
    contract = select_directional_contract(
        _provider(), underlying="SPY", direction="short",
        config=SelectionConfig(target_dte=30), now=_NOW,
    )
    assert contract is not None
    assert contract.option_type == "put"


def test_select_vertical_spread_returns_two_legs() -> None:
    legs = select_vertical_spread(
        _provider(), underlying="SPY", direction="long",
        config=SelectionConfig(spread_width_strikes=1), now=_NOW,
    )
    assert legs is not None
    long_leg, short_leg = legs
    assert long_leg.option_type == "call"
    assert short_leg.strike > long_leg.strike  # bull call: short higher strike


def test_select_iron_condor_returns_four_legs() -> None:
    legs = select_iron_condor(_provider(), underlying="SPY", now=_NOW)
    assert legs is not None
    assert set(legs.keys()) == {"short_put", "long_put", "short_call", "long_call"}


# ---------------------------------------------------------------------------
# Risk rules
# ---------------------------------------------------------------------------
def test_risk_blocks_premium_over_cap() -> None:
    engine = OptionsRiskEngine(OptionsRiskConfig(max_premium_per_trade=10.0))
    contract = select_directional_contract(
        _provider(), underlying="SPY", direction="long", now=_NOW
    )
    assert contract is not None
    decision = engine.evaluate_entry(contract, qty=1, open_positions=0, now=_NOW)
    assert not decision.approved
    assert "premium" in decision.reason


def test_risk_blocks_max_open_positions() -> None:
    engine = OptionsRiskEngine(OptionsRiskConfig(max_open_positions=2))
    contract = select_directional_contract(
        _provider(), underlying="SPY", direction="long", now=_NOW
    )
    assert contract is not None
    decision = engine.evaluate_entry(contract, qty=1, open_positions=2, now=_NOW)
    assert not decision.approved
    assert "max_open_positions" in decision.reason


def test_risk_approves_reasonable_trade() -> None:
    engine = OptionsRiskEngine(
        OptionsRiskConfig(max_premium_per_trade=100_000.0, max_open_positions=5)
    )
    contract = select_directional_contract(
        _provider(), underlying="SPY", direction="long", now=_NOW
    )
    assert contract is not None
    decision = engine.evaluate_entry(contract, qty=1, open_positions=0, now=_NOW)
    assert decision.approved


# ---------------------------------------------------------------------------
# Execution (mock)
# ---------------------------------------------------------------------------
def test_mock_executor_single_leg() -> None:
    executor = MockOptionsExecutor()
    contract = select_directional_contract(
        _provider(), underlying="SPY", direction="long", now=_NOW
    )
    assert contract is not None
    result = executor.place_single_leg(
        contract=contract, action="buy_to_open", qty=1
    )
    assert result.success
    assert result.simulated
    assert result.order_class == "simple"
    assert len(result.legs) == 1


def test_mock_executor_multi_leg() -> None:
    executor = MockOptionsExecutor()
    legs = select_vertical_spread(
        _provider(), underlying="SPY", direction="long", now=_NOW
    )
    assert legs is not None
    long_leg, short_leg = legs
    result = executor.place_multi_leg(
        underlying="SPY",
        legs=[
            OptionLeg(contract=long_leg, action="buy_to_open"),
            OptionLeg(contract=short_leg, action="sell_to_open"),
        ],
        qty=1,
        order_class="spread",
    )
    assert result.success
    assert len(result.legs) == 2


# ---------------------------------------------------------------------------
# Position manager
# ---------------------------------------------------------------------------
def test_position_manager_open_and_persist(tmp_path) -> None:
    executor = MockOptionsExecutor()
    provider = _provider()
    pm = OptionsPositionManager(
        executor=executor,
        chain_provider=provider,
        state_path=tmp_path / "positions.json",
    )
    contract = select_directional_contract(
        provider, underlying="SPY", direction="long", now=_NOW
    )
    assert contract is not None
    pm.open_position(contract, qty=1, thesis="test", now=_NOW)
    assert pm.open_count() == 1
    assert (tmp_path / "positions.json").exists()

    # Reload from disk
    pm2 = OptionsPositionManager(
        executor=executor,
        chain_provider=provider,
        state_path=tmp_path / "positions.json",
    )
    assert pm2.open_count() == 1


def test_position_manager_auto_closes_near_expiry(tmp_path) -> None:
    from options.position_manager import ManagerConfig

    executor = MockOptionsExecutor()
    # Chain with a contract expiring in 3 days; auto_close_dte=5 → close.
    provider = SyntheticChainProvider(
        spot_by_symbol={"SPY": 550.0}, expirations_days=[3], now=_NOW
    )
    pm = OptionsPositionManager(
        executor=executor,
        chain_provider=provider,
        state_path=tmp_path / "p.json",
        config=ManagerConfig(auto_close_dte=5, auto_roll=False),
    )
    contract = select_directional_contract(
        provider, underlying="SPY", direction="long",
        config=SelectionConfig(min_dte=0, max_dte=10, target_dte=3), now=_NOW,
    )
    assert contract is not None
    pm.open_position(contract, qty=1, now=_NOW)
    actions = pm.manage_cycle(now=_NOW)
    assert any(a["action"] == "close" for a in actions)
    assert pm.open_count() == 0


# ---------------------------------------------------------------------------
# Trader end-to-end (dry run)
# ---------------------------------------------------------------------------
def _settings(monkeypatch, **overrides):
    monkeypatch.setenv("OPTIONS_ENABLED", "true")
    monkeypatch.setenv("OPTIONS_ENABLED_UNDERLYINGS", "SPY,QQQ")
    monkeypatch.setenv("OPTIONS_MAX_PREMIUM_PER_TRADE", "100000")
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    return reload_settings()


def test_trader_directional_opens_position(monkeypatch, tmp_path) -> None:
    settings = _settings(
        monkeypatch, OPTIONS_STATE_PATH=str(tmp_path / "pos.json")
    )
    trader = OptionsTrader.for_dry_run(settings, now=_NOW)
    result = trader.handle_signal(
        underlying="SPY", direction="long", now=_NOW, thesis="vwap pullback"
    )
    assert result["status"] == "opened"
    assert trader.pm.open_count() == 1


def test_trader_blocks_premium_over_cap(monkeypatch, tmp_path) -> None:
    settings = _settings(
        monkeypatch,
        OPTIONS_MAX_PREMIUM_PER_TRADE="1",
        OPTIONS_STATE_PATH=str(tmp_path / "pos.json"),
    )
    trader = OptionsTrader.for_dry_run(settings, now=_NOW)
    result = trader.handle_signal(underlying="SPY", direction="long", now=_NOW)
    assert result["status"] == "blocked"


def test_trader_skips_disabled_underlying(monkeypatch, tmp_path) -> None:
    settings = _settings(
        monkeypatch,
        OPTIONS_ENABLED_UNDERLYINGS="SPY",
        OPTIONS_STATE_PATH=str(tmp_path / "pos.json"),
    )
    trader = OptionsTrader.for_dry_run(settings, now=_NOW)
    result = trader.handle_signal(underlying="TSLA", direction="long", now=_NOW)
    assert result["status"] == "skipped"


def test_trader_disabled_when_options_off(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPTIONS_ENABLED", "false")
    monkeypatch.setenv("OPTIONS_STATE_PATH", str(tmp_path / "pos.json"))
    settings = reload_settings()
    trader = OptionsTrader.for_dry_run(settings, now=_NOW)
    result = trader.handle_signal(underlying="SPY", direction="long", now=_NOW)
    assert result["status"] == "disabled"


def test_trader_vertical_spread(monkeypatch, tmp_path) -> None:
    settings = _settings(
        monkeypatch,
        OPTIONS_STRATEGY="vertical_spread",
        OPTIONS_STATE_PATH=str(tmp_path / "pos.json"),
    )
    trader = OptionsTrader.for_dry_run(settings, now=_NOW)
    result = trader.handle_signal(underlying="SPY", direction="long", now=_NOW)
    assert result["status"] == "opened"


def test_trader_iron_condor(monkeypatch, tmp_path) -> None:
    settings = _settings(
        monkeypatch,
        OPTIONS_STRATEGY="iron_condor",
        OPTIONS_STATE_PATH=str(tmp_path / "pos.json"),
    )
    trader = OptionsTrader.for_dry_run(settings, now=_NOW)
    result = trader.handle_signal(underlying="SPY", direction="long", now=_NOW)
    assert result["status"] == "opened"
