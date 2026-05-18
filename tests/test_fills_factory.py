"""Factory wiring for ``make_fills_model`` across futures vs crypto.

Two regressions the factory must prevent:

1. Crypto instruments must accept ``CRYPTO_SLIPPAGE_BPS`` /
   ``CRYPTO_FEE_BPS`` instead of silently inheriting the futures-only
   ``SLIPPAGE_TICKS`` / ``COMMISSION_PER_CONTRACT`` knobs.
2. Crypto fills must not be a no-op — non-zero slippage and fees apply.
"""

from __future__ import annotations

import pytest

from backtesting.fills import (
    CryptoFillsModel,
    FuturesFillsModel,
    make_fills_model,
)


def test_factory_returns_futures_model_for_mes() -> None:
    fills = make_fills_model("MES", slippage_ticks=1.0, commission_per_contract=1.5)
    assert isinstance(fills, FuturesFillsModel)
    assert fills.slippage_ticks == 1.0
    assert fills.commission_per_contract == 1.5


def test_factory_returns_crypto_model_for_btc_with_bps_kwargs() -> None:
    fills = make_fills_model(
        "BTC",
        slippage_ticks=0.0,
        commission_per_contract=0.0,
        crypto_slippage_bps=2.0,
        crypto_fee_bps=10.0,
    )
    assert isinstance(fills, CryptoFillsModel)
    assert fills.slippage_bps == 2.0
    assert fills.fee_bps == 10.0


def test_crypto_fills_apply_slippage_and_fees() -> None:
    fills = make_fills_model(
        "BTC",
        slippage_ticks=0.0,
        commission_per_contract=0.0,
        crypto_slippage_bps=10.0,  # 10 bps = 0.10%
        crypto_fee_bps=5.0,
    )
    entry = fills.entry(direction="long", raw_price=50_000.0, quantity=1.0)
    # Slippage adverse: 50000 * 0.001 = 50 → fill = 50050.
    assert entry.fill_price == pytest.approx(50_050.0)
    assert entry.slippage == pytest.approx(50.0)
    # Fee = fill * 5bps = 50050 * 0.0005 ≈ 25.025
    assert entry.commission == pytest.approx(50_050.0 * 0.0005, rel=1e-6)


def test_factory_warns_when_crypto_drops_tick_config(capsys) -> None:
    """The factory must surface a warning when futures-only knobs are
    passed along with a crypto instrument — silent absorption hides
    config bugs. structlog's PrintLoggerFactory writes to stdout, so we
    capture its output rather than going through stdlib logging.
    """
    make_fills_model(
        "BTC",
        slippage_ticks=2.0,                # would silently be ignored
        commission_per_contract=1.5,       # would silently be ignored
        crypto_slippage_bps=1.0,
        crypto_fee_bps=5.0,
    )
    captured = capsys.readouterr()
    assert "fills.crypto_drops_tick_config" in (captured.out + captured.err)
