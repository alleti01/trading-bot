"""Shared helper to stub the workflow SignalEngine in wiring tests.

These tests exercise broker / options / order wiring, not signal
generation (covered by ``test_signal_engine.py``). They patch the
engine so a deterministic approved long signal is produced for the
given symbol.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from workflows.signal_engine import WorkflowSignal


def make_signal(symbol: str, *, direction: str = "long", price: float = 100.0) -> WorkflowSignal:
    if direction == "long":
        stop = price * 0.99
        target = price * 1.02
    else:
        stop = price * 1.01
        target = price * 0.98
    return WorkflowSignal(
        symbol=symbol.upper(),
        direction=direction,
        entry_price=price,
        stop_price=round(stop, 2),
        target_price=round(target, 2),
        confidence=0.70,
        approved=True,
        model_name=None,
        reason="stubbed",
    )


@contextmanager
def stub_market_open_signal(*, direction: str = "long", price: float = 100.0):
    """Patch SignalEngine so market-open gets a deterministic signal."""

    def _fake_generate(self, symbol):  # noqa: ANN001
        return make_signal(symbol, direction=direction, price=price)

    with patch(
        "workflows.signal_engine.SignalEngine.generate_signal",
        new=_fake_generate,
    ):
        yield
