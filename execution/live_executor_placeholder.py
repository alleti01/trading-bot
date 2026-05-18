"""Live executor placeholder.

This is the **second line of defense** against accidental live trading.
Even if ``MODE=LIVE`` and ``LIVE_ADAPTER_CONFIRMED=true``, this class will
refuse to instantiate until a real broker/exchange adapter is implemented.

To actually go live (in some far-future state):
1. Implement a real adapter in ``execution/futures_adapter_placeholder.py``
   or ``execution/crypto_adapter_placeholder.py`` (and rename it).
2. Replace this module's ``LiveExecutor`` with one that delegates to that
   adapter.
3. Add integration tests that prove dry-run order submission works against
   a sandbox account.
"""

from __future__ import annotations

from execution.base import Executor, Fill, Order


class LiveExecutorRefusedError(RuntimeError):
    """Raised whenever someone tries to use the live executor placeholder."""


class LiveExecutor(Executor):
    """A placeholder that refuses to do anything dangerous."""

    def __init__(self) -> None:
        raise LiveExecutorRefusedError(
            "LiveExecutor is not implemented. Real-money trading is disabled "
            "in this MVP. Implement a real broker/exchange adapter first."
        )

    def submit(self, order: Order) -> Fill:  # pragma: no cover - unreachable
        raise LiveExecutorRefusedError("LiveExecutor cannot submit orders.")
