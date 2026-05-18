"""Futures broker adapter placeholder (Tradovate / Rithmic / etc.).

Refuses to instantiate. Real implementation is out of scope for the MVP.
"""

from __future__ import annotations

from execution.live_executor_placeholder import LiveExecutorRefusedError


class FuturesAdapter:
    def __init__(self, *_: object, **__: object) -> None:
        raise LiveExecutorRefusedError(
            "FuturesAdapter is not implemented. Configure a real broker adapter first."
        )
