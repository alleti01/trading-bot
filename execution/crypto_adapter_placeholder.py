"""Crypto exchange adapter placeholder (Binance / Bybit / etc.).

Refuses to instantiate. Real implementation is out of scope for the MVP.
"""

from __future__ import annotations

from execution.live_executor_placeholder import LiveExecutorRefusedError


class CryptoAdapter:
    def __init__(self, *_: object, **__: object) -> None:
        raise LiveExecutorRefusedError(
            "CryptoAdapter is not implemented. Configure a real exchange adapter first."
        )
