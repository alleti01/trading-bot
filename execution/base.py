"""Abstract Executor interface.

Concrete implementations:
- ``PaperExecutor``                — Day 5, simulates fills.
- ``LiveExecutorPlaceholder``      — refuses to instantiate (Day 1).
- ``FuturesAdapterPlaceholder``    — refuses to instantiate (Day 1).
- ``CryptoAdapterPlaceholder``     — refuses to instantiate (Day 1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class Order:
    instrument: str
    direction: Literal["long", "short"]
    quantity: float
    entry_price: float
    stop_price: float
    target_price: float
    setup_id: str | None = None


@dataclass(frozen=True)
class Fill:
    order: Order
    fill_ts: datetime
    fill_price: float
    commission: float
    slippage: float


class Executor(ABC):
    """All trade routing must go through an Executor implementation."""

    @abstractmethod
    def submit(self, order: Order) -> Fill:
        """Submit an order and return the resulting Fill, or raise."""
