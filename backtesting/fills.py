"""Slippage + commission models, polymorphic by market type.

The MVP supports two fill regimes:

- ``FuturesFillsModel`` — slippage in ticks (per ``InstrumentSpec.tick_size``)
  and a flat per-contract commission. This matches MES/MNQ retail brokers.
- ``CryptoFillsModel`` — slippage in basis points and a bps-based fee. This
  is a reasonable Day-4 placeholder; a real exchange adapter would replace it
  with maker/taker semantics later.

Convention: slippage is **always adverse**. A long gets a higher fill on
entry, a lower fill on exit. A short gets the mirror. We never let the
backtester pretend it filled at the favorable side of the spread.

Day 4 sticks to a *signal-on-bar-close, fill-on-next-bar-open* model.
The Engine is responsible for choosing the bar; the FillsModel only
shifts the price by slippage and computes commission.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from config.instruments import InstrumentSpec, get_instrument


@dataclass(frozen=True)
class FillCosts:
    fill_price: float
    slippage: float        # absolute price movement applied (always >= 0)
    commission: float      # dollars per side, positive


class FillsModel(ABC):
    """Computes adverse slippage + commission for a single side of a trade."""

    @abstractmethod
    def entry(self, *, direction: str, raw_price: float, quantity: float) -> FillCosts: ...

    @abstractmethod
    def exit(self, *, direction: str, raw_price: float, quantity: float) -> FillCosts: ...


# ---------------------------------------------------------------------------
# Futures
# ---------------------------------------------------------------------------
class FuturesFillsModel(FillsModel):
    """Tick-based slippage + flat per-contract commission."""

    def __init__(
        self,
        instrument: InstrumentSpec | str,
        *,
        slippage_ticks: float,
        commission_per_contract: float,
    ) -> None:
        self.spec = (
            instrument if isinstance(instrument, InstrumentSpec) else get_instrument(instrument)
        )
        if self.spec.market_type != "futures":
            raise ValueError(
                f"FuturesFillsModel got non-futures instrument {self.spec.symbol!r} "
                f"(market_type={self.spec.market_type!r})"
            )
        if slippage_ticks < 0:
            raise ValueError("slippage_ticks must be non-negative")
        if commission_per_contract < 0:
            raise ValueError("commission_per_contract must be non-negative")
        self.slippage_ticks = float(slippage_ticks)
        self.commission_per_contract = float(commission_per_contract)

    def _slippage(self) -> float:
        return self.slippage_ticks * self.spec.tick_size

    def _commission(self, quantity: float) -> float:
        return self.commission_per_contract * abs(float(quantity))

    def entry(self, *, direction: str, raw_price: float, quantity: float) -> FillCosts:
        s = self._slippage()
        if direction == "long":
            fill = raw_price + s
        elif direction == "short":
            fill = raw_price - s
        else:
            raise ValueError(f"Unknown direction {direction!r}")
        return FillCosts(fill_price=fill, slippage=s, commission=self._commission(quantity))

    def exit(self, *, direction: str, raw_price: float, quantity: float) -> FillCosts:
        s = self._slippage()
        if direction == "long":
            fill = raw_price - s
        elif direction == "short":
            fill = raw_price + s
        else:
            raise ValueError(f"Unknown direction {direction!r}")
        return FillCosts(fill_price=fill, slippage=s, commission=self._commission(quantity))


# ---------------------------------------------------------------------------
# Crypto
# ---------------------------------------------------------------------------
class CryptoFillsModel(FillsModel):
    """Basis-point slippage + basis-point fee. Placeholder for Day 4."""

    def __init__(
        self,
        instrument: InstrumentSpec | str,
        *,
        slippage_bps: float = 1.0,
        fee_bps: float = 5.0,
    ) -> None:
        self.spec = (
            instrument if isinstance(instrument, InstrumentSpec) else get_instrument(instrument)
        )
        if self.spec.market_type != "crypto":
            raise ValueError(f"CryptoFillsModel got non-crypto instrument {self.spec.symbol!r}")
        if slippage_bps < 0 or fee_bps < 0:
            raise ValueError("slippage_bps and fee_bps must be non-negative")
        self.slippage_bps = float(slippage_bps)
        self.fee_bps = float(fee_bps)

    def _slippage_amount(self, price: float) -> float:
        return price * (self.slippage_bps / 10_000.0)

    def _commission(self, price: float, quantity: float) -> float:
        return abs(float(quantity)) * price * (self.fee_bps / 10_000.0)

    def entry(self, *, direction: str, raw_price: float, quantity: float) -> FillCosts:
        s = self._slippage_amount(raw_price)
        if direction == "long":
            fill = raw_price + s
        elif direction == "short":
            fill = raw_price - s
        else:
            raise ValueError(f"Unknown direction {direction!r}")
        return FillCosts(fill_price=fill, slippage=s, commission=self._commission(fill, quantity))

    def exit(self, *, direction: str, raw_price: float, quantity: float) -> FillCosts:
        s = self._slippage_amount(raw_price)
        if direction == "long":
            fill = raw_price - s
        elif direction == "short":
            fill = raw_price + s
        else:
            raise ValueError(f"Unknown direction {direction!r}")
        return FillCosts(fill_price=fill, slippage=s, commission=self._commission(fill, quantity))


def make_fills_model(
    instrument: str,
    *,
    slippage_ticks: float,
    commission_per_contract: float,
    crypto_slippage_bps: float | None = None,
    crypto_fee_bps: float | None = None,
) -> FillsModel:
    """Construct the right FillsModel for an instrument by market type.

    For futures the tick-based knobs apply directly. For crypto the
    bps-based knobs apply and the futures-only knobs are *ignored* — we
    log a warning if the operator passed non-default tick/commission
    values into a crypto instrument so a misconfigured ``.env`` is
    visible at boot rather than silently absorbed.
    """
    from app.logging_config import get_logger

    log = get_logger("backtesting.fills")
    spec = get_instrument(instrument)
    if spec.market_type == "futures":
        return FuturesFillsModel(
            spec,
            slippage_ticks=slippage_ticks,
            commission_per_contract=commission_per_contract,
        )

    # Crypto path. Default to the model's safe defaults when not provided.
    if slippage_ticks > 0 or commission_per_contract > 0:
        log.warning(
            "fills.crypto_drops_tick_config",
            instrument=instrument,
            slippage_ticks=slippage_ticks,
            commission_per_contract=commission_per_contract,
            note="Use CRYPTO_SLIPPAGE_BPS / CRYPTO_FEE_BPS for crypto.",
        )
    kwargs: dict[str, float] = {}
    if crypto_slippage_bps is not None:
        kwargs["slippage_bps"] = float(crypto_slippage_bps)
    if crypto_fee_bps is not None:
        kwargs["fee_bps"] = float(crypto_fee_bps)
    return CryptoFillsModel(spec, **kwargs)
