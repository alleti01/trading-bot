"""Triple-barrier (TP/SL/time) labeling.

For each ``Setup`` we walk forward bar-by-bar from the bar AFTER the
setup timestamp for up to ``max_hold_bars`` bars and label:

- ``1`` — TP was reached cleanly before SL.
- ``0`` — SL was reached, OR same-bar TP+SL ambiguity (see below), OR
  neither was reached within ``max_hold_bars`` (time-out).

Same-bar ambiguity rule
-----------------------
If a single future bar's [low, high] range straddles BOTH the take-profit
and stop-loss levels, we cannot tell which was hit first without intrabar
data. We resolve this conservatively as **SL hit first** (label ``0``,
``exit_reason="sl"``). This is the standard Lopez de Prado treatment.

Lookahead invariant
-------------------
The labeler is the only place future bars are used, and they are used
ONLY for label computation — never to recompute features. Features are
already frozen on the ``Setup`` at signal time.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from app.logging_config import get_logger
from strategies.base import Setup

ExitReason = Literal["tp", "sl", "time"]


class LabelResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: int  # 0 or 1
    exit_reason: ExitReason
    exit_bar: int  # 1-based; 1 = first future bar after the setup
    exit_price: float
    bars_held: int


def label_setup(
    setup: Setup,
    future_bars: pd.DataFrame,
    max_hold_bars: int,
) -> LabelResult:
    """Walk forward bar-by-bar to determine the outcome of a setup.

    ``future_bars`` is expected to be the OHLCV slice starting at the bar
    AFTER ``setup.timestamp`` (i.e., the next bar the trade can fill on).
    """
    if max_hold_bars <= 0:
        raise ValueError("max_hold_bars must be positive")

    bars_to_eval = future_bars.iloc[:max_hold_bars]
    if bars_to_eval.empty:
        # No future bars at all — we couldn't even take the trade. Conservative
        # zero label, exit at entry price, zero bars held.
        return LabelResult(
            label=0,
            exit_reason="time",
            exit_bar=0,
            exit_price=setup.entry_price,
            bars_held=0,
        )

    is_long = setup.direction == "long"
    tp = setup.target_price
    sl = setup.stop_price

    for i, (_, bar) in enumerate(bars_to_eval.iterrows()):
        h = float(bar["high"])
        lo = float(bar["low"])

        if is_long:
            tp_hit = h >= tp
            sl_hit = lo <= sl
        else:
            tp_hit = lo <= tp
            sl_hit = h >= sl

        if tp_hit and sl_hit:
            # Same-bar ambiguity → conservative SL-first.
            return LabelResult(
                label=0,
                exit_reason="sl",
                exit_bar=i + 1,
                exit_price=sl,
                bars_held=i + 1,
            )
        if tp_hit:
            return LabelResult(
                label=1,
                exit_reason="tp",
                exit_bar=i + 1,
                exit_price=tp,
                bars_held=i + 1,
            )
        if sl_hit:
            return LabelResult(
                label=0,
                exit_reason="sl",
                exit_bar=i + 1,
                exit_price=sl,
                bars_held=i + 1,
            )

    # Time-out: neither barrier was hit within max_hold_bars.
    last_bar = bars_to_eval.iloc[-1]
    return LabelResult(
        label=0,
        exit_reason="time",
        exit_bar=len(bars_to_eval),
        exit_price=float(last_bar["close"]),
        bars_held=len(bars_to_eval),
    )


def label_setups(
    setups: list[Setup],
    ohlcv_df: pd.DataFrame,
    max_hold_bars: int,
) -> list[LabelResult]:
    """Batch labeler. Each setup is located in ``ohlcv_df`` by timestamp."""
    log = get_logger("labeling.tp_sl_labeler")
    if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
        raise ValueError("ohlcv_df must have a DatetimeIndex")

    results: list[LabelResult] = []
    for setup in setups:
        try:
            i = ohlcv_df.index.get_loc(setup.timestamp)
        except KeyError as e:
            raise KeyError(
                f"setup.timestamp {setup.timestamp} not in ohlcv_df index"
            ) from e
        future = ohlcv_df.iloc[i + 1 : i + 1 + max_hold_bars]
        results.append(label_setup(setup, future, max_hold_bars))

    n_pos = sum(r.label for r in results)
    by_reason: dict[str, int] = {"tp": 0, "sl": 0, "time": 0}
    for r in results:
        by_reason[r.exit_reason] += 1

    log.info(
        "setups.labeled",
        n=len(results),
        positive=n_pos,
        positive_rate=(n_pos / len(results)) if results else 0.0,
        by_reason=by_reason,
        max_hold_bars=max_hold_bars,
    )
    return results
