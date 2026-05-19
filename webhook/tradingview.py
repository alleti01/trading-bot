"""Validate TradingView-style webhook signals.

The bot ingests external alerts as *advisory* inputs only. They flow
through the same risk engine + caps that govern internal signals, and
they are accepted only if the symbol is in ``ENABLED_SYMBOLS``. Live
broker execution is out of scope for the MVP, so this module just
returns a structured :class:`WebhookSignal` that downstream code (paper
mode, future live adapter) can consume.

Expected payload shape (TradingView's standard alert message JSON):

.. code-block:: json

    {
        "symbol": "MES",
        "direction": "long",
        "price": 4500.25,
        "stop": 4495.0,
        "target": 4510.0,
        "strategy": "external_breakout",
        "ts": "2026-05-19T14:00:00Z",
        "secret": "..."
    }

``secret`` is optional but, if provided in settings, must match before
the signal is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from app.logging_config import get_logger
from config.instruments import SymbolUniverse


class InvalidWebhookSignal(ValueError):
    """Raised when a webhook payload is malformed or refers to a
    disabled / unknown symbol. The HTTP layer should turn this into
    a 400 Bad Request without further trade-side processing."""


@dataclass(frozen=True)
class WebhookSignal:
    """A validated webhook signal.

    The ``source`` field exists so downstream systems can keep
    webhook-driven setups separate in reports / journals from
    internally-detected ones.
    """

    symbol: str
    direction: str          # "long" | "short"
    price: float
    stop_price: Optional[float]
    target_price: Optional[float]
    strategy: str
    ts: datetime
    source: str = "tradingview"


def _normalize_direction(d: Any) -> str:
    s = str(d or "").strip().lower()
    if s in ("long", "buy"):
        return "long"
    if s in ("short", "sell"):
        return "short"
    raise InvalidWebhookSignal(
        f"Unknown direction {d!r}; expected 'long'/'short' (or 'buy'/'sell')."
    )


def _coerce_ts(raw: Any) -> datetime:
    if raw is None or raw == "":
        return datetime.now(timezone.utc)
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    s = str(raw).strip()
    # Accept the ISO Z suffix that TradingView emits.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        ts = datetime.fromisoformat(s)
    except ValueError as e:
        raise InvalidWebhookSignal(f"Invalid timestamp {raw!r}: {e}") from e
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def validate_webhook_signal(
    payload: Mapping[str, Any],
    *,
    universe: SymbolUniverse,
    expected_secret: Optional[str] = None,
) -> WebhookSignal:
    """Validate + normalize a TradingView-style alert.

    Behavior:

    - Empty or non-dict payload → :class:`InvalidWebhookSignal`.
    - Missing required keys (``symbol``, ``direction``, ``price``) →
      :class:`InvalidWebhookSignal`.
    - Unknown direction → :class:`InvalidWebhookSignal`.
    - ``secret`` mismatch (when ``expected_secret`` is set) →
      :class:`InvalidWebhookSignal`.
    - **Symbol not in ``universe`` → :class:`InvalidWebhookSignal`**.
      This is the headline guarantee: the bot never trades a symbol
      operations did not whitelist, even if a third-party webhook says
      to.

    On success returns a :class:`WebhookSignal` ready to be turned
    into a :class:`Setup` by the caller.
    """
    log = get_logger("webhook.tradingview")

    if not isinstance(payload, Mapping) or not payload:
        raise InvalidWebhookSignal("Webhook payload must be a non-empty JSON object.")

    if expected_secret:
        provided = str(payload.get("secret", "")).strip()
        if provided != expected_secret:
            log.warning("webhook.bad_secret")
            raise InvalidWebhookSignal("Invalid or missing webhook secret.")

    raw_symbol = payload.get("symbol")
    if not raw_symbol:
        raise InvalidWebhookSignal("Webhook missing 'symbol'.")
    symbol = str(raw_symbol).strip().upper()

    if symbol not in universe:
        log.warning(
            "webhook.symbol_rejected",
            symbol=symbol,
            enabled=universe.as_list(),
            note="Symbol is not in ENABLED_SYMBOLS.",
        )
        raise InvalidWebhookSignal(
            f"Symbol {symbol!r} is not in ENABLED_SYMBOLS "
            f"({universe.as_list()}); rejecting webhook."
        )

    direction = _normalize_direction(payload.get("direction"))

    raw_price = payload.get("price")
    if raw_price is None:
        raise InvalidWebhookSignal("Webhook missing 'price'.")
    try:
        price = float(raw_price)
    except (TypeError, ValueError) as e:
        raise InvalidWebhookSignal(f"Invalid price {raw_price!r}: {e}") from e

    def _opt_float(name: str) -> Optional[float]:
        val = payload.get(name)
        if val is None or val == "":
            return None
        try:
            return float(val)
        except (TypeError, ValueError) as e:
            raise InvalidWebhookSignal(f"Invalid {name} {val!r}: {e}") from e

    signal = WebhookSignal(
        symbol=symbol,
        direction=direction,
        price=price,
        stop_price=_opt_float("stop") or _opt_float("stop_price"),
        target_price=_opt_float("target") or _opt_float("target_price"),
        strategy=str(payload.get("strategy") or "tradingview"),
        ts=_coerce_ts(payload.get("ts") or payload.get("timestamp")),
    )
    log.info(
        "webhook.accepted",
        symbol=signal.symbol,
        direction=signal.direction,
        strategy=signal.strategy,
    )
    return signal
