"""Pydantic models for the TradingView webhook endpoint.

The bot receives external candidate signals as JSON from TradingView (or
any other alert provider that can POST JSON). The schema is intentionally
forgiving on input — TradingView strips quoting in alert messages, so
``price`` arrives as a number sometimes and as a string other times —
and strict on output: every accepted payload is normalized into a
:class:`NormalizedWebhookOrder` before any trading-pipeline code sees it.

Two layers:

1. :class:`TradingViewWebhookPayload` — the raw inbound shape. Used as
   the FastAPI request body so framework validation produces a clean
   422 for malformed JSON.
2. :class:`NormalizedWebhookOrder` — what the trading pipeline consumes.
   Always uppercase symbol, always ``"long"`` / ``"short"`` / ``"close"``
   for direction, always a float ``price``, always a tz-aware
   ``received_at``.

The endpoint never touches live broker execution. Even when a payload
is approved end-to-end the only side-effect is a paper trade.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Action vocabulary the upstream alerts may use. The validator below
# collapses ``buy`` -> ``long`` and ``sell`` -> ``short`` so the rest of
# the pipeline only ever sees the canonical 3-way set.
ActionLiteral = Literal["long", "short", "close", "buy", "sell"]
DirectionLiteral = Literal["long", "short", "close"]


def _coerce_float(v: Any) -> float:
    """Convert string/number to float; raise the same ValueError shape pydantic emits."""
    if v is None:
        raise ValueError("must be a number")
    if isinstance(v, bool):
        # bool is a subclass of int; reject explicitly so ``True`` doesn't sneak in
        raise ValueError("must be a number, not a boolean")
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"could not parse {v!r} as float") from e


class TradingViewWebhookPayload(BaseModel):
    """Raw TradingView alert body.

    The expected JSON shape is:

    .. code-block:: json

        {
            "secret": "shared-secret",
            "source": "tradingview",
            "symbol": "MNQ1!",
            "time": "2026-05-19T14:00:00Z",
            "price": "18000.25",
            "action": "long",
            "strategy": "vwap_pullback",
            "timeframe": "1m",
            "stop": 17995.0,
            "target": 18010.0
        }

    ``stop`` / ``target`` are optional extensions to TradingView's
    canonical alert variables. When absent, the endpoint synthesizes
    defaults from ``WEBHOOK_DEFAULT_STOP_TICKS`` /
    ``WEBHOOK_DEFAULT_TARGET_TICKS`` so the risk engine has the
    distances it needs.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    secret: Optional[str] = None
    source: str = Field(default="tradingview")
    symbol: str
    time: Optional[str] = None
    price: Union[str, float]
    action: ActionLiteral
    strategy: Optional[str] = None
    timeframe: Optional[str] = None
    stop: Optional[Union[str, float]] = None
    target: Optional[Union[str, float]] = None

    @field_validator("symbol")
    @classmethod
    def _symbol_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("symbol must be non-empty")
        return v.strip()

    @field_validator("price")
    @classmethod
    def _price_parses(cls, v: Any) -> float:
        # We always store as float so downstream code never has to repeat
        # the str/number coercion dance.
        return _coerce_float(v)

    @field_validator("stop", "target")
    @classmethod
    def _opt_float(cls, v: Any) -> Optional[float]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return _coerce_float(v)


class NormalizedWebhookOrder(BaseModel):
    """The trading pipeline's view of an accepted webhook signal.

    Built only by :mod:`webhooks.tradingview` after symbol normalization
    and direction collapsing. The risk engine, paper executor, and the
    notifier all consume this — never the raw payload.
    """

    model_config = ConfigDict(frozen=True)

    raw_symbol: str          # e.g. "MNQ1!" — kept for the audit trail
    symbol: str              # normalized, uppercase, e.g. "MNQ"
    direction: DirectionLiteral
    price: float
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    strategy: str = "tradingview"
    timeframe: Optional[str] = None
    source: str = "tradingview"
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_time: Optional[str] = None

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @field_validator("received_at")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        # ``datetime.now()`` without tz is a common test bug; refuse so
        # downstream code (which prints isoformat) can't blow up.
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------
ResponseStatus = Literal[
    "accepted",   # paper trade opened
    "blocked",    # risk engine refused
    "closed",     # action=close hit and a position was flattened
    "noop",       # action=close but nothing was open
    "rejected",   # validation failed (bad symbol, bad direction, etc.)
]


class WebhookResponse(BaseModel):
    """The JSON the endpoint always returns.

    We always reply with structured info even on error paths so
    TradingView (or any operator-side replay tool) can log a meaningful
    reason. We never echo the secret, even when it's invalid — to avoid
    leaking it back to a hostile client that's probing.
    """

    status: ResponseStatus
    symbol: Optional[str] = None
    direction: Optional[DirectionLiteral] = None
    reason: Optional[str] = None
    detail: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ActionLiteral",
    "DirectionLiteral",
    "NormalizedWebhookOrder",
    "ResponseStatus",
    "TradingViewWebhookPayload",
    "WebhookResponse",
]
