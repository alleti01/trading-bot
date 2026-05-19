"""TradingView webhook FastAPI router + processor.

Pipeline (one HTTP request -> one decision):

1. **Validate the payload.** FastAPI does the structural check using
   :class:`TradingViewWebhookPayload`. Bad JSON / wrong types ->
   FastAPI returns 422 before the handler runs.
2. **Validate the secret.** When ``TRADINGVIEW_WEBHOOK_SECRET`` is set,
   the request must include either a matching ``secret`` field or an
   ``X-Webhook-Secret`` header. Mismatches return 401, never run the
   pipeline, and never reveal whether the secret was missing or wrong.
3. **Normalize the symbol.** TradingView symbols often arrive with the
   ``"1!"`` suffix (front-month continuous), or as ``"BINANCE:BTCUSDT"``
   for crypto. We strip the suffix / exchange prefix and uppercase the
   result. The normalized symbol must be in the operator-configured
   :class:`SymbolUniverse` (i.e. ``ENABLED_SYMBOLS``); otherwise we
   reject without trading.
4. **Build a Setup adapter.** Webhook signals don't carry a feature
   snapshot, so we synthesize a ``Setup`` whose ``features`` dict is
   the canonical zero vector. This keeps the existing risk engine + DB
   writers happy without a parallel "webhook risk" path.
5. **Optional model gate.** The endpoint can carry an injected
   :class:`Predictor`. Because zero-feature scoring is meaningless, the
   default factory does NOT load one — operators have to pass
   ``predictor=...`` explicitly to opt in.
6. **Risk engine.** Authoritative. Kill switch, trading window, daily
   trade caps, per-trade risk cap all apply just like internal signals.
   Blocks are persisted to ``risk_blocks``.
7. **Paper executor.** When approved, sizing + paper submit happens
   exactly as in the paper loop. The endpoint NEVER touches a live
   broker — even when ``MODE=LIVE`` (live mode is locked behind a
   separate flag and the live executor is still a placeholder).

The endpoint always responds quickly: no model training, no backtests,
no historical CSV scans inside the request. Heavy work belongs in the
scheduler / paper loop.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app.logging_config import get_logger
from config.instruments import InstrumentSpec, SymbolUniverse, get_instrument
from config.settings import Settings
from execution.base import Order
from execution.paper_executor import KillSwitchActive, PaperExecutor
from features.feature_builder import FEATURE_COLUMNS
from models.predictor import Predictor
from risk.kill_switch import KillSwitch
from risk.position_sizing import size_position
from risk.risk_engine import RiskConfig, RiskDecision, evaluate
from storage.db import session_scope
from storage.tables import FeatureSnapshot as FeatureSnapshotRow
from storage.tables import RiskBlock as RiskBlockRow
from storage.tables import Setup as SetupRow
from strategies.base import Setup
from webhooks.schemas import (
    NormalizedWebhookOrder,
    TradingViewWebhookPayload,
    WebhookResponse,
)


_log = get_logger("webhooks.tradingview")


# ---------------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------------
_FRONT_MONTH_RE = re.compile(r"^([A-Z0-9]+?)\d*!$")
_CRYPTO_SUFFIXES = ("USDT", "USDC", "USD", "BUSD", "FDUSD")


def normalize_symbol(raw: str, *, market_type: str = "futures") -> str:
    """Map a TradingView-style ticker to the bot's internal symbol.

    Examples (futures front-month continuous):

    - ``"MNQ1!"`` -> ``"MNQ"``
    - ``"MES1!"`` -> ``"MES"``
    - ``"NQ1!"``  -> ``"NQ"``
    - ``"ES1!"``  -> ``"ES"``

    Examples (crypto, when ``market_type="crypto"``):

    - ``"BTCUSDT"``        -> ``"BTC"``
    - ``"BINANCE:BTCUSD"`` -> ``"BTC"``
    - ``"ETH/USDT"``       -> ``"ETH"``

    Already-normalized symbols pass through untouched. The function does
    NOT verify the result is in the universe — that's the caller's job.
    """
    if not raw:
        raise ValueError("symbol is empty after normalization")
    s = raw.strip().upper()
    # Strip exchange prefix used by some TradingView feeds, e.g. "BINANCE:BTCUSDT".
    if ":" in s:
        s = s.split(":", 1)[1]
    # Strip any "/" used by spot pairs e.g. "ETH/USDT".
    if "/" in s:
        s = s.replace("/", "")

    if market_type == "crypto":
        for suffix in _CRYPTO_SUFFIXES:
            if s.endswith(suffix) and len(s) > len(suffix):
                return s[: -len(suffix)]
        return s

    # Futures: TradingView's front-month continuous ticker is "<root><digits>!".
    m = _FRONT_MONTH_RE.match(s)
    if m:
        return m.group(1)
    # Strip a trailing "!" defensively (covers e.g. "MNQ!" without digits).
    if s.endswith("!"):
        s = s.rstrip("!")
    return s


# ---------------------------------------------------------------------------
# Setup adapter
# ---------------------------------------------------------------------------
def _zero_features() -> dict[str, float]:
    """Canonical-shape feature dict with all zeros.

    Webhook signals don't carry features. The ``Setup`` validator only
    cares about the *key set* matching ``FEATURE_COLUMNS`` — the values
    don't have to be meaningful for the risk engine, only for a
    Predictor (which is opt-in for webhooks).
    """
    return {col: 0.0 for col in FEATURE_COLUMNS}


def _build_setup(
    order: NormalizedWebhookOrder,
    *,
    spec: InstrumentSpec,
    settings: Settings,
) -> Setup:
    """Convert a normalized webhook order into the repo's :class:`Setup`.

    Stop / target are derived from instrument tick size when the
    payload omits them. Direction "close" is handled separately and
    must not reach this function.
    """
    if order.direction == "close":  # pragma: no cover - guarded by caller
        raise ValueError("close orders cannot be turned into entry Setups")

    default_stop_dist = settings.WEBHOOK_DEFAULT_STOP_TICKS * spec.tick_size
    default_target_dist = settings.WEBHOOK_DEFAULT_TARGET_TICKS * spec.tick_size

    if order.direction == "long":
        stop = order.stop_price if order.stop_price is not None else order.price - default_stop_dist
        target = (
            order.target_price
            if order.target_price is not None
            else order.price + default_target_dist
        )
        if not (stop < order.price < target):
            raise ValueError(
                f"long webhook requires stop < price < target "
                f"(stop={stop}, price={order.price}, target={target})"
            )
    else:  # short
        stop = order.stop_price if order.stop_price is not None else order.price + default_stop_dist
        target = (
            order.target_price
            if order.target_price is not None
            else order.price - default_target_dist
        )
        if not (target < order.price < stop):
            raise ValueError(
                f"short webhook requires target < price < stop "
                f"(target={target}, price={order.price}, stop={stop})"
            )

    # ``atr_at_entry`` must be > 0; we don't have a real ATR so use the
    # stop distance as a benign placeholder. Position sizing reads
    # entry-stop distance directly, so this value is not on a hot path.
    atr_placeholder = max(abs(order.price - stop), spec.tick_size)

    return Setup(
        id=str(uuid.uuid4()),
        instrument=order.symbol,
        timestamp=order.received_at,
        strategy_name=f"webhook:{order.strategy or 'tradingview'}",
        direction=order.direction,
        entry_price=float(order.price),
        stop_price=float(stop),
        target_price=float(target),
        atr_at_entry=float(atr_placeholder),
        features=_zero_features(),
        bar_index=0,
    )


# ---------------------------------------------------------------------------
# Processor — owns the per-request pipeline
# ---------------------------------------------------------------------------
@dataclass
class WebhookProcessor:
    """Stateful pipeline shared by every request.

    Holds long-lived dependencies (settings, universe, executor,
    notifier, optional predictor). One instance per FastAPI app.
    Dependencies are passed in by :func:`build_router` so tests can
    inject stubs without touching settings or the DB.
    """

    settings: Settings
    universe: SymbolUniverse
    executor: PaperExecutor
    notifier_notify: Callable[..., None]
    predictor: Optional[Predictor] = None
    kill_switch: Optional[KillSwitch] = None
    high_risk_news_fn: Optional[Callable[[], bool]] = None

    def __post_init__(self) -> None:
        self.kill_switch = self.kill_switch or KillSwitch()
        self._risk_config = RiskConfig.from_settings(self.settings)

    # ---- helpers --------------------------------------------------
    def _safe_notify(self, kind: str, /, **payload) -> None:
        """``notifier.notify`` is contracted to never raise, but defend anyway."""
        try:
            self.notifier_notify(kind, **payload)
        except Exception as e:  # noqa: BLE001
            _log.warning("webhook.notify_failed", kind=kind, error=str(e))

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    # ---- entry point ----------------------------------------------
    def process(
        self,
        payload: TradingViewWebhookPayload,
        *,
        secret_header: Optional[str],
    ) -> WebhookResponse:
        """Run the full pipeline. Always returns a :class:`WebhookResponse`.

        Raises :class:`HTTPException` only for the secret check (so
        FastAPI emits a 401 with no trade-side processing).
        """
        # 1. Notify "received" first so the audit trail captures even
        #    payloads that fail downstream validation.
        self._safe_notify(
            "webhook.received",
            source=payload.source,
            symbol=payload.symbol,
            action=payload.action,
            strategy=payload.strategy or "",
            timeframe=payload.timeframe or "",
        )

        # 2. Secret check.
        self._check_secret(payload=payload, header=secret_header)

        # 3. Symbol normalization + universe gate.
        try:
            normalized = self._normalize_or_reject(payload)
        except ValueError as e:
            return self._reject(str(e), symbol=payload.symbol)

        # 4. action=close branch — flat any open position regardless of risk.
        if normalized.direction == "close":
            return self._handle_close(normalized)

        # 5. Build Setup adapter.
        spec = get_instrument(normalized.symbol)
        try:
            setup = _build_setup(normalized, spec=spec, settings=self.settings)
        except ValueError as e:
            return self._reject(
                f"Could not build Setup from webhook: {e}",
                symbol=normalized.symbol,
                direction=normalized.direction,
            )

        # 6. Persist the setup so the audit trail mirrors internally-
        #    detected setups. Failures are logged but never abort the
        #    pipeline (the trade is independently persisted by the
        #    executor on submit).
        self._persist_setup(setup)

        # 7. Optional model gate. Off by default for webhook signals.
        if self.predictor is not None:
            try:
                pred = self.predictor.predict_setup(setup)
            except Exception as e:  # noqa: BLE001
                _log.warning("webhook.predictor_failed", error=str(e))
                pred = None
            if pred is not None and not pred.approved:
                self._safe_notify(
                    "webhook.blocked",
                    symbol=normalized.symbol,
                    direction=normalized.direction,
                    rule="model_gate",
                    reason=f"probability={pred.probability:.4f} < threshold={pred.threshold:.4f}",
                )
                return WebhookResponse(
                    status="blocked",
                    symbol=normalized.symbol,
                    direction=normalized.direction,
                    reason="model_gate",
                    detail={
                        "probability": float(pred.probability),
                        "threshold": float(pred.threshold),
                    },
                )

        # 8. Risk engine.
        decision = self._evaluate_risk(setup)
        if not decision.allowed:
            self._persist_risk_block(setup, decision)
            self._safe_notify(
                "webhook.blocked",
                symbol=normalized.symbol,
                direction=normalized.direction,
                rule=decision.rule,
                reason=decision.reason,
            )
            return WebhookResponse(
                status="blocked",
                symbol=normalized.symbol,
                direction=normalized.direction,
                reason=decision.rule,
                detail={"engine_reason": decision.reason},
            )

        # 9. Sizing + paper submit.
        return self._submit_paper_trade(setup=setup, normalized=normalized, spec=spec)

    # ---- pipeline steps ------------------------------------------
    def _check_secret(
        self,
        *,
        payload: TradingViewWebhookPayload,
        header: Optional[str],
    ) -> None:
        configured = (
            self.settings.TRADINGVIEW_WEBHOOK_SECRET.get_secret_value()
            if self.settings.TRADINGVIEW_WEBHOOK_SECRET is not None
            else None
        )
        if not configured:
            return  # no secret configured -> accept (with a logged note)

        provided = (payload.secret or header or "").strip()
        if provided != configured:
            _log.warning(
                "webhook.bad_secret",
                source=payload.source,
                symbol=payload.symbol,
                has_field_secret=payload.secret is not None,
                has_header_secret=header is not None,
            )
            self._safe_notify(
                "webhook.invalid",
                reason="bad_secret",
                source=payload.source,
                symbol=payload.symbol,
            )
            # 401 by spec; we never reveal which side mismatched.
            raise HTTPException(status_code=401, detail="invalid webhook secret")

    def _normalize_or_reject(
        self, payload: TradingViewWebhookPayload
    ) -> NormalizedWebhookOrder:
        try:
            symbol = normalize_symbol(
                payload.symbol, market_type=self.settings.MARKET_TYPE
            )
        except ValueError as e:
            raise ValueError(f"could not normalize symbol {payload.symbol!r}: {e}") from e

        if symbol not in self.universe:
            self._safe_notify(
                "webhook.invalid",
                reason="symbol_not_in_universe",
                raw_symbol=payload.symbol,
                normalized=symbol,
                enabled=self.universe.as_list(),
            )
            raise ValueError(
                f"Symbol {symbol!r} (from {payload.symbol!r}) is not in "
                f"ENABLED_SYMBOLS={self.universe.as_list()}"
            )

        action = payload.action.lower()
        if action in ("buy", "long"):
            direction: str = "long"
        elif action in ("sell", "short"):
            direction = "short"
        elif action == "close":
            direction = "close"
        else:  # pragma: no cover - guarded by Pydantic Literal
            raise ValueError(f"unknown action {payload.action!r}")

        stop_val = float(payload.stop) if payload.stop is not None else None
        target_val = float(payload.target) if payload.target is not None else None

        return NormalizedWebhookOrder(
            raw_symbol=payload.symbol,
            symbol=symbol,
            direction=direction,  # type: ignore[arg-type]
            price=float(payload.price),
            stop_price=stop_val,
            target_price=target_val,
            strategy=payload.strategy or "tradingview",
            timeframe=payload.timeframe,
            source=payload.source,
            received_at=self._now(),
            raw_time=payload.time,
        )

    def _handle_close(self, order: NormalizedWebhookOrder) -> WebhookResponse:
        portfolio = self.executor.portfolio
        if portfolio.open_position is None:
            self._safe_notify(
                "webhook.received",
                kind="close_noop",
                symbol=order.symbol,
            )
            return WebhookResponse(
                status="noop",
                symbol=order.symbol,
                direction="close",
                reason="no_open_position",
            )

        if portfolio.open_position.instrument.upper() != order.symbol.upper():
            return self._reject(
                f"close requested for {order.symbol} but open position is "
                f"{portfolio.open_position.instrument}",
                symbol=order.symbol,
                direction="close",
            )

        try:
            record = self.executor.close_position(
                ts=self._now(),
                exit_raw_price=float(order.price),
                exit_reason="webhook_close",
            )
        except Exception as e:  # noqa: BLE001
            _log.error("webhook.close_failed", error=str(e))
            return self._reject(
                f"executor close failed: {e}",
                symbol=order.symbol,
                direction="close",
            )

        self._safe_notify(
            "webhook.closed",
            symbol=record.instrument,
            direction=record.direction,
            net_pnl=round(float(record.net_pnl), 2),
        )
        return WebhookResponse(
            status="closed",
            symbol=order.symbol,
            direction="close",
            reason="webhook_close",
            detail={
                "net_pnl": float(record.net_pnl),
                "exit_price": float(record.exit_price),
            },
        )

    def _evaluate_risk(self, setup: Setup) -> RiskDecision:
        try:
            high_risk_news = bool(
                self.high_risk_news_fn() if self.high_risk_news_fn else False
            )
        except Exception as e:  # noqa: BLE001
            _log.warning("webhook.news_flag_failed", error=str(e))
            high_risk_news = False

        return evaluate(
            setup,
            self.executor.portfolio,
            self._risk_config,
            self._now(),
            kill_switch_tripped=self.kill_switch.is_tripped(),
            high_risk_news_window=high_risk_news,
            instrument_spec=get_instrument(setup.instrument),
        )

    def _submit_paper_trade(
        self,
        *,
        setup: Setup,
        normalized: NormalizedWebhookOrder,
        spec: InstrumentSpec,
    ) -> WebhookResponse:
        try:
            sizing = size_position(
                entry_price=setup.entry_price,
                stop_price=setup.stop_price,
                instrument=spec,
                risk_per_trade=self.settings.RISK_PER_TRADE,
                max_position_size=self.settings.MAX_POSITION_SIZE,
            )
        except Exception as e:  # noqa: BLE001
            _log.error("webhook.sizing_failed", error=str(e))
            return self._reject(
                f"sizing failed: {e}",
                symbol=setup.instrument,
                direction=setup.direction,
            )

        order = Order(
            instrument=setup.instrument,
            direction=setup.direction,
            quantity=float(sizing.quantity),
            entry_price=float(setup.entry_price),
            stop_price=float(setup.stop_price),
            target_price=float(setup.target_price),
            setup_id=setup.id,
        )

        # "Approved" notification fires before submit so the audit trail
        # records approval even if the executor errors out.
        self._safe_notify(
            "webhook.approved",
            symbol=setup.instrument,
            direction=setup.direction,
            entry_price=round(float(setup.entry_price), 4),
            stop=round(float(setup.stop_price), 4),
            target=round(float(setup.target_price), 4),
            quantity=float(order.quantity),
        )

        try:
            fill = self.executor.submit(order)
        except KillSwitchActive:
            return self._reject(
                "executor refused: kill switch tripped",
                symbol=setup.instrument,
                direction=setup.direction,
            )
        except Exception as e:  # noqa: BLE001
            _log.error("webhook.executor_failed", error=str(e))
            return self._reject(
                f"executor error: {e}",
                symbol=setup.instrument,
                direction=setup.direction,
            )

        self._safe_notify(
            "webhook.trade_opened",
            symbol=setup.instrument,
            direction=setup.direction,
            quantity=float(order.quantity),
            entry_price=round(float(fill.fill_price), 4),
            stop=round(float(setup.stop_price), 4),
            target=round(float(setup.target_price), 4),
            setup_id=setup.id,
            source=normalized.source,
        )

        return WebhookResponse(
            status="accepted",
            symbol=setup.instrument,
            direction=setup.direction,  # type: ignore[arg-type]
            reason=None,
            detail={
                "fill_price": float(fill.fill_price),
                "quantity": float(order.quantity),
                "stop_price": float(setup.stop_price),
                "target_price": float(setup.target_price),
                "setup_id": setup.id,
                "raw_symbol": normalized.raw_symbol,
            },
        )

    # ---- DB persistence helpers ---------------------------------
    def _persist_setup(self, setup: Setup) -> None:
        try:
            with session_scope() as session:
                snapshot = FeatureSnapshotRow(
                    instrument=setup.instrument,
                    ts=setup.timestamp,
                    features=dict(setup.features),
                )
                session.add(snapshot)
                session.flush()
                row = SetupRow(
                    id=setup.id,
                    instrument=setup.instrument,
                    strategy_name=setup.strategy_name,
                    direction=setup.direction,
                    ts=setup.timestamp,
                    entry_price=setup.entry_price,
                    stop_price=setup.stop_price,
                    target_price=setup.target_price,
                    atr_at_entry=setup.atr_at_entry,
                    feature_snapshot_id=snapshot.id,
                )
                session.add(row)
        except Exception as e:  # noqa: BLE001
            _log.warning("webhook.persist_setup_failed", error=str(e))

    def _persist_risk_block(self, setup: Setup, decision: RiskDecision) -> None:
        try:
            with session_scope() as session:
                row = RiskBlockRow(
                    setup_id=setup.id,
                    ts=self._now(),
                    rule=decision.rule,
                    reason=decision.reason,
                )
                session.add(row)
        except Exception as e:  # noqa: BLE001
            _log.warning("webhook.persist_risk_block_failed", error=str(e))

    # ---- response helper ----------------------------------------
    def _reject(
        self,
        reason: str,
        *,
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> WebhookResponse:
        self._safe_notify(
            "webhook.invalid",
            reason=reason,
            symbol=symbol,
            direction=direction,
        )
        return WebhookResponse(
            status="rejected",
            symbol=symbol,
            direction=direction if direction in ("long", "short", "close") else None,  # type: ignore[arg-type]
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Router builder
# ---------------------------------------------------------------------------
def build_router(processor: WebhookProcessor) -> APIRouter:
    """Create a FastAPI router bound to a configured :class:`WebhookProcessor`.

    Separating router construction from the FastAPI app makes it
    trivial to mount the endpoint inside a larger service or to test
    multiple configurations against the same code path.
    """
    router = APIRouter(prefix="/webhooks", tags=["webhooks"])

    @router.post("/tradingview", response_model=WebhookResponse)
    def receive_tradingview_webhook(
        payload: TradingViewWebhookPayload,
        request: Request,
        x_webhook_secret: Optional[str] = Header(default=None),
    ) -> WebhookResponse:  # pragma: no cover - FastAPI route wiring
        # We log the raw client to ease ngrok / Cloudflare debugging.
        client_host = request.client.host if request.client is not None else "?"
        _log.info(
            "webhook.request",
            path="/webhooks/tradingview",
            client=client_host,
            symbol=payload.symbol,
            action=payload.action,
        )
        try:
            return processor.process(payload, secret_header=x_webhook_secret)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 - never let the endpoint 500 silently
            _log.error("webhook.unhandled", error=str(e))
            return JSONResponse(
                status_code=500,
                content=WebhookResponse(
                    status="rejected",
                    reason="internal_error",
                    detail={"error": str(e)},
                ).model_dump(),
            )

    @router.get("/healthz")
    def healthz() -> dict:  # pragma: no cover - trivial
        return {"status": "ok"}

    return router


__all__ = [
    "WebhookProcessor",
    "build_router",
    "normalize_symbol",
]
