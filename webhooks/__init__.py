"""FastAPI integration for external webhook signal sources.

This package is the HTTP-facing complement to :mod:`webhook` (the
lightweight payload validator). It owns the FastAPI router, request
schemas, and the pipeline that turns an inbound webhook into a paper
trade attempt:

    POST /webhooks/tradingview  ->  validate -> normalize -> risk ->
                                     paper-executor (only).

Live broker execution is intentionally out of scope. Even when
``MODE=LIVE`` the endpoint never escapes the :class:`PaperExecutor` —
real-money execution requires a separate, not-yet-shipped adapter.

Two ways to run the endpoint:

1. Standalone — :func:`create_app` builds a ready-to-serve FastAPI app
   with its own portfolio + paper executor. Suitable for lightweight
   dev / ngrok testing.
2. Embedded — call :func:`build_webhook_router` from a larger service
   that already owns a :class:`PaperExecutor` (e.g. the future combined
   paper-loop + webhook process) so signals share state with internal
   scanning.
"""

from __future__ import annotations

from typing import Callable, Optional

from fastapi import FastAPI

from app.logging_config import configure_logging, get_logger
from backtesting.fills import make_fills_model
from backtesting.portfolio import Portfolio
from config.instruments import SymbolUniverse, get_instrument
from config.settings import Settings, get_settings
from execution.paper_executor import PaperExecutor
from models.predictor import Predictor
from notifications.notification_service import NotificationService
from risk.kill_switch import KillSwitch
from storage.db import init_db
from webhooks.schemas import (
    NormalizedWebhookOrder,
    TradingViewWebhookPayload,
    WebhookResponse,
)
from webhooks.tradingview import (
    WebhookProcessor,
    build_router,
    normalize_symbol,
)


def build_webhook_router(
    *,
    settings: Settings,
    universe: SymbolUniverse,
    executor: PaperExecutor,
    notifier_notify: Callable[..., None],
    predictor: Optional[Predictor] = None,
    kill_switch: Optional[KillSwitch] = None,
    high_risk_news_fn: Optional[Callable[[], bool]] = None,
):
    """Build the TradingView webhook router with explicit dependencies.

    Use this when mounting the endpoint inside a larger FastAPI app
    that already owns a :class:`PaperExecutor` and notifier (so
    webhook fills update the same in-memory portfolio + DB rows the
    paper loop sees).
    """
    processor = WebhookProcessor(
        settings=settings,
        universe=universe,
        executor=executor,
        notifier_notify=notifier_notify,
        predictor=predictor,
        kill_switch=kill_switch,
        high_risk_news_fn=high_risk_news_fn,
    )
    return build_router(processor)


def create_app(
    *,
    settings: Optional[Settings] = None,
    executor: Optional[PaperExecutor] = None,
    notifier: Optional[NotificationService] = None,
    universe: Optional[SymbolUniverse] = None,
    predictor: Optional[Predictor] = None,
    kill_switch: Optional[KillSwitch] = None,
    high_risk_news_fn: Optional[Callable[[], bool]] = None,
    init_database: bool = True,
) -> FastAPI:
    """Create a standalone FastAPI app exposing the TradingView webhook.

    Default dependencies (when not injected):

    - ``settings`` -> :func:`config.settings.get_settings`
    - ``universe`` -> :meth:`SymbolUniverse.from_settings`
    - ``executor`` -> a fresh :class:`PaperExecutor` whose portfolio is
      scoped to ``PRIMARY_SYMBOL`` (or ``INSTRUMENT``). When the bot is
      multi-symbol the operator should inject an executor that shares
      state with the paper loop's per-symbol books.
    - ``notifier`` -> :meth:`NotificationService.from_settings`.
    - ``predictor`` left ``None`` because webhook signals don't carry a
      feature snapshot — see ``webhooks/tradingview.py`` for details.

    The factory is intentionally explicit. Hidden globals would make
    cross-process state bugs (two PaperExecutors writing the same DB)
    almost impossible to debug.
    """
    settings = settings or get_settings()
    log = get_logger("webhooks.app")
    configure_logging(level=settings.LOG_LEVEL, json_format=settings.LOG_JSON)

    if init_database:
        init_db()

    if universe is None:
        universe = SymbolUniverse.from_settings(settings)

    if executor is None:
        primary = settings.PRIMARY_SYMBOL or settings.INSTRUMENT
        spec = get_instrument(primary)
        portfolio = Portfolio(instrument_spec=spec)
        fills = make_fills_model(
            primary,
            slippage_ticks=settings.SLIPPAGE_TICKS,
            commission_per_contract=settings.COMMISSION_PER_CONTRACT,
            crypto_slippage_bps=settings.CRYPTO_SLIPPAGE_BPS,
            crypto_fee_bps=settings.CRYPTO_FEE_BPS,
        )
        executor = PaperExecutor(
            portfolio=portfolio,
            fills_model=fills,
            kill_switch=kill_switch,
        )
        log.info(
            "webhooks.app.executor_built",
            primary_symbol=primary,
            note="Standalone executor — does NOT share state with paper loop.",
        )

    if notifier is None:
        notifier = NotificationService.from_settings(settings)

    app = FastAPI(
        title="Tradeify webhook receiver",
        version=settings.APP_VERSION,
        description=(
            "External webhook ingest for TradingView (and similar) alerts. "
            "Signals flow through the same risk engine as internal setups "
            "and are paper-traded only — no live broker execution."
        ),
    )

    router = build_webhook_router(
        settings=settings,
        universe=universe,
        executor=executor,
        notifier_notify=notifier.notify,
        predictor=predictor,
        kill_switch=kill_switch,
        high_risk_news_fn=high_risk_news_fn,
    )
    app.include_router(router)

    @app.get("/")
    def root() -> dict:  # pragma: no cover - trivial
        return {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "endpoints": ["/webhooks/tradingview", "/webhooks/healthz"],
        }

    log.info(
        "webhooks.app.ready",
        symbols=universe.as_list(),
        market_type=universe.market_type(),
        secret_required=bool(settings.TRADINGVIEW_WEBHOOK_SECRET),
    )
    return app


__all__ = [
    "NormalizedWebhookOrder",
    "TradingViewWebhookPayload",
    "WebhookProcessor",
    "WebhookResponse",
    "build_router",
    "build_webhook_router",
    "create_app",
    "normalize_symbol",
]
