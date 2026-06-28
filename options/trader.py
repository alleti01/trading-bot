"""Options trader: VWAP/EMA underlying signal → options entry pipeline.

Pipeline (paper/sim only):
    signal(direction, underlying)
      → selection (atm_directional | vertical_spread | iron_condor)
      → risk gate (premium / DTE / open positions)
      → execution (Alpaca paper or mock)
      → position manager (tracks greeks, expiry, rolling)
      → Discord alert (tagged with underlying + strategy + reason)

The trader holds no live path. The broker router / settings enforce that
options execution only happens in PAPER (Alpaca paper) or DRY_RUN (mock).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.logging_config import get_logger
from config.settings import Settings
from notifications.notification_service import NotificationService
from notifications.trade_events import classify_result
from options.chain import BaseChainProvider, SyntheticChainProvider
from options.execution import (
    BaseOptionsExecutor,
    MockOptionsExecutor,
    OptionLeg,
)
from options.position_manager import ManagerConfig, OptionsPositionManager
from options.risk_rules import OptionsRiskConfig, OptionsRiskEngine
from options.selection import (
    SelectionConfig,
    select_directional_contract,
    select_iron_condor,
    select_vertical_spread,
)

_log = get_logger("options.trader")


class OptionsTrader:
    def __init__(
        self,
        settings: Settings,
        *,
        chain_provider: BaseChainProvider,
        executor: BaseOptionsExecutor,
        notifier: Optional[NotificationService] = None,
        position_manager: Optional[OptionsPositionManager] = None,
    ) -> None:
        self.settings = settings
        self.chain_provider = chain_provider
        self.executor = executor
        self.notifier = notifier
        self.log = _log
        self.selection_config = SelectionConfig(
            target_dte=settings.OPTIONS_DEFAULT_DTE,
            min_dte=settings.OPTIONS_MIN_DTE,
            max_dte=settings.OPTIONS_MAX_DTE,
            target_delta=settings.OPTIONS_TARGET_DELTA,
            spread_width_strikes=settings.OPTIONS_SPREAD_WIDTH_STRIKES,
        )
        self.risk = OptionsRiskEngine(
            OptionsRiskConfig(
                max_premium_per_trade=settings.OPTIONS_MAX_PREMIUM_PER_TRADE,
                max_open_positions=settings.OPTIONS_MAX_OPEN_POSITIONS,
                min_dte=settings.OPTIONS_MIN_DTE,
                max_dte=settings.OPTIONS_MAX_DTE,
            )
        )
        self.pm = position_manager or OptionsPositionManager(
            executor=executor,
            chain_provider=chain_provider,
            state_path=Path(settings.OPTIONS_STATE_PATH),
            config=ManagerConfig(
                auto_close_dte=settings.OPTIONS_AUTO_CLOSE_DTE,
                auto_roll=settings.OPTIONS_AUTO_ROLL,
                roll_dte_trigger=settings.OPTIONS_ROLL_DTE_TRIGGER,
                profit_target_pct=settings.OPTIONS_PROFIT_TARGET_PCT,
                stop_loss_pct=settings.OPTIONS_STOP_LOSS_PCT,
            ),
        )

    @classmethod
    def for_dry_run(cls, settings: Settings, *, now: Optional[datetime] = None) -> "OptionsTrader":
        """Build a fully in-memory trader (synthetic chain + mock executor)."""
        provider = SyntheticChainProvider(
            spot_by_symbol={
                "SPY": 550.0, "QQQ": 480.0, "AAPL": 210.0, "MSFT": 440.0,
            },
            now=now,
        )
        return cls(
            settings,
            chain_provider=provider,
            executor=MockOptionsExecutor(),
            notifier=None,
        )

    def _allowed_underlyings(self) -> set[str]:
        return {
            s.strip().upper()
            for s in self.settings.OPTIONS_ENABLED_UNDERLYINGS.split(",")
            if s.strip()
        }

    def _notify(self, kind: str, **payload: Any) -> None:
        if self.notifier is None:
            return
        try:
            self.notifier.notify(kind, **payload)
        except Exception as e:  # noqa: BLE001
            self.log.warning("options.notify_failed", error=str(e))

    def handle_signal(
        self,
        *,
        underlying: str,
        direction: str,
        now: Optional[datetime] = None,
        thesis: str = "",
    ) -> dict[str, Any]:
        """Convert a directional underlying signal into an options trade."""
        now = now or datetime.now(tz=timezone.utc)
        sym = underlying.upper()

        if not self.settings.OPTIONS_ENABLED:
            return {"status": "disabled", "reason": "OPTIONS_ENABLED=false"}
        if sym not in self._allowed_underlyings():
            return {"status": "skipped", "reason": f"{sym} not in OPTIONS_ENABLED_UNDERLYINGS"}

        strategy = self.settings.OPTIONS_STRATEGY
        qty = self.settings.OPTIONS_QTY_PER_TRADE
        try:
            if strategy == "iron_condor":
                return self._handle_iron_condor(sym, qty=qty, now=now, thesis=thesis)
            if strategy == "vertical_spread":
                return self._handle_vertical(sym, direction, qty=qty, now=now, thesis=thesis)
            return self._handle_directional(sym, direction, qty=qty, now=now, thesis=thesis)
        except Exception as e:  # noqa: BLE001
            self.log.error("options.handle_signal_failed", underlying=sym, error=str(e))
            self._notify(
                "system.error",
                source="options.trader",
                underlying=sym,
                strategy=strategy,
                action="entry",
                reason=str(e),
            )
            return {"status": "error", "reason": str(e)}

    def _handle_directional(
        self, underlying: str, direction: str, *, qty: int, now: datetime, thesis: str
    ) -> dict[str, Any]:
        contract = select_directional_contract(
            self.chain_provider,
            underlying=underlying,
            direction=direction,
            config=self.selection_config,
            now=now,
        )
        if contract is None:
            return {"status": "skipped", "reason": "no_contract_selected"}

        decision = self.risk.evaluate_entry(
            contract, qty=qty, open_positions=self.pm.open_count(), now=now
        )
        if not decision.approved:
            self._notify(
                "options.blocked",
                broker_provider=self.executor.provider_name,
                underlying=underlying,
                strategy="atm_directional",
                action="entry",
                symbol=contract.occ_symbol,
                reason=decision.reason,
            )
            return {"status": "blocked", "reason": decision.reason}

        pos = self.pm.open_position(contract, qty=qty, thesis=thesis, now=now)
        self._notify(
            "options.opened",
            broker_provider=self.executor.provider_name,
            underlying=underlying,
            strategy="atm_directional",
            action="buy_to_open",
            symbol=contract.occ_symbol,
            reason=thesis or "vwap_ema signal",
            premium=self.risk.estimate_premium(contract, qty=qty),
        )
        return {"status": "opened", "contract": contract.to_payload(), "position": pos.occ_symbol}

    def _handle_vertical(
        self, underlying: str, direction: str, *, qty: int, now: datetime, thesis: str
    ) -> dict[str, Any]:
        legs = select_vertical_spread(
            self.chain_provider,
            underlying=underlying,
            direction=direction,
            config=self.selection_config,
            now=now,
        )
        if legs is None:
            return {"status": "skipped", "reason": "no_spread_selected"}
        long_leg, short_leg = legs
        decision = self.risk.evaluate_entry(
            long_leg, qty=qty, open_positions=self.pm.open_count(), now=now
        )
        if not decision.approved:
            self._notify(
                "options.blocked",
                broker_provider=self.executor.provider_name,
                underlying=underlying,
                strategy="vertical_spread",
                action="entry",
                symbol=long_leg.occ_symbol,
                reason=decision.reason,
            )
            return {"status": "blocked", "reason": decision.reason}

        result = self.executor.place_multi_leg(
            underlying=underlying,
            legs=[
                OptionLeg(contract=long_leg, action="buy_to_open"),
                OptionLeg(contract=short_leg, action="sell_to_open"),
            ],
            qty=qty,
            order_class="spread",
        )
        self._notify(
            "options.opened",
            broker_provider=self.executor.provider_name,
            underlying=underlying,
            strategy="vertical_spread",
            action="buy_to_open",
            symbol=f"{long_leg.occ_symbol}/{short_leg.occ_symbol}",
            reason=thesis or "vwap_ema signal",
        )
        return {"status": "opened", "order": result.to_payload()}

    def _handle_iron_condor(
        self, underlying: str, *, qty: int, now: datetime, thesis: str
    ) -> dict[str, Any]:
        legs = select_iron_condor(
            self.chain_provider,
            underlying=underlying,
            config=self.selection_config,
            now=now,
        )
        if legs is None:
            return {"status": "skipped", "reason": "no_condor_selected"}
        order_legs = [
            OptionLeg(contract=legs["short_put"], action="sell_to_open"),
            OptionLeg(contract=legs["long_put"], action="buy_to_open"),
            OptionLeg(contract=legs["short_call"], action="sell_to_open"),
            OptionLeg(contract=legs["long_call"], action="buy_to_open"),
        ]
        result = self.executor.place_multi_leg(
            underlying=underlying,
            legs=order_legs,
            qty=qty,
            order_class="iron_condor",
        )
        self._notify(
            "options.opened",
            broker_provider=self.executor.provider_name,
            underlying=underlying,
            strategy="iron_condor",
            action="open",
            symbol=underlying,
            reason=thesis or "neutral premium",
        )
        return {"status": "opened", "order": result.to_payload()}

    def manage(self, *, now: Optional[datetime] = None) -> list[dict[str, Any]]:
        actions = self.pm.manage_cycle(now=now)
        for action in actions:
            detail = action.get("detail") or {}
            is_close = action.get("action") == "close" and isinstance(detail, dict)
            if is_close and detail.get("realized_pnl") is not None:
                self._notify_options_closed(action, detail)
            else:
                self._notify(
                    "options.managed",
                    broker_provider=self.executor.provider_name,
                    underlying=detail.get("underlying", "") if isinstance(detail, dict) else "",
                    strategy=self.settings.OPTIONS_STRATEGY,
                    action=action.get("action", "manage"),
                    symbol=action.get("occ", ""),
                    reason=action.get("reason", "manage_cycle"),
                )
        return actions

    def _notify_options_closed(self, action: dict[str, Any], detail: dict[str, Any]) -> None:
        """Emit a rich close alert reporting realised dollar PnL.

        Mirrors the futures ``trade.closed`` layout so the Discord message
        clearly says whether the option made or lost money and how much.
        """
        net = float(detail["realized_pnl"])
        self._notify(
            "options.closed",
            broker_provider=self.executor.provider_name,
            symbol=detail.get("underlying", "") or action.get("occ", ""),
            direction=detail.get("option_type", ""),
            result=classify_result(net),
            exit_reason=action.get("reason", "manage_cycle"),
            net_pnl=net,
            return_pct=detail.get("realized_pnl_pct"),
            entry_price=detail.get("entry_price"),
            exit_price=detail.get("exit_price"),
            quantity=detail.get("qty"),
            contract=action.get("occ", ""),
            strategy=self.settings.OPTIONS_STRATEGY,
        )
