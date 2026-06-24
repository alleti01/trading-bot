"""Options position manager: track positions, expiry, rolling, greeks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.logging_config import get_logger
from options.chain import BaseChainProvider
from options.contract import OptionContract, parse_occ_symbol
from options.execution import BaseOptionsExecutor
from options.selection import SelectionConfig, select_directional_contract

_log = get_logger("options.position_manager")


@dataclass
class OptionPosition:
    occ_symbol: str
    underlying: str
    option_type: str
    strike: float
    expiry: str  # ISO date
    qty: int
    entry_price: float
    opened_at: str
    thesis: str = ""
    current_price: Optional[float] = None
    delta: Optional[float] = None
    theta: Optional[float] = None

    def days_to_expiry(self, *, now: Optional[datetime] = None) -> int:
        ref = (now or datetime.now(tz=timezone.utc)).date()
        return (date.fromisoformat(self.expiry) - ref).days

    def unrealized_pnl(self) -> Optional[float]:
        if self.current_price is None:
            return None
        return round((self.current_price - self.entry_price) * 100.0 * self.qty, 2)

    def unrealized_pnl_pct(self) -> Optional[float]:
        if self.current_price is None or self.entry_price == 0:
            return None
        return round(
            (self.current_price - self.entry_price) / self.entry_price * 100.0, 2
        )


@dataclass
class ManagerConfig:
    auto_close_dte: int = 5
    auto_roll: bool = True
    roll_dte_trigger: int = 7
    profit_target_pct: float = 50.0
    stop_loss_pct: float = -50.0


class OptionsPositionManager:
    """Tracks open option positions and decides expiry/roll/close actions.

    State persists to a JSON file so it survives restarts and stays
    isolated per evaluation track (the path is supplied by the caller).
    """

    def __init__(
        self,
        *,
        executor: BaseOptionsExecutor,
        chain_provider: Optional[BaseChainProvider] = None,
        state_path: Optional[Path] = None,
        config: Optional[ManagerConfig] = None,
    ) -> None:
        self.executor = executor
        self.chain_provider = chain_provider
        self.state_path = Path(state_path) if state_path else None
        self.config = config or ManagerConfig()
        self.positions: dict[str, OptionPosition] = {}
        self.log = _log
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.state_path and self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                for occ, pos in data.get("positions", {}).items():
                    self.positions[occ] = OptionPosition(**pos)
            except Exception as e:  # noqa: BLE001
                self.log.warning("options.pm.load_failed", error=str(e))

    def save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "positions": {occ: asdict(p) for occ, p in self.positions.items()},
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self.state_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------
    def open_position(
        self,
        contract: OptionContract,
        *,
        qty: int,
        thesis: str = "",
        now: Optional[datetime] = None,
    ) -> OptionPosition:
        now = now or datetime.now(tz=timezone.utc)
        result = self.executor.place_single_leg(
            contract=contract, action="buy_to_open", qty=qty
        )
        price = result.limit_price or contract.mid_price or 0.0
        pos = OptionPosition(
            occ_symbol=contract.occ_symbol,
            underlying=contract.underlying,
            option_type=contract.option_type,
            strike=contract.strike,
            expiry=contract.expiry.isoformat(),
            qty=qty,
            entry_price=float(price),
            opened_at=now.isoformat(),
            thesis=thesis,
            current_price=contract.mid_price,
            delta=contract.delta,
            theta=contract.theta,
        )
        self.positions[contract.occ_symbol] = pos
        self.save()
        self.log.info(
            "options.pm.opened",
            occ=contract.occ_symbol,
            qty=qty,
            entry_price=price,
            order_id=result.order_id,
        )
        return pos

    def close_position(
        self, occ_symbol: str, *, reason: str = "manual"
    ) -> Optional[dict[str, Any]]:
        pos = self.positions.get(occ_symbol)
        if pos is None:
            return None
        contract = parse_occ_symbol(occ_symbol)
        result = self.executor.close_contract(contract=contract, qty=pos.qty)
        self.positions.pop(occ_symbol, None)
        self.save()
        self.log.info(
            "options.pm.closed",
            occ=occ_symbol,
            reason=reason,
            order_id=result.order_id,
        )
        return {"reason": reason, "order": result.to_payload()}

    # ------------------------------------------------------------------
    # Management cycle (called by midday / scheduler workflows)
    # ------------------------------------------------------------------
    def manage_cycle(self, *, now: Optional[datetime] = None) -> list[dict[str, Any]]:
        now = now or datetime.now(tz=timezone.utc)
        actions: list[dict[str, Any]] = []
        for occ in list(self.positions.keys()):
            pos = self.positions[occ]
            self._refresh_price(pos, now=now)
            dte = pos.days_to_expiry(now=now)
            pnl_pct = pos.unrealized_pnl_pct()

            # Profit target / stop loss.
            if pnl_pct is not None and pnl_pct >= self.config.profit_target_pct:
                closed = self.close_position(occ, reason="profit_target")
                actions.append({"occ": occ, "action": "close", "reason": "profit_target", "detail": closed})
                continue
            if pnl_pct is not None and pnl_pct <= self.config.stop_loss_pct:
                closed = self.close_position(occ, reason="stop_loss")
                actions.append({"occ": occ, "action": "close", "reason": "stop_loss", "detail": closed})
                continue

            # Expiry management.
            if dte <= self.config.auto_close_dte:
                if self.config.auto_roll and self.chain_provider is not None:
                    rolled = self._roll(pos, now=now)
                    actions.append({"occ": occ, "action": "roll", "detail": rolled})
                else:
                    closed = self.close_position(occ, reason="expiry_close")
                    actions.append({"occ": occ, "action": "close", "reason": "expiry_close", "detail": closed})
                continue

            if dte <= self.config.roll_dte_trigger and self.config.auto_roll and self.chain_provider is not None:
                rolled = self._roll(pos, now=now)
                actions.append({"occ": occ, "action": "roll", "reason": "roll_trigger", "detail": rolled})

        self.save()
        return actions

    def _refresh_price(self, pos: OptionPosition, *, now: datetime) -> None:
        if self.chain_provider is None:
            return
        try:
            chain = self.chain_provider.get_chain(
                pos.underlying,
                expiry=date.fromisoformat(pos.expiry),
                option_type=pos.option_type,  # type: ignore[arg-type]
            )
        except Exception:  # noqa: BLE001
            return
        match = next((c for c in chain if c.occ_symbol == pos.occ_symbol), None)
        if match is not None:
            pos.current_price = match.mid_price
            pos.delta = match.delta
            pos.theta = match.theta

    def _roll(self, pos: OptionPosition, *, now: datetime) -> Optional[dict[str, Any]]:
        if self.chain_provider is None:
            return None
        # Close the expiring leg, open a fresh one in the same direction.
        direction = "long" if pos.option_type == "call" else "short"
        self.close_position(pos.occ_symbol, reason="roll_close")
        new_contract = select_directional_contract(
            self.chain_provider,
            underlying=pos.underlying,
            direction=direction,
            config=SelectionConfig(),
            now=now,
        )
        if new_contract is None:
            return {"rolled": False, "reason": "no_replacement_contract"}
        new_pos = self.open_position(
            new_contract, qty=pos.qty, thesis=f"roll of {pos.occ_symbol}", now=now
        )
        return {"rolled": True, "new_occ": new_pos.occ_symbol}

    def open_count(self) -> int:
        return len(self.positions)

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                **asdict(p),
                "days_to_expiry": p.days_to_expiry(),
                "unrealized_pnl": p.unrealized_pnl(),
                "unrealized_pnl_pct": p.unrealized_pnl_pct(),
            }
            for p in self.positions.values()
        ]
