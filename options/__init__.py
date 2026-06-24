"""Options trading layer (paper / simulation only — no live execution)."""

from options.chain import (
    AlpacaChainProvider,
    BaseChainProvider,
    OptionsChainError,
    SyntheticChainProvider,
)
from options.contract import (
    OptionContract,
    build_occ_symbol,
    parse_occ_symbol,
)
from options.execution import (
    AlpacaOptionsExecutor,
    BaseOptionsExecutor,
    MockOptionsExecutor,
    OptionLeg,
    OptionOrderResult,
)
from options.greeks import Greeks, black_scholes_greeks
from options.position_manager import (
    ManagerConfig,
    OptionPosition,
    OptionsPositionManager,
)
from options.risk_rules import (
    OptionsRiskConfig,
    OptionsRiskEngine,
    RiskDecision,
)
from options.selection import (
    SelectionConfig,
    select_directional_contract,
    select_iron_condor,
    select_vertical_spread,
)
from options.trader import OptionsTrader

__all__ = [
    "AlpacaChainProvider",
    "AlpacaOptionsExecutor",
    "BaseChainProvider",
    "BaseOptionsExecutor",
    "Greeks",
    "ManagerConfig",
    "MockOptionsExecutor",
    "OptionContract",
    "OptionLeg",
    "OptionOrderResult",
    "OptionPosition",
    "OptionsChainError",
    "OptionsPositionManager",
    "OptionsRiskConfig",
    "OptionsRiskEngine",
    "OptionsTrader",
    "RiskDecision",
    "SelectionConfig",
    "SyntheticChainProvider",
    "black_scholes_greeks",
    "build_occ_symbol",
    "parse_occ_symbol",
    "select_directional_contract",
    "select_iron_condor",
    "select_vertical_spread",
]
