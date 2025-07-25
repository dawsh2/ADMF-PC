"""Risk management module for ADMF-PC.

Architecture Reference: docs/SYSTEM_ARCHITECTURE_V5.MD#risk-module  
Style Guide: STYLE.md - Canonical risk implementations

This module provides canonical Risk management implementations:
- Position sizing strategies  
- Risk constraint enforcement
- Signal processing pipeline
- Risk limit validation

THE canonical implementations:
- Position sizers: Fixed, percentage, volatility-based strategies
- Risk limits: Position, exposure, drawdown constraints
- Signal processors: Signal to order conversion pipeline
- Risk validators: Stateless risk validation components

Note: Portfolio state tracking has been moved to the portfolio module.
"""

from .protocols import (
    RiskPortfolioProtocol,
    SignalType,
    OrderSide,
    PositionSizerProtocol,
    RiskLimitProtocol,
    SignalProcessorProtocol,
    StatelessRiskValidator,
    RiskCapability,
    PositionSizingCapability,
    RiskLimitCapability,
)
# Import portfolio types from portfolio module
from ..portfolio.protocols import (
    Position,
    RiskMetrics,
    PortfolioStateProtocol,
)
# Import canonical types from their proper modules
from ..core.events.types import Event
from ..execution.types import OrderType, OrderSide
from ..strategy.types import SignalType, Signal
from .position_sizing import (
    # Pure functional position size calculators
    calculate_fixed_position_size,
    calculate_percentage_position_size,
    calculate_kelly_position_size,
    calculate_volatility_based_position_size,
    calculate_atr_based_position_size,
    apply_position_constraints,
    # Backward compatibility wrappers
    FixedPositionSizer,
    PercentagePositionSizer,
    KellyCriterionSizer,
    VolatilityBasedSizer,
)
from .limits import (
    # Pure functional limit checks
    check_max_position_limit,
    check_max_drawdown_limit,
    check_var_limit,
    check_max_exposure_limit,
    check_concentration_limit,
    check_leverage_limit,
    check_daily_loss_limit,
    check_symbol_restrictions,
    check_all_limits,
    # Backward compatibility
    RiskLimits,
)

__all__ = [
    # Protocols
    "RiskPortfolioProtocol",
    "PositionSizerProtocol", 
    "RiskLimitProtocol",
    "PortfolioStateProtocol",
    "SignalProcessorProtocol",
    "StatelessRiskValidator",
    # Capabilities
    "RiskCapability",
    "PositionSizingCapability", 
    "RiskLimitCapability",
    # Types (canonical from other modules)
    "Event",
    "Signal",
    "SignalType", 
    "OrderSide",
    "OrderType",
    "Position",
    "RiskMetrics",
    # Position Sizers
    "FixedPositionSizer",
    "PercentagePositionSizer", 
    "KellyCriterionSizer",
    "VolatilityBasedSizer",
    # Pure functional calculators
    "calculate_fixed_position_size",
    "calculate_percentage_position_size",
    "calculate_kelly_position_size",
    "calculate_volatility_based_position_size",
    "calculate_atr_based_position_size",
    "apply_position_constraints",
    # Pure functional limit checks
    "check_max_position_limit",
    "check_max_drawdown_limit",
    "check_var_limit",
    "check_max_exposure_limit",
    "check_concentration_limit",
    "check_leverage_limit",
    "check_daily_loss_limit",
    "check_symbol_restrictions",
    "check_all_limits",
    # Backward compatibility classes
    "RiskLimits",
]