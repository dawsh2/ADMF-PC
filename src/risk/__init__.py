"""Risk management module for ADMF-PC.

Architecture Reference: docs/SYSTEM_ARCHITECTURE_V5.MD#risk-module  
Style Guide: STYLE.md - Canonical risk implementations

This module provides canonical Risk management implementations:
- Portfolio state tracking
- Position sizing strategies  
- Risk constraint enforcement
- Signal processing pipeline

THE canonical implementations:
- PortfolioState: Global portfolio tracking
- Position sizers: Fixed, percentage, volatility-based strategies
- Risk limits: Position, exposure, drawdown constraints
- Signal processors: Signal to order conversion pipeline
"""

from .protocols import (
    RiskPortfolioProtocol,
    SignalType,
    OrderSide,
    PositionSizerProtocol,
    RiskLimitProtocol,
    PortfolioStateProtocol,
    SignalProcessorProtocol,
    Signal,
    Order,
    Position,
    RiskMetrics,
)
# RiskPortfolioContainer deprecated - use separate Risk and Portfolio containers
from .position_sizing import (
    FixedPositionSizer,
    PercentagePositionSizer,
    KellyCriterionSizer,
    VolatilityBasedSizer,
)
from .limits import (
    MaxPositionLimit,
    MaxExposureLimit,
    MaxDrawdownLimit,
    VaRLimit,
    ConcentrationLimit,
    LeverageLimit,
)
from .portfolio_state import PortfolioState
# Capabilities deprecated with RiskPortfolioContainer
# Use separate Risk and Portfolio containers instead

__all__ = [
    # Protocols
    "RiskPortfolioProtocol",
    "PositionSizerProtocol", 
    "RiskLimitProtocol",
    "PortfolioStateProtocol",
    "SignalProcessorProtocol",
    # Types
    "Signal",
    "SignalType", 
    "OrderSide",
    "Order",
    "Position",
    "RiskMetrics",
    # Position Sizers
    "FixedPositionSizer",
    "PercentagePositionSizer", 
    "KellyCriterionSizer",
    "VolatilityBasedSizer",
    # Risk Limits
    "MaxPositionLimit",
    "MaxExposureLimit",
    "MaxDrawdownLimit", 
    "VaRLimit",
    "ConcentrationLimit",
    "LeverageLimit",
    # Portfolio State
    "PortfolioState",
]