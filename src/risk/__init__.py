"""Risk management module for ADMF-PC.

This module provides a unified Risk & Portfolio management system that:
- Manages multiple strategy components
- Converts signals to orders (with veto capability)
- Tracks portfolio state globally
- Implements position sizing strategies
- Enforces risk limits

Key components:
- RiskPortfolioContainer: Unified risk and portfolio management
- PositionSizer: Position sizing strategies
- RiskLimit: Risk limit implementations
- PortfolioState: Global portfolio tracking
- SignalProcessor: Signal to order pipeline
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
from .risk_portfolio import RiskPortfolioContainer
from .position_sizing import (
    FixedPositionSizer,
    PercentagePositionSizer,
    KellyCriterionSizer,
    VolatilityBasedSizer,
)
from .risk_limits import (
    MaxPositionLimit,
    MaxExposureLimit,
    MaxDrawdownLimit,
    VaRLimit,
    ConcentrationLimit,
    LeverageLimit,
)
from .portfolio_state import PortfolioState
from .signal_processing import SignalProcessor, SignalAggregator
from .signal_advanced import (
    SignalRouter,
    SignalValidator,
    RiskAdjustedSignalProcessor,
    SignalCache,
    SignalPrioritizer,
)
from .signal_flow import (
    SignalFlowManager,
    MultiSymbolSignalFlow,
)
from .capabilities import (
    RiskPortfolioCapability,
    ThreadSafeRiskPortfolioCapability
)

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
    # Implementations
    "RiskPortfolioContainer",
    "FixedPositionSizer",
    "PercentagePositionSizer",
    "KellyCriterionSizer",
    "VolatilityBasedSizer",
    "MaxPositionLimit",
    "MaxExposureLimit",
    "MaxDrawdownLimit",
    "VaRLimit",
    "ConcentrationLimit",
    "LeverageLimit",
    "PortfolioState",
    "SignalProcessor",
    "SignalAggregator",
    # Advanced Signal Processing
    "SignalRouter",
    "SignalValidator",
    "RiskAdjustedSignalProcessor",
    "SignalCache",
    "SignalPrioritizer",
    # Signal Flow
    "SignalFlowManager",
    "MultiSymbolSignalFlow",
    # Capabilities
    "RiskPortfolioCapability",
    "ThreadSafeRiskPortfolioCapability",
]