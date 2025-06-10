"""Portfolio management module for ADMF-PC.

Architecture Reference: docs/SYSTEM_ARCHITECTURE_V5.MD#portfolio-module  
Style Guide: STYLE.md - Canonical portfolio implementations

This module provides canonical Portfolio management implementations:
- Portfolio state tracking and management
- Performance metrics calculation
- Position management
- Risk metrics computation

THE canonical implementations:
- PortfolioState: Portfolio state tracking and position management
- Position: Individual position tracking with P&L calculation
- RiskMetrics: Portfolio-wide risk and performance metrics
"""

from .protocols import (
    PortfolioStateProtocol,
    PortfolioManagerProtocol,
    PortfolioTrackingCapability,
    PerformanceAnalysisCapability,
    Position,
    RiskMetrics,
)
# Import canonical types from their proper modules
from ..core.events.types import Event
from ..execution.types import OrderType, OrderSide
from ..strategy.types import SignalType, Signal
from .state import PortfolioState

__all__ = [
    # Protocols
    "PortfolioStateProtocol",
    "PortfolioManagerProtocol",
    # Capabilities
    "PortfolioTrackingCapability",
    "PerformanceAnalysisCapability",
    # Types (canonical from other modules)
    "Event",
    "Signal",
    "Position",
    "RiskMetrics",
    "OrderType",
    "OrderSide", 
    "SignalType",
    # Implementations
    "PortfolioState",
]