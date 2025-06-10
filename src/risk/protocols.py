"""Protocol definitions for risk management components."""

from abc import abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Protocol, Optional, Dict, List, Any, Set, runtime_checkable
from enum import Enum

from ..core.components.protocols import Component, Capability
from ..execution.types import OrderType, OrderSide
from ..strategy.types import SignalType


class PositionSizerProtocol(Protocol):
    """Protocol for position sizing strategies."""
    
    @abstractmethod
    def calculate_size(
        self,
        signal: Signal,
        portfolio_state: "PortfolioStateProtocol",
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Position size (quantity)
        """
        ...


class RiskLimitProtocol(Protocol):
    """Protocol for risk limit checks."""
    
    @abstractmethod
    def check_limit(
        self,
        order: Order,
        portfolio_state: "PortfolioStateProtocol",
        market_data: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Check if order violates risk limit.
        
        Args:
            order: Proposed order
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Tuple of (passes_check, reason_if_failed)
        """
        ...
    
    @abstractmethod
    def get_limit_info(self) -> Dict[str, Any]:
        """Get information about this limit."""
        ...


# Portfolio types moved to portfolio module
from ..portfolio.protocols import (
    PortfolioStateProtocol,
    Position,
    RiskMetrics,
)
# Use canonical types
from ..core.events.types import Event
from ..strategy.types import Signal


class SignalProcessorProtocol(Protocol):
    """Protocol for signal to order processing."""
    
    @abstractmethod
    def process_signal(
        self,
        signal_event: Event,
        portfolio_state: PortfolioStateProtocol,
        position_sizer: PositionSizerProtocol,
        risk_limits: List[RiskLimitProtocol],
        market_data: Dict[str, Any]
    ) -> Optional[Event]:
        """Process signal into order.
        
        Args:
            signal_event: Trading signal event
            portfolio_state: Current portfolio state
            position_sizer: Position sizing strategy
            risk_limits: Risk limits to check
            market_data: Current market data
            
        Returns:
            Order event if approved, None if vetoed
        """
        ...


class RiskPortfolioProtocol(Component, Protocol):
    """Unified Risk & Portfolio management protocol."""
    
    @abstractmethod
    def process_signals(
        self,
        signals: List[Signal],
        market_data: Dict[str, Any]
    ) -> List[Order]:
        """Process multiple signals into orders.
        
        Args:
            signals: List of trading signals
            market_data: Current market data
            
        Returns:
            List of approved orders
        """
        ...
    
    @abstractmethod
    def get_portfolio_state(self) -> PortfolioStateProtocol:
        """Get current portfolio state."""
        ...
    
    @abstractmethod
    def update_fills(self, fills: List[Dict[str, Any]]) -> None:
        """Update portfolio with executed fills."""
        ...
    
    @abstractmethod
    def update_market_data(self, market_data: Dict[str, Any]) -> None:
        """Update market data for risk calculations."""
        ...
    
    @abstractmethod
    def add_risk_limit(self, limit: RiskLimitProtocol) -> None:
        """Add a risk limit."""
        ...
    
    @abstractmethod
    def remove_risk_limit(self, limit_type: type) -> None:
        """Remove a risk limit by type."""
        ...
    
    @abstractmethod
    def set_position_sizer(self, sizer: PositionSizerProtocol) -> None:
        """Set position sizing strategy."""
        ...
    
    @abstractmethod
    def get_risk_report(self) -> Dict[str, Any]:
        """Get comprehensive risk report."""
        ...


@runtime_checkable
class StatelessRiskValidator(Protocol):
    """
    Protocol for stateless risk validation components in unified architecture.
    
    Risk validators are pure functions that validate orders against risk limits.
    They maintain no internal state - portfolio state is passed as a parameter.
    This enables perfect parallelization for testing multiple risk configurations
    simultaneously without container overhead.
    """
    
    def validate_order(
        self, 
        order: Dict[str, Any], 
        portfolio_state: Dict[str, Any], 
        risk_limits: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate an order against risk limits.
        
        This is a pure function - no side effects or state mutations.
        
        Args:
            order: Order to validate with symbol, quantity, side, etc.
            portfolio_state: Current portfolio state (positions, cash, etc.)
            risk_limits: Risk parameters (max position, max drawdown, etc.)
            market_data: Current market prices and conditions
            
        Returns:
            Validation result with:
                - approved: bool indicating if order passes risk checks
                - adjusted_quantity: optional adjusted order size
                - reason: string explanation if rejected
                - risk_metrics: calculated risk metrics
        """
        ...
    
    def calculate_position_size(
        self,
        signal: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        risk_params: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> float:
        """
        Calculate appropriate position size for a signal.
        
        This is a pure function - no side effects or state mutations.
        
        Args:
            signal: Trading signal with direction and strength
            portfolio_state: Current portfolio state
            risk_params: Risk parameters for sizing
            market_data: Current market prices
            
        Returns:
            Position size (number of shares/contracts)
        """
        ...


# Risk capabilities
class RiskCapability(Capability):
    """Base capability for risk management."""
    pass


class PositionSizingCapability(RiskCapability):
    """Capability for position sizing."""
    pass


class RiskLimitCapability(RiskCapability):
    """Capability for risk limit enforcement."""
    pass


# PortfolioTrackingCapability moved to portfolio module


# DEPRECATED: Legacy protocols for backward compatibility
# TODO: Migrate code to use RiskPortfolioProtocol instead
class RiskManager(Protocol):
    """Protocol for risk management components."""
    
    @abstractmethod
    def check_risk_limits(self, signal: Signal, portfolio_state: Any) -> bool:
        """Check if a signal passes risk limits."""
        ...
    
    @abstractmethod
    def calculate_position_size(self, signal: Signal, portfolio_state: Any) -> Decimal:
        """Calculate appropriate position size."""
        ...
    
    @abstractmethod
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        ...
    
    @abstractmethod
    def update_risk_state(self, position: Position) -> None:
        """Update risk state with new position."""
        ...


class PortfolioManager(Protocol):
    """DEPRECATED: Protocol for portfolio management components.
    TODO: Migrate code to use PortfolioStateProtocol instead."""
    
    @abstractmethod
    def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value."""
        ...
    
    @abstractmethod
    def get_cash_balance(self) -> Decimal:
        """Get available cash balance."""
        ...
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Get all open positions."""
        ...
    
    @abstractmethod
    def update_position(self, symbol: str, position: Position) -> None:
        """Update or add a position."""
        ...
    
    @abstractmethod
    def close_position(self, symbol: str) -> Optional[Position]:
        """Close a position and return it."""
        ...