"""Protocol definitions for portfolio management components."""

from abc import abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Protocol, Optional, Dict, List, Any, runtime_checkable

from ..core.components.protocols import Component, Capability
# Import from module-specific types  
from ..core.events.types import Event
from ..strategy.types import SignalType, Signal
from ..execution.types import OrderType, OrderSide, Position

# Use alias to avoid confusion - portfolio has extended position info
@dataclass  
class PortfolioPosition:
    """Current position in a symbol."""
    symbol: str
    quantity: Decimal
    average_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    opened_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any]
    
    @property
    def market_value(self) -> Decimal:
        """Calculate current market value."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> Decimal:
        """Calculate cost basis."""
        return self.quantity * self.average_price
    
    @property
    def pnl_percentage(self) -> Decimal:
        """Calculate P&L percentage."""
        if self.cost_basis == 0:
            return Decimal(0)
        return (self.unrealized_pnl / abs(self.cost_basis)) * 100


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    total_value: Decimal
    cash_balance: Decimal
    positions_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    max_drawdown: Decimal
    current_drawdown: Decimal
    sharpe_ratio: Optional[Decimal]
    var_95: Optional[Decimal]  # Value at Risk 95%
    leverage: Decimal
    concentration: Dict[str, Decimal]  # Symbol -> % of portfolio
    timestamp: datetime


class PortfolioStateProtocol(Protocol):
    """Protocol for portfolio state tracking."""
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""
        ...
    
    @abstractmethod
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        ...
    
    @abstractmethod
    def get_cash_balance(self) -> Decimal:
        """Get current cash balance."""
        ...
    
    @abstractmethod
    def get_total_value(self) -> Decimal:
        """Get total portfolio value."""
        ...
    
    @abstractmethod
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        ...
    
    @abstractmethod
    def update_position(
        self,
        symbol: str,
        quantity_delta: Decimal,
        price: Decimal,
        timestamp: datetime
    ) -> Position:
        """Update position with a trade."""
        ...
    
    @abstractmethod
    def update_market_prices(self, prices: Dict[str, Decimal]) -> None:
        """Update market prices for positions."""
        ...
    
    @abstractmethod
    def process_event(self, event: Event) -> None:
        """Process an event (fill, market data, etc)."""
        ...


class PortfolioManagerProtocol(Component, Protocol):
    """Protocol for portfolio management components."""
    
    @abstractmethod
    def get_portfolio_state(self) -> PortfolioStateProtocol:
        """Get current portfolio state."""
        ...
    
    @abstractmethod
    def process_event(self, event: Event) -> Optional[Event]:
        """Process an event and optionally return a response event."""
        ...
    
    @abstractmethod
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive portfolio performance report."""
        ...


# Portfolio capabilities
class PortfolioCapability(Capability):
    """Base capability for portfolio management."""
    pass


class PortfolioTrackingCapability(PortfolioCapability):
    """Capability for portfolio state tracking."""
    pass


class PerformanceAnalysisCapability(PortfolioCapability):
    """Capability for performance analysis."""
    pass


# Legacy protocols removed - use PortfolioManagerProtocol instead
