"""Protocol definitions for risk management components."""

from abc import abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Protocol, Optional, Dict, List, Any, Set
from enum import Enum

from ..core.components.protocols import Component, Capability
from ..execution.protocols import OrderType, OrderSide


class SignalType(Enum):
    """Signal type enumeration."""
    ENTRY = "entry"
    EXIT = "exit"
    REBALANCE = "rebalance"
    RISK_EXIT = "risk_exit"


@dataclass(frozen=True)
class Signal:
    """Trading signal from strategy."""
    signal_id: str
    strategy_id: str
    symbol: str
    signal_type: SignalType
    side: OrderSide
    strength: Decimal  # -1 to 1, magnitude indicates confidence
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate signal strength."""
        if not -1 <= self.strength <= 1:
            raise ValueError(f"Signal strength must be between -1 and 1, got {self.strength}")


@dataclass(frozen=True)
class Order:
    """Trading order to be executed."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal]  # None for market orders
    stop_price: Optional[Decimal]  # For stop orders
    time_in_force: str  # GTC, IOC, FOK, etc.
    source_signal: Signal
    risk_checks_passed: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class Position:
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


class SignalProcessorProtocol(Protocol):
    """Protocol for signal to order processing."""
    
    @abstractmethod
    def process_signal(
        self,
        signal: Signal,
        portfolio_state: PortfolioStateProtocol,
        position_sizer: PositionSizerProtocol,
        risk_limits: List[RiskLimitProtocol],
        market_data: Dict[str, Any]
    ) -> Optional[Order]:
        """Process signal into order.
        
        Args:
            signal: Trading signal
            portfolio_state: Current portfolio state
            position_sizer: Position sizing strategy
            risk_limits: Risk limits to check
            market_data: Current market data
            
        Returns:
            Order if approved, None if vetoed
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


class PortfolioTrackingCapability(RiskCapability):
    """Capability for portfolio state tracking."""
    pass


# Add missing protocols for backward compatibility
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
    """Protocol for portfolio management components."""
    
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