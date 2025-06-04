"""
File: src/risk/models.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#risk-models
Step: 2 - Add Risk Container
Dependencies: dataclasses, decimal, datetime, enum

Data models for Step 2 risk container components.
Defines all risk-related data structures using Protocol + Composition pattern.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import uuid

# Import shared types from core (single source of truth)
from ..core.types import OrderSide, OrderType, SignalType


@dataclass
class TradingSignal:
    """
    Trading signal from strategy to risk management.
    
    Represents a trading opportunity identified by a strategy,
    to be processed by the risk management system.
    """
    signal_id: str
    strategy_id: str
    symbol: str
    signal_type: SignalType
    side: OrderSide
    strength: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal data."""
        if not (0 <= self.strength <= 1):
            raise ValueError("Signal strength must be between 0 and 1")


@dataclass
class Order:
    """
    Order to be sent to execution engine.
    
    Represents a risk-adjusted order created from a trading signal.
    """
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "GTC"
    source_signal: Optional[TradingSignal] = None
    risk_checks_passed: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate order data."""
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")


@dataclass
class Fill:
    """
    Fill event from execution engine.
    
    Represents an executed order that needs to update portfolio state.
    """
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    commission: Decimal = Decimal('0')
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate fill data."""
        if self.quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        if self.price <= 0:
            raise ValueError("Fill price must be positive")


@dataclass
class Position:
    """
    Position in a single symbol.
    
    Tracks quantity, average price, and P&L for a position.
    """
    symbol: str
    quantity: Decimal = Decimal('0')
    avg_price: Decimal = Decimal('0')
    current_price: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    opened_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, fill: Fill) -> None:
        """
        Update position with a fill.
        
        Args:
            fill: Fill event to apply to position
        """
        fill_quantity = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        
        if self.quantity == 0:
            # New position
            self.quantity = fill_quantity
            self.avg_price = fill.price
            self.opened_at = fill.timestamp
        else:
            old_quantity = self.quantity
            new_quantity = old_quantity + fill_quantity
            
            if new_quantity == 0:
                # Position closed - calculate realized P&L
                self.realized_pnl += (fill.price - self.avg_price) * abs(fill_quantity)
                self.quantity = Decimal('0')
                self.avg_price = Decimal('0')
            elif (old_quantity > 0 and fill_quantity > 0) or (old_quantity < 0 and fill_quantity < 0):
                # Adding to position - update average price
                total_cost = self.avg_price * abs(old_quantity) + fill.price * abs(fill_quantity)
                total_quantity = abs(old_quantity) + abs(fill_quantity)
                self.avg_price = total_cost / total_quantity
                self.quantity = new_quantity
            else:
                # Reducing position - realize some P&L
                close_quantity = min(abs(old_quantity), abs(fill_quantity))
                self.realized_pnl += (fill.price - self.avg_price) * close_quantity
                self.quantity = new_quantity
        
        self.last_updated = fill.timestamp
        self._update_unrealized_pnl()
    
    def _update_unrealized_pnl(self) -> None:
        """Update unrealized P&L based on current price."""
        if self.quantity != 0 and self.current_price != 0:
            self.unrealized_pnl = (self.current_price - self.avg_price) * self.quantity
    
    def update_current_price(self, price: Decimal) -> None:
        """
        Update current market price.
        
        Args:
            price: New current market price
        """
        self.current_price = price
        self._update_unrealized_pnl()
        self.last_updated = datetime.now()
    
    @property
    def market_value(self) -> Decimal:
        """Get current market value of position."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> Decimal:
        """Get cost basis of position."""
        return self.quantity * self.avg_price


@dataclass
class RiskConfig:
    """
    Risk management configuration.
    
    Contains all parameters needed to configure risk management components.
    """
    # Portfolio settings
    initial_capital: float = 100000.0
    sizing_method: str = "fixed"  # "fixed", "percent_risk", "volatility"
    
    # Position limits
    max_position_size: float = 0.1  # 10% max position size
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk per trade
    max_correlation: float = 0.7  # Max correlation between positions
    max_drawdown: float = 0.2  # 20% max drawdown
    
    # Position sizing parameters
    fixed_position_size: float = 1000.0  # Fixed dollar amount
    percent_risk_per_trade: float = 0.01  # 1% risk per trade
    volatility_lookback: int = 20  # Days for volatility calculation
    
    # Risk limits
    max_leverage: float = 1.0  # No leverage by default
    max_concentration: float = 0.2  # 20% max concentration in single position
    
    # Timing constraints
    max_orders_per_minute: int = 10
    cooldown_period_seconds: int = 60
    
    # Stop loss settings
    default_stop_loss_pct: float = 0.05  # 5% stop loss
    use_trailing_stops: bool = False
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'initial_capital': self.initial_capital,
            'sizing_method': self.sizing_method,
            'max_position_size': self.max_position_size,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_correlation': self.max_correlation,
            'max_drawdown': self.max_drawdown,
            'fixed_position_size': self.fixed_position_size,
            'percent_risk_per_trade': self.percent_risk_per_trade,
            'volatility_lookback': self.volatility_lookback,
            'max_leverage': self.max_leverage,
            'max_concentration': self.max_concentration,
            'max_orders_per_minute': self.max_orders_per_minute,
            'cooldown_period_seconds': self.cooldown_period_seconds,
            'default_stop_loss_pct': self.default_stop_loss_pct,
            'use_trailing_stops': self.use_trailing_stops,
            'trailing_stop_pct': self.trailing_stop_pct
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskConfig':
        """Create from dictionary."""
        return cls(**data)


# Utility functions for creating test objects

def create_test_signal(
    symbol: str = "TEST",
    side: OrderSide = OrderSide.BUY,
    strength: float = 0.8,
    strategy_id: str = "test_strategy"
) -> TradingSignal:
    """Create a test trading signal."""
    return TradingSignal(
        signal_id=str(uuid.uuid4()),
        strategy_id=strategy_id,
        symbol=symbol,
        signal_type=SignalType.ENTRY,
        side=side,
        strength=Decimal(str(strength)),
        timestamp=datetime.now()
    )


def create_test_order(
    symbol: str = "TEST",
    side: OrderSide = OrderSide.BUY,
    quantity: float = 100.0,
    price: Optional[float] = None
) -> Order:
    """Create a test order."""
    return Order(
        order_id=str(uuid.uuid4()),
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET if price is None else OrderType.LIMIT,
        quantity=Decimal(str(quantity)),
        price=Decimal(str(price)) if price else None
    )


def create_test_fill(
    symbol: str = "TEST",
    side: OrderSide = OrderSide.BUY,
    quantity: float = 100.0,
    price: float = 100.0,
    order_id: Optional[str] = None
) -> Fill:
    """Create a test fill."""
    return Fill(
        fill_id=str(uuid.uuid4()),
        order_id=order_id or str(uuid.uuid4()),
        symbol=symbol,
        side=side,
        quantity=Decimal(str(quantity)),
        price=Decimal(str(price)),
        timestamp=datetime.now()
    )