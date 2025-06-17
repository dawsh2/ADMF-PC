"""
Execution types and data structures.

Core trading types used across backtest and live execution.
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional, Dict, Any
from enum import Enum
from decimal import Decimal


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class FillStatus(str, Enum):
    """Fill status enumeration."""
    FILLED = "filled"
    PARTIAL = "partial"


@dataclass
class Order:
    """Trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None  # For limit orders
    stop_price: Optional[Decimal] = None  # For stop orders
    time_in_force: str = "DAY"
    created_at: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': str(self.quantity),
            'price': str(self.price) if self.price else None,
            'stop_price': str(self.stop_price) if self.stop_price else None,
            'time_in_force': self.time_in_force,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'status': self.status.value,
            'metadata': self.metadata
        }


@dataclass
class Fill:
    """Order execution fill."""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    commission: Decimal
    executed_at: datetime
    status: FillStatus = FillStatus.FILLED
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fill to dictionary."""
        return {
            'fill_id': self.fill_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': str(self.quantity),
            'price': str(self.price),
            'commission': str(self.commission),
            'executed_at': self.executed_at.isoformat() if self.executed_at else None,
            'status': self.status.value,
            'metadata': self.metadata
        }


@dataclass
class Position:
    """Position representation."""
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Bar:
    """Market bar data (OHLCV)."""
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': float(self.open),
            'high': float(self.high),
            'low': float(self.low),
            'close': float(self.close),
            'volume': float(self.volume)
        }


@dataclass
class ExecutionStats:
    """Execution statistics."""
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_cancelled: int = 0
    orders_rejected: int = 0
    total_commission: Decimal = Decimal('0')
    total_slippage: Decimal = Decimal('0')
    avg_fill_time_ms: float = 0.0
    
    @property
    def fill_rate(self) -> float:
        """Calculate fill rate percentage."""
        if self.orders_submitted == 0:
            return 0.0
        return (self.orders_filled / self.orders_submitted) * 100


# ============================================
# Trading time utilities
# ============================================

# US Market hours (Eastern Time)
MARKET_OPEN = time(9, 30)  # 9:30 AM ET
MARKET_CLOSE = time(16, 0)  # 4:00 PM ET
PRE_MARKET_OPEN = time(4, 0)  # 4:00 AM ET
POST_MARKET_CLOSE = time(20, 0)  # 8:00 PM ET


def is_market_open(dt: datetime, include_extended: bool = False) -> bool:
    """
    Check if market is open at given time.
    
    Args:
        dt: Datetime to check (should be in Eastern Time)
        include_extended: Include pre/post market hours
        
    Returns:
        True if market is open
    """
    # Check if it's a weekday
    if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    market_time = dt.time()
    
    if include_extended:
        return PRE_MARKET_OPEN <= market_time <= POST_MARKET_CLOSE
    else:
        return MARKET_OPEN <= market_time <= MARKET_CLOSE


def next_market_open(dt: datetime) -> datetime:
    """
    Get next market open time.
    
    Args:
        dt: Current datetime
        
    Returns:
        Next market open datetime
    """
    # If already during market hours, return current time
    if is_market_open(dt):
        return dt
    
    # If before market open today and it's a weekday
    if dt.weekday() < 5 and dt.time() < MARKET_OPEN:
        return dt.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # Find next weekday
    next_day = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    while True:
        next_day = next_day.replace(day=next_day.day + 1)
        if next_day.weekday() < 5:  # Found a weekday
            return next_day


def format_trading_time(dt: datetime) -> str:
    """
    Format datetime for trading display.
    
    Args:
        dt: Datetime to format
        
    Returns:
        Formatted string like "2024-01-05 09:30:00 ET"
    """
    return dt.strftime('%Y-%m-%d %H:%M:%S ET')