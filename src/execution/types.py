"""
Trading types and time utilities for execution module.

Simple dataclasses for trading concepts - no complex inheritance.
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Optional, Dict, Any
from enum import Enum


# Simple enums for compatibility
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
class Bar:
    """Market bar data (OHLCV)."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


@dataclass
class Position:
    """Position in a security."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        return (self.current_price - self.avg_price) * self.quantity
    
    @property
    def pnl_percent(self) -> float:
        """Calculate P&L percentage."""
        if self.avg_price == 0:
            return 0.0
        return (self.current_price - self.avg_price) / self.avg_price


@dataclass
class Order:
    """Trading order."""
    order_id: str
    symbol: str
    direction: str  # 'long' or 'short'
    quantity: float
    order_type: str  # 'market', 'limit', etc.
    price: Optional[float] = None
    timestamp: Optional[datetime] = None
    portfolio_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Fill:
    """Order execution fill."""
    fill_id: str
    order_id: str
    symbol: str
    direction: str
    quantity: float
    price: float
    timestamp: datetime
    portfolio_id: Optional[str] = None
    commission: float = 0.0
    status: FillStatus = FillStatus.FILLED
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


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


def market_minutes_between(start: datetime, end: datetime) -> int:
    """
    Calculate number of market minutes between two times.
    
    Args:
        start: Start time
        end: End time
        
    Returns:
        Number of minutes when market was open
    """
    if end < start:
        return 0
    
    minutes = 0
    current = start
    
    while current < end:
        if is_market_open(current):
            minutes += 1
        current = current.replace(minute=current.minute + 1)
    
    return minutes


def format_trading_time(dt: datetime) -> str:
    """
    Format datetime for trading display.
    
    Args:
        dt: Datetime to format
        
    Returns:
        Formatted string like "2024-01-05 09:30:00 ET"
    """
    return dt.strftime('%Y-%m-%d %H:%M:%S ET')