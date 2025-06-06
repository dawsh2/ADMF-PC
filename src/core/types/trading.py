"""
Trading-related types for ADMF-PC.

Simple dataclasses for trading concepts - no complex inheritance.
"""

from dataclasses import dataclass
from datetime import datetime
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


class SignalType(str, Enum):
    """Signal type enumeration."""
    ENTRY = "entry"
    EXIT = "exit"
    REBALANCE = "rebalance"


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
    metadata: Optional[Dict[str, Any]] = None


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
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Signal:
    """Trading signal."""
    symbol: str
    direction: str  # 'long', 'short', 'flat'
    strength: float  # 0.0 to 1.0
    timestamp: datetime
    source: str  # Strategy that generated the signal
    metadata: Optional[Dict[str, Any]] = None