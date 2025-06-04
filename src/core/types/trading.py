"""
File: src/core/types.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#shared-types
Dependencies: enum, decimal, datetime

Shared types used across multiple modules.
Breaks circular dependencies by providing common type definitions.
"""

from enum import Enum
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = 1
    SELL = -1


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class SignalType(Enum):
    """Signal type enumeration."""
    ENTRY = "entry"
    EXIT = "exit"
    REBALANCE = "rebalance"
    RISK_EXIT = "risk_exit"


class FillType(Enum):
    """Fill type enumeration."""
    PARTIAL = "partial"
    FULL = "full"


class FillStatus(Enum):
    """Fill status enumeration."""
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# Core data structures
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal strength."""
        if not Decimal('-1') <= self.strength <= Decimal('1'):
            raise ValueError(f"Signal strength must be between -1 and 1, got {self.strength}")


@dataclass(frozen=True)
class Order:
    """Trading order to be executed."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None  # None for market orders
    stop_price: Optional[Decimal] = None  # For stop orders
    time_in_force: str = "GTC"  # GTC, IOC, FOK, etc.
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Current position in a symbol."""
    symbol: str
    quantity: Decimal
    average_price: Decimal
    current_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    realized_pnl: Decimal = Decimal('0')
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def market_value(self) -> Optional[Decimal]:
        """Calculate market value of position."""
        if self.current_price is not None:
            return self.quantity * self.current_price
        return None