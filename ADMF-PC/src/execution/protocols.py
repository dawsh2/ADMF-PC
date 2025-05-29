"""Execution protocols and interfaces."""

from typing import Protocol, Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum, auto

from ..core.events.types import Event


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = auto()
    SELL = auto()


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = auto()
    SUBMITTED = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()


class FillType(Enum):
    """Fill type enumeration."""
    FULL = auto()
    PARTIAL = auto()


@dataclass
class Order:
    """Order representation."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    time_in_force: str = "DAY"
    created_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Fill:
    """Fill/execution representation."""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    fill_type: FillType
    executed_at: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Position:
    """Position representation."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Broker(Protocol):
    """Broker interface for order execution."""
    
    async def submit_order(self, order: Order) -> str:
        """Submit order for execution."""
        ...
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        ...
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current order status."""
        ...
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        ...
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        ...


class OrderProcessor(Protocol):
    """Order processing interface."""
    
    async def process_order(self, order: Order) -> Optional[Fill]:
        """Process order and return fill if executed."""
        ...
    
    async def validate_order(self, order: Order) -> bool:
        """Validate order before processing."""
        ...
    
    async def update_order_status(self, order_id: str, status: OrderStatus) -> None:
        """Update order status."""
        ...


class MarketSimulator(Protocol):
    """Market simulation interface."""
    
    async def simulate_fill(
        self,
        order: Order,
        market_price: float,
        volume: float
    ) -> Optional[Fill]:
        """Simulate order fill with market conditions."""
        ...
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
        volume: float
    ) -> float:
        """Calculate slippage for order."""
        ...
    
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """Calculate commission for order."""
        ...


class ExecutionEngine(Protocol):
    """Main execution engine interface."""
    
    async def initialize(self) -> None:
        """Initialize execution engine."""
        ...
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process incoming event."""
        ...
    
    async def execute_order(self, order: Order) -> Optional[Fill]:
        """Execute order through broker."""
        ...
    
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        ...
    
    async def shutdown(self) -> None:
        """Shutdown execution engine."""
        ...


class ExecutionCapability(Protocol):
    """Execution capability interface."""
    
    @property
    def capability_id(self) -> str:
        """Unique capability identifier."""
        ...
    
    @property
    def supported_order_types(self) -> List[OrderType]:
        """Supported order types."""
        ...
    
    @property
    def supported_symbols(self) -> List[str]:
        """Supported trading symbols."""
        ...
    
    async def validate_capability(self, order: Order) -> bool:
        """Validate if order is supported by capability."""
        ...