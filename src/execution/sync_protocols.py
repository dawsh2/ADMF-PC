"""
Synchronous protocols for execution.

These define the interfaces for synchronous execution operations.
"""

from typing import Protocol, List, Dict, Any, Optional
from .types import Order, Fill, Position, ExecutionStats


class SyncBroker(Protocol):
    """Synchronous broker interface for backtesting."""
    
    @property
    def supported_order_types(self) -> List[str]:
        """Get supported order types."""
        ...
    
    @property
    def min_order_size(self) -> float:
        """Get minimum order size."""
        ...
    
    def validate_order(self, order: Order) -> tuple[bool, Optional[str]]:
        """Validate order constraints. Returns (valid, error_message)."""
        ...
    
    def submit_order(self, order: Order) -> str:
        """Submit order for execution. Returns broker order ID."""
        ...
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        ...
    
    def get_order_status(self, order_id: str) -> Optional[str]:
        """Get current order status."""
        ...
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        ...
    
    def process_market_data(self, market_data: Dict[str, Any]) -> List[Fill]:
        """Process market data and generate fills."""
        ...


class SyncEngine(Protocol):
    """Synchronous execution engine for backtesting."""
    
    def initialize(self) -> None:
        """Initialize execution engine."""
        ...
    
    def execute_order(self, order: Order) -> Optional[Fill]:
        """Execute order and return fill if successful."""
        ...
    
    def process_market_data(self, market_data: Dict[str, Any]) -> List[Fill]:
        """Process market data and execute pending orders."""
        ...
    
    def get_execution_stats(self) -> ExecutionStats:
        """Get execution statistics."""
        ...
    
    def shutdown(self) -> None:
        """Shutdown execution engine."""
        ...


class SlippageModel(Protocol):
    """Interface for slippage calculation models."""
    
    def calculate_slippage(
        self, 
        order: Order, 
        market_price: float, 
        volume: float = 0
    ) -> float:
        """Calculate slippage for order."""
        ...


class CommissionModel(Protocol):
    """Interface for commission calculation models."""
    
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """Calculate commission for order."""
        ...


class LiquidityModel(Protocol):
    """Interface for liquidity simulation models."""
    
    def can_fill_order(
        self, 
        order: Order, 
        market_data: Dict[str, Any]
    ) -> tuple[bool, float]:
        """Check if order can be filled. Returns (can_fill, fill_ratio)."""
        ...