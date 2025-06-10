"""
Asynchronous protocols for execution.

These define the interfaces for asynchronous execution operations.
"""

from typing import Protocol, List, Dict, Any, Optional
from .types import Order, Fill, Position, ExecutionStats


class AsyncBroker(Protocol):
    """Asynchronous broker interface for live trading."""
    
    @property
    def supported_order_types(self) -> List[str]:
        """Get supported order types."""
        ...
    
    @property
    def min_order_size(self) -> float:
        """Get minimum order size."""
        ...
    
    async def connect(self) -> None:
        """Connect to broker API."""
        ...
    
    async def disconnect(self) -> None:
        """Disconnect from broker API."""
        ...
    
    async def validate_order(self, order: Order) -> tuple[bool, Optional[str]]:
        """Validate order constraints. Returns (valid, error_message)."""
        ...
    
    async def submit_order(self, order: Order) -> str:
        """Submit order for execution. Returns broker order ID."""
        ...
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        ...
    
    async def get_order_status(self, order_id: str) -> Optional[str]:
        """Get current order status."""
        ...
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        ...
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        ...
    
    async def get_recent_fills(self) -> List[Fill]:
        """Get recent fills from broker."""
        ...


class AsyncEngine(Protocol):
    """Asynchronous execution engine for live trading."""
    
    async def start(self) -> None:
        """Start execution engine."""
        ...
    
    async def stop(self) -> None:
        """Stop execution engine."""
        ...
    
    async def submit_order(self, order: Order) -> Optional[str]:
        """Submit order for execution. Returns broker order ID."""
        ...
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        ...
    
    async def get_execution_stats(self) -> ExecutionStats:
        """Get execution statistics."""
        ...


class MarketDataFeed(Protocol):
    """Asynchronous market data feed interface."""
    
    async def connect(self) -> None:
        """Connect to market data feed."""
        ...
    
    async def disconnect(self) -> None:
        """Disconnect from market data feed."""
        ...
    
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols."""
        ...
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols."""
        ...
    
    async def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest bar data for symbol."""
        ...


class OrderMonitor(Protocol):
    """Asynchronous order monitoring interface."""
    
    async def start_monitoring(self) -> None:
        """Start order monitoring."""
        ...
    
    async def stop_monitoring(self) -> None:
        """Stop order monitoring."""
        ...
    
    async def track_order(self, order: Order, broker_order_id: str) -> None:
        """Track order status."""
        ...
    
    async def get_order_updates(self) -> List[Dict[str, Any]]:
        """Get recent order updates."""
        ...