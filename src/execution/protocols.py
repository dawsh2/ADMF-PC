"""
Execution module protocols.
"""

from typing import Protocol, Set, Dict, Any, Optional
from enum import Enum
from ..core.events.types import Event

class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"

class OrderProcessor(Protocol):
    """Protocol for order processing components."""
    
    def get_pending_orders(self) -> Set[str]:
        """Get all pending order IDs."""
        ...
    
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order details by ID."""
        ...