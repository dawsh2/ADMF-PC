"""Execution capabilities definition."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .protocols import ExecutionCapability, OrderType, Order
from ..core.infrastructure.capabilities import Capability
import logging


logger = logging.getLogger(__name__)


@dataclass
class BasicExecutionCapability:
    """Basic execution capability implementation."""
    
    capability_id: str
    supported_order_types: List[OrderType] = field(default_factory=list)
    supported_symbols: List[str] = field(default_factory=list)
    max_order_size: Optional[float] = None
    min_order_size: float = 1.0
    
    def __post_init__(self):
        if not self.supported_order_types:
            # Default to all order types
            self.supported_order_types = list(OrderType)
        
        if not self.supported_symbols:
            # Default to supporting all symbols
            self.supported_symbols = ["*"]  # Wildcard
    
    async def validate_capability(self, order: Order) -> bool:
        """Validate if order is supported by capability."""
        # Check order type
        if order.order_type not in self.supported_order_types:
            logger.warning(
                f"Order type {order.order_type} not supported by {self.capability_id}"
            )
            return False
        
        # Check symbol
        if "*" not in self.supported_symbols and order.symbol not in self.supported_symbols:
            logger.warning(
                f"Symbol {order.symbol} not supported by {self.capability_id}"
            )
            return False
        
        # Check order size
        if order.quantity < self.min_order_size:
            logger.warning(
                f"Order size {order.quantity} below minimum {self.min_order_size}"
            )
            return False
        
        if self.max_order_size and order.quantity > self.max_order_size:
            logger.warning(
                f"Order size {order.quantity} above maximum {self.max_order_size}"
            )
            return False
        
        return True


class ExecutionCapabilities:
    """Collection of execution capabilities."""
    
    # Standard market order capability
    MARKET_ORDERS = BasicExecutionCapability(
        capability_id="market_orders",
        supported_order_types=[OrderType.MARKET],
        supported_symbols=["*"]
    )
    
    # Limit order capability
    LIMIT_ORDERS = BasicExecutionCapability(
        capability_id="limit_orders",
        supported_order_types=[OrderType.LIMIT],
        supported_symbols=["*"]
    )
    
    # Stop order capability
    STOP_ORDERS = BasicExecutionCapability(
        capability_id="stop_orders",
        supported_order_types=[OrderType.STOP, OrderType.STOP_LIMIT],
        supported_symbols=["*"]
    )
    
    # All order types capability
    ALL_ORDER_TYPES = BasicExecutionCapability(
        capability_id="all_order_types",
        supported_order_types=list(OrderType),
        supported_symbols=["*"]
    )
    
    # Equity trading capability
    EQUITY_TRADING = BasicExecutionCapability(
        capability_id="equity_trading",
        supported_order_types=list(OrderType),
        supported_symbols=["*"],  # Could be restricted to specific symbols
        min_order_size=1.0,
        max_order_size=10000.0
    )
    
    # Small lot capability
    SMALL_LOTS = BasicExecutionCapability(
        capability_id="small_lots",
        supported_order_types=[OrderType.MARKET, OrderType.LIMIT],
        supported_symbols=["*"],
        min_order_size=1.0,
        max_order_size=100.0
    )
    
    # Large order capability
    LARGE_ORDERS = BasicExecutionCapability(
        capability_id="large_orders",
        supported_order_types=list(OrderType),
        supported_symbols=["*"],
        min_order_size=1000.0,
        max_order_size=100000.0
    )
    
    @classmethod
    def create_custom(
        cls,
        capability_id: str,
        order_types: Optional[List[OrderType]] = None,
        symbols: Optional[List[str]] = None,
        min_size: float = 1.0,
        max_size: Optional[float] = None
    ) -> BasicExecutionCapability:
        """Create custom execution capability."""
        return BasicExecutionCapability(
            capability_id=capability_id,
            supported_order_types=order_types or list(OrderType),
            supported_symbols=symbols or ["*"],
            min_order_size=min_size,
            max_order_size=max_size
        )
    
    @classmethod
    def combine(
        cls,
        capabilities: List[ExecutionCapability],
        capability_id: str = "combined"
    ) -> BasicExecutionCapability:
        """Combine multiple capabilities."""
        # Aggregate supported order types and symbols
        order_types = set()
        symbols = set()
        min_size = 0.0
        max_size = float('inf')
        
        for cap in capabilities:
            order_types.update(cap.supported_order_types)
            
            if "*" in cap.supported_symbols:
                symbols.add("*")
            else:
                symbols.update(cap.supported_symbols)
            
            if hasattr(cap, 'min_order_size'):
                min_size = max(min_size, cap.min_order_size)
            
            if hasattr(cap, 'max_order_size') and cap.max_order_size:
                max_size = min(max_size, cap.max_order_size)
        
        return BasicExecutionCapability(
            capability_id=capability_id,
            supported_order_types=list(order_types),
            supported_symbols=list(symbols),
            min_order_size=min_size,
            max_order_size=max_size if max_size != float('inf') else None
        )


def create_execution_capability(
    name: str,
    order_types: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
    constraints: Optional[Dict[str, Any]] = None
) -> Capability:
    """Create execution capability for infrastructure."""
    config = {
        "order_types": order_types or ["MARKET", "LIMIT"],
        "symbols": symbols or ["*"],
        "constraints": constraints or {}
    }
    
    return Capability(
        name=f"execution.{name}",
        version="1.0.0",
        dependencies=[],
        config=config
    )