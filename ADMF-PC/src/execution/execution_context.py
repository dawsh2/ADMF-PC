"""Execution context for thread safety management."""

import asyncio
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import threading
from contextlib import asynccontextmanager

from ..core.logging.structured import get_logger


logger = get_logger(__name__)


@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    total_fills: int = 0
    total_volume: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    def add_order(self):
        """Increment order count."""
        self.total_orders += 1
    
    def add_fill(self, volume: float, commission: float, slippage: float):
        """Add fill metrics."""
        self.total_fills += 1
        self.total_volume += volume
        self.total_commission += commission
        self.total_slippage += slippage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_orders": self.total_orders,
            "filled_orders": self.filled_orders,
            "cancelled_orders": self.cancelled_orders,
            "rejected_orders": self.rejected_orders,
            "total_fills": self.total_fills,
            "total_volume": self.total_volume,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "fill_rate": self.filled_orders / self.total_orders if self.total_orders > 0 else 0,
            "avg_commission_per_fill": self.total_commission / self.total_fills if self.total_fills > 0 else 0,
            "avg_slippage_per_fill": self.total_slippage / self.total_fills if self.total_fills > 0 else 0
        }


class ExecutionContext:
    """Thread-safe execution context."""
    
    def __init__(self):
        """Initialize execution context."""
        self._state: Dict[str, Any] = {}
        self._metrics = ExecutionMetrics()
        self._active_orders: Set[str] = set()
        self._locks: Dict[str, asyncio.Lock] = {}
        self._thread_locals = threading.local()
        self._global_lock = asyncio.Lock()
        
        # Initialize default locks
        self._locks["state"] = asyncio.Lock()
        self._locks["metrics"] = asyncio.Lock()
        self._locks["orders"] = asyncio.Lock()
        
        logger.info("Initialized ExecutionContext")
    
    @asynccontextmanager
    async def transaction(self, transaction_id: str):
        """Execute within a transaction context."""
        async with self._global_lock:
            self._thread_locals.transaction_id = transaction_id
            self._thread_locals.start_time = datetime.now()
            
            logger.debug(f"Starting transaction: {transaction_id}")
        
        try:
            yield self
        finally:
            async with self._global_lock:
                if hasattr(self._thread_locals, "transaction_id"):
                    duration = (datetime.now() - self._thread_locals.start_time).total_seconds()
                    logger.debug(
                        f"Completed transaction: {transaction_id} "
                        f"(duration: {duration:.3f}s)"
                    )
                    delattr(self._thread_locals, "transaction_id")
                    delattr(self._thread_locals, "start_time")
    
    async def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value thread-safely."""
        async with self._locks["state"]:
            return self._state.get(key, default)
    
    async def set_state(self, key: str, value: Any) -> None:
        """Set state value thread-safely."""
        async with self._locks["state"]:
            self._state[key] = value
            logger.debug(f"State updated: {key}")
    
    async def update_state(self, updates: Dict[str, Any]) -> None:
        """Update multiple state values atomically."""
        async with self._locks["state"]:
            self._state.update(updates)
            logger.debug(f"State batch updated: {list(updates.keys())}")
    
    async def add_active_order(self, order_id: str) -> None:
        """Add order to active set."""
        async with self._locks["orders"]:
            self._active_orders.add(order_id)
            await self.increment_metric("total_orders")
            logger.debug(f"Order added: {order_id}")
    
    async def remove_active_order(self, order_id: str) -> None:
        """Remove order from active set."""
        async with self._locks["orders"]:
            self._active_orders.discard(order_id)
            logger.debug(f"Order removed: {order_id}")
    
    async def is_order_active(self, order_id: str) -> bool:
        """Check if order is active."""
        async with self._locks["orders"]:
            return order_id in self._active_orders
    
    async def get_active_orders(self) -> Set[str]:
        """Get copy of active orders."""
        async with self._locks["orders"]:
            return self._active_orders.copy()
    
    async def record_fill(
        self,
        order_id: str,
        volume: float,
        commission: float,
        slippage: float
    ) -> None:
        """Record fill metrics."""
        async with self._locks["metrics"]:
            self._metrics.add_fill(volume, commission, slippage)
            self._metrics.filled_orders += 1
            logger.debug(
                f"Fill recorded: {order_id} - volume: {volume}, "
                f"commission: {commission}, slippage: {slippage}"
            )
    
    async def record_order_status(self, status: str) -> None:
        """Record order status update."""
        async with self._locks["metrics"]:
            if status == "cancelled":
                self._metrics.cancelled_orders += 1
            elif status == "rejected":
                self._metrics.rejected_orders += 1
    
    async def increment_metric(self, metric: str, value: float = 1) -> None:
        """Increment a metric value."""
        async with self._locks["metrics"]:
            if hasattr(self._metrics, metric):
                current = getattr(self._metrics, metric)
                setattr(self._metrics, metric, current + value)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        async with self._locks["metrics"]:
            return self._metrics.to_dict()
    
    async def acquire_lock(self, resource: str) -> asyncio.Lock:
        """Acquire or create a resource lock."""
        async with self._global_lock:
            if resource not in self._locks:
                self._locks[resource] = asyncio.Lock()
            return self._locks[resource]
    
    async def execute_atomic(self, func, *args, **kwargs):
        """Execute function atomically."""
        async with self._global_lock:
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
    
    def get_thread_id(self) -> int:
        """Get current thread ID."""
        return threading.get_ident()
    
    def get_transaction_id(self) -> Optional[str]:
        """Get current transaction ID if in transaction."""
        return getattr(self._thread_locals, "transaction_id", None)
    
    async def reset(self) -> None:
        """Reset context state."""
        async with self._global_lock:
            self._state.clear()
            self._metrics = ExecutionMetrics()
            self._active_orders.clear()
            logger.info("ExecutionContext reset")