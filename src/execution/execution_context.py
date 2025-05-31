"""Execution context for thread safety management."""

import threading
from typing import Dict, Any, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from decimal import Decimal

import logging


logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    total_fills: int = 0
    total_volume: Decimal = Decimal("0")
    total_commission: Decimal = Decimal("0")
    total_slippage: Decimal = Decimal("0")
    
    def add_order(self):
        """Increment order count."""
        self.total_orders += 1
    
    def add_fill(self, volume: Union[float, Decimal], commission: Union[float, Decimal], slippage: Union[float, Decimal]):
        """Add fill metrics."""
        self.total_fills += 1
        # Convert to Decimal if needed for consistent arithmetic
        volume_decimal = Decimal(str(volume)) if not isinstance(volume, Decimal) else volume
        commission_decimal = Decimal(str(commission)) if not isinstance(commission, Decimal) else commission
        slippage_decimal = Decimal(str(slippage)) if not isinstance(slippage, Decimal) else slippage
        
        self.total_volume += volume_decimal
        self.total_commission += commission_decimal
        self.total_slippage += slippage_decimal
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_orders": self.total_orders,
            "filled_orders": self.filled_orders,
            "cancelled_orders": self.cancelled_orders,
            "rejected_orders": self.rejected_orders,
            "total_fills": self.total_fills,
            "total_volume": float(self.total_volume),
            "total_commission": float(self.total_commission),
            "total_slippage": float(self.total_slippage),
            "fill_rate": self.filled_orders / self.total_orders if self.total_orders > 0 else 0,
            "avg_commission_per_fill": float(self.total_commission / self.total_fills) if self.total_fills > 0 else 0,
            "avg_slippage_per_fill": float(self.total_slippage / self.total_fills) if self.total_fills > 0 else 0
        }


class ExecutionContext:
    """Thread-safe execution context."""
    
    def __init__(self):
        """Initialize execution context."""
        self._state: Dict[str, Any] = {}
        self._metrics = ExecutionMetrics()
        self._active_orders: Set[str] = set()
        self._locks: Dict[str, threading.Lock] = {}
        self._thread_locals = threading.local()
        self._global_lock = threading.Lock()
        
        # Initialize default locks
        self._locks["state"] = threading.Lock()
        self._locks["metrics"] = threading.Lock()
        self._locks["orders"] = threading.Lock()
        
        logger.info("Initialized ExecutionContext")
    
    @contextmanager
    def transaction(self, transaction_id: str):
        """Execute within a transaction context."""
        with self._global_lock:
            self._thread_locals.transaction_id = transaction_id
            self._thread_locals.start_time = datetime.now()
            
            logger.debug(f"Starting transaction: {transaction_id}")
        
        try:
            yield self
        finally:
            with self._global_lock:
                if hasattr(self._thread_locals, "transaction_id"):
                    duration = (datetime.now() - self._thread_locals.start_time).total_seconds()
                    logger.debug(
                        f"Completed transaction: {transaction_id} "
                        f"(duration: {duration:.3f}s)"
                    )
                    delattr(self._thread_locals, "transaction_id")
                    delattr(self._thread_locals, "start_time")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value thread-safely."""
        with self._locks["state"]:
            return self._state.get(key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        """Set state value thread-safely."""
        with self._locks["state"]:
            self._state[key] = value
            logger.debug(f"State updated: {key}")
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update multiple state values atomically."""
        with self._locks["state"]:
            self._state.update(updates)
            logger.debug(f"State batch updated: {list(updates.keys())}")
    
    def add_active_order(self, order_id: str) -> None:
        """Add order to active set."""
        with self._locks["orders"]:
            self._active_orders.add(order_id)
            self.increment_metric("total_orders")
            logger.debug(f"Order added: {order_id}")
    
    def remove_active_order(self, order_id: str) -> None:
        """Remove order from active set."""
        with self._locks["orders"]:
            self._active_orders.discard(order_id)
            logger.debug(f"Order removed: {order_id}")
    
    def is_order_active(self, order_id: str) -> bool:
        """Check if order is active."""
        with self._locks["orders"]:
            return order_id in self._active_orders
    
    def get_active_orders(self) -> Set[str]:
        """Get copy of active orders."""
        with self._locks["orders"]:
            return self._active_orders.copy()
    
    def record_fill(
        self,
        order_id: str,
        volume: Union[float, Decimal],
        commission: Union[float, Decimal],
        slippage: Union[float, Decimal]
    ) -> None:
        """Record fill metrics."""
        with self._locks["metrics"]:
            self._metrics.add_fill(volume, commission, slippage)
            self._metrics.filled_orders += 1
            logger.debug(
                f"Fill recorded: {order_id} - volume: {volume}, "
                f"commission: {commission}, slippage: {slippage}"
            )
    
    def record_order_status(self, status: str) -> None:
        """Record order status update."""
        with self._locks["metrics"]:
            if status == "cancelled":
                self._metrics.cancelled_orders += 1
            elif status == "rejected":
                self._metrics.rejected_orders += 1
    
    def increment_metric(self, metric: str, value: Union[float, Decimal] = 1) -> None:
        """Increment a metric value."""
        with self._locks["metrics"]:
            if hasattr(self._metrics, metric):
                current = getattr(self._metrics, metric)
                # Convert value to appropriate type for arithmetic
                if isinstance(current, Decimal):
                    value_decimal = Decimal(str(value)) if not isinstance(value, Decimal) else value
                    setattr(self._metrics, metric, current + value_decimal)
                else:
                    # For integer metrics like total_orders, total_fills
                    setattr(self._metrics, metric, current + value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._locks["metrics"]:
            return self._metrics.to_dict()
    
    def acquire_lock(self, resource: str) -> threading.Lock:
        """Acquire or create a resource lock."""
        with self._global_lock:
            if resource not in self._locks:
                self._locks[resource] = threading.Lock()
            return self._locks[resource]
    
    def execute_atomic(self, func, *args, **kwargs):
        """Execute function atomically."""
        with self._global_lock:
            return func(*args, **kwargs)
    
    def get_thread_id(self) -> int:
        """Get current thread ID."""
        return threading.get_ident()
    
    def get_transaction_id(self) -> Optional[str]:
        """Get current transaction ID if in transaction."""
        return getattr(self._thread_locals, "transaction_id", None)
    
    def reset(self) -> None:
        """Reset context state."""
        with self._global_lock:
            self._state.clear()
            self._metrics = ExecutionMetrics()
            self._active_orders.clear()
            logger.info("ExecutionContext reset")