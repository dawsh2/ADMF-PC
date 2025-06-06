"""Canonical order lifecycle management implementation for ADMF-PC.

This unified order manager follows Protocol + Composition principles
with proper dependency injection and comprehensive validation.
"""

import asyncio
from typing import Dict, Optional, List, Set, Any
from datetime import datetime, timedelta
import uuid
from collections import defaultdict
import logging

from ..core.components.protocols import Component, Lifecycle
from .protocols import (
    Order, OrderStatus, OrderProcessor, Fill,
    OrderType, OrderSide
)
from .engine import ValidationResult


logger = logging.getLogger(__name__)


class OrderManager(Component, Lifecycle, OrderProcessor):
    """Canonical order manager implementation using Protocol + Composition.
    
    This order manager:
    - Uses dependency injection for clean separation of concerns
    - Follows Component/Lifecycle protocols for ADMF-PC compliance
    - Supports comprehensive order validation and state management
    - Handles both backtesting and live trading through configuration
    """
    
    def __init__(
        self,
        component_id: str = None,
        max_order_age: timedelta = timedelta(days=1),
        validation_enabled: bool = True
    ):
        """Initialize order manager with dependency injection.
        
        Args:
            component_id: Unique component identifier
            max_order_age: Maximum age for completed orders before cleanup
            validation_enabled: Whether to enable order validation
        """
        self._component_id = component_id or f"order_manager_{uuid.uuid4().hex[:8]}"
        self._max_order_age = max_order_age
        self._validation_enabled = validation_enabled
        
        # Order storage
        self._orders: Dict[str, Order] = {}
        self._order_status: Dict[str, OrderStatus] = {}
        self._order_fills: Dict[str, List[Fill]] = defaultdict(list)
        self._pending_orders: Set[str] = set()
        self._active_orders: Set[str] = set()
        
        # Async thread safety
        self._order_lock = asyncio.Lock()
        
        # Metrics tracking
        self._order_stats = {
            'total_created': 0,
            'total_submitted': 0,
            'total_filled': 0,
            'total_cancelled': 0,
            'total_rejected': 0,
            'validation_failures': 0
        }
        
        # Lifecycle state
        self._initialized = False
        self._running = False
        
        logger.info(f"OrderManager initialized - ID: {self._component_id}")
    
    @property
    def component_id(self) -> str:
        """Get component ID."""
        return self._component_id
    
    # Lifecycle methods
    async def initialize(self) -> None:
        """Initialize the order manager."""
        if self._initialized:
            logger.warning(f"OrderManager {self._component_id} already initialized")
            return
        
        self._initialized = True
        logger.info(f"OrderManager {self._component_id} initialized successfully")
    
    async def start(self) -> None:
        """Start the order manager."""
        if not self._initialized:
            await self.initialize()
        
        if self._running:
            logger.warning(f"OrderManager {self._component_id} already running")
            return
        
        self._running = True
        logger.info(f"OrderManager {self._component_id} started")
    
    async def stop(self) -> None:
        """Stop the order manager."""
        if not self._running:
            logger.warning(f"OrderManager {self._component_id} not running")
            return
        
        self._running = False
        
        # Cancel all active orders
        await self._cancel_all_active_orders()
        
        logger.info(f"OrderManager {self._component_id} stopped")
    
    async def reset(self) -> None:
        """Reset order manager state."""
        # Clear all orders and state
        async with self._order_lock:
            self._orders.clear()
            self._order_status.clear()
            self._order_fills.clear()
            self._pending_orders.clear()
            self._active_orders.clear()
        
        # Reset metrics
        self._order_stats = {
            'total_created': 0,
            'total_submitted': 0,
            'total_filled': 0,
            'total_cancelled': 0,
            'total_rejected': 0,
            'validation_failures': 0
        }
        
        logger.info(f"OrderManager {self._component_id} reset")
    
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Order:
        """Create new order with comprehensive validation."""
        if not self._running:
            raise RuntimeError("OrderManager not running")
        
        order_id = str(uuid.uuid4())
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        # Validate order if enabled
        if self._validation_enabled:
            validation_result = await self._validate_order_comprehensive(order)
            if not validation_result.is_valid:
                self._order_stats['validation_failures'] += 1
                raise ValueError(f"Order validation failed: {validation_result.reason}")
        
        async with self._order_lock:
            # Store order
            self._orders[order_id] = order
            self._order_status[order_id] = OrderStatus.PENDING
            self._pending_orders.add(order_id)
            self._order_stats['total_created'] += 1
        
        logger.info(
            f"Order created - ID: {order_id}, "
            f"{side.name} {quantity} {symbol} @ {order_type.name}"
        )
        
        return order
    
    async def submit_order(self, order_id: str) -> bool:
        """Submit order for execution with proper validation."""
        if not self._running:
            logger.warning("OrderManager not running, cannot submit order")
            return False
        
        async with self._order_lock:
            if order_id not in self._orders:
                logger.error(f"Order not found: {order_id}")
                return False
            
            current_status = self._order_status.get(order_id)
            if current_status != OrderStatus.PENDING:
                logger.error(
                    f"Cannot submit order {order_id} with status {current_status}"
                )
                return False
            
            # Update status and tracking
            self._order_status[order_id] = OrderStatus.SUBMITTED
            self._pending_orders.discard(order_id)
            self._active_orders.add(order_id)
            self._order_stats['total_submitted'] += 1
            
            logger.info(f"Order submitted - ID: {order_id}")
            return True
    
    async def update_order_status(
        self,
        order_id: str,
        status: OrderStatus
    ) -> bool:
        """Update order status with proper state transitions."""
        if not self._running:
            logger.warning("OrderManager not running, cannot update status")
            return False
        
        async with self._order_lock:
            if order_id not in self._orders:
                logger.error(f"Order not found for status update: {order_id}")
                return False
            
            old_status = self._order_status.get(order_id)
            
            # Validate state transition
            if not self._is_valid_status_transition(old_status, status):
                logger.error(
                    f"Invalid status transition for order {order_id}: "
                    f"{old_status} -> {status}"
                )
                return False
            
            # Update status
            self._order_status[order_id] = status
            
            # Update tracking sets and metrics
            self._update_tracking_sets(order_id, status)
            self._update_status_metrics(status)
            
            logger.info(f"Order {order_id} status updated: {old_status} -> {status}")
            return True
    
    async def add_fill(self, order_id: str, fill: Fill) -> bool:
        """Add fill to order with comprehensive validation."""
        if not self._running:
            logger.warning("OrderManager not running, cannot add fill")
            return False
        
        async with self._order_lock:
            if order_id not in self._orders:
                logger.error(f"Order not found for fill: {order_id}")
                return False
            
            order = self._orders[order_id]
            
            # Validate fill
            validation_result = await self._validate_fill(order, fill)
            if not validation_result.is_valid:
                logger.error(f"Fill validation failed: {validation_result.reason}")
                return False
            
            # Add fill
            self._order_fills[order_id].append(fill)
            
            # Calculate fill status
            total_filled = sum(f.quantity for f in self._order_fills[order_id])
            
            if total_filled >= order.quantity:
                await self.update_order_status(order_id, OrderStatus.FILLED)
            else:
                await self.update_order_status(order_id, OrderStatus.PARTIAL)
            
            logger.info(
                f"Fill added to order {order_id}: "
                f"{fill.quantity} @ {fill.price} "
                f"(total filled: {total_filled}/{order.quantity})"
            )
            return True
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with proper validation."""
        if not self._running:
            logger.warning("OrderManager not running, cannot cancel order")
            return False
        
        async with self._order_lock:
            if order_id not in self._orders:
                logger.error(f"Order not found for cancellation: {order_id}")
                return False
            
            current_status = self._order_status.get(order_id)
            
            # Check if order can be cancelled
            if current_status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                logger.warning(
                    f"Cannot cancel order {order_id} with status {current_status}"
                )
                return False
            
            # Update status
            await self.update_order_status(order_id, OrderStatus.CANCELLED)
            
            logger.info(f"Order cancelled - ID: {order_id}")
            return True
    
    # OrderProcessor interface methods
    async def validate_order(self, order: Order) -> bool:
        """Validate order (OrderProcessor interface)."""
        validation_result = await self._validate_order_comprehensive(order)
        return validation_result.is_valid
    
    async def process_order(self, order: Order) -> Optional[Fill]:
        """Process order (OrderProcessor interface) - delegates to execution engine."""
        # This method is part of the OrderProcessor interface
        # but actual order processing is handled by the execution engine
        logger.debug(f"Order processing delegated for order: {order.order_id}")
        return None
    
    # Query methods
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        return self._order_status.get(order_id)
    
    async def get_order_fills(self, order_id: str) -> List[Fill]:
        """Get fills for order."""
        return self._order_fills.get(order_id, []).copy()
    
    async def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        async with self._order_lock:
            return [
                self._orders[order_id]
                for order_id in self._active_orders
                if order_id in self._orders
            ]
    
    async def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        async with self._order_lock:
            return [
                self._orders[order_id]
                for order_id in self._pending_orders
                if order_id in self._orders
            ]
    
    async def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        async with self._order_lock:
            return [
                order for order in self._orders.values()
                if order.symbol == symbol
            ]
    
    async def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get all orders with specific status."""
        async with self._order_lock:
            return [
                self._orders[order_id]
                for order_id, order_status in self._order_status.items()
                if order_status == status and order_id in self._orders
            ]
    
    # Maintenance and statistics
    async def cleanup_old_orders(self) -> int:
        """Clean up old completed orders."""
        if not self._running:
            return 0
        
        async with self._order_lock:
            now = datetime.now()
            orders_to_remove = []
            
            for order_id, order in self._orders.items():
                status = self._order_status[order_id]
                if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    if now - order.created_at > self._max_order_age:
                        orders_to_remove.append(order_id)
            
            # Remove old orders
            for order_id in orders_to_remove:
                del self._orders[order_id]
                del self._order_status[order_id]
                if order_id in self._order_fills:
                    del self._order_fills[order_id]
                # Remove from tracking sets (should already be removed)
                self._pending_orders.discard(order_id)
                self._active_orders.discard(order_id)
            
            if orders_to_remove:
                logger.info(f"Cleaned up {len(orders_to_remove)} old orders")
            
            return len(orders_to_remove)
    
    def get_order_summary(self) -> Dict[str, Any]:
        """Get comprehensive order summary statistics."""
        status_counts = defaultdict(int)
        for status in self._order_status.values():
            status_counts[status.name] += 1
        
        return {
            "component_id": self._component_id,
            "running": self._running,
            "total_orders": len(self._orders),
            "active_orders": len(self._active_orders),
            "pending_orders": len(self._pending_orders),
            "order_stats": self._order_stats.copy(),
            "status_breakdown": dict(status_counts),
            "avg_fills_per_order": (
                sum(len(fills) for fills in self._order_fills.values()) / len(self._orders)
                if self._orders else 0
            )
        }
    
    # Private validation and helper methods
    async def _validate_order_comprehensive(self, order: Order) -> ValidationResult:
        """Comprehensive order validation."""
        # Basic validation
        if order.quantity <= 0:
            return ValidationResult(False, f"Invalid quantity: {order.quantity}")
        
        if not order.symbol or not order.symbol.strip():
            return ValidationResult(False, "Invalid symbol")
        
        # Order type specific validation
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None or order.price <= 0:
                return ValidationResult(False, f"Invalid limit price: {order.price}")
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None or order.stop_price <= 0:
                return ValidationResult(False, f"Invalid stop price: {order.stop_price}")
        
        # Stop-limit price relationship validation
        if order.order_type == OrderType.STOP_LIMIT:
            if order.side == OrderSide.BUY:
                if order.stop_price > order.price:
                    return ValidationResult(
                        False, 
                        "Buy stop-limit: stop price must be <= limit price"
                    )
            else:  # SELL
                if order.stop_price < order.price:
                    return ValidationResult(
                        False, 
                        "Sell stop-limit: stop price must be >= limit price"
                    )
        
        return ValidationResult(True, "Valid")
    
    async def _validate_fill(self, order: Order, fill: Fill) -> ValidationResult:
        """Validate fill against order."""
        # Basic checks
        if fill.order_id != order.order_id:
            return ValidationResult(False, "Fill order ID mismatch")
        
        if fill.symbol != order.symbol:
            return ValidationResult(False, "Fill symbol mismatch")
        
        if fill.side != order.side:
            return ValidationResult(False, "Fill side mismatch")
        
        if fill.quantity <= 0:
            return ValidationResult(False, f"Invalid fill quantity: {fill.quantity}")
        
        if fill.price <= 0:
            return ValidationResult(False, f"Invalid fill price: {fill.price}")
        
        # Check against existing fills
        existing_fills = self._order_fills.get(order.order_id, [])
        total_filled = sum(f.quantity for f in existing_fills)
        
        if total_filled + fill.quantity > order.quantity * 1.001:  # Allow small rounding
            return ValidationResult(
                False, 
                f"Over-fill: {total_filled + fill.quantity} > {order.quantity}"
            )
        
        return ValidationResult(True, "Valid")
    
    def _is_valid_status_transition(
        self, 
        old_status: Optional[OrderStatus], 
        new_status: OrderStatus
    ) -> bool:
        """Validate order status transitions."""
        if old_status is None:
            return new_status == OrderStatus.PENDING
        
        # Define valid transitions
        valid_transitions = {
            OrderStatus.PENDING: [OrderStatus.SUBMITTED, OrderStatus.CANCELLED, OrderStatus.REJECTED],
            OrderStatus.SUBMITTED: [OrderStatus.PARTIAL, OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED],
            OrderStatus.PARTIAL: [OrderStatus.FILLED, OrderStatus.CANCELLED],
            OrderStatus.FILLED: [],  # Terminal state
            OrderStatus.CANCELLED: [],  # Terminal state
            OrderStatus.REJECTED: []  # Terminal state
        }
        
        return new_status in valid_transitions.get(old_status, [])
    
    def _update_tracking_sets(self, order_id: str, status: OrderStatus) -> None:
        """Update tracking sets based on status."""
        if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self._active_orders.discard(order_id)
            self._pending_orders.discard(order_id)
        elif status == OrderStatus.SUBMITTED:
            self._pending_orders.discard(order_id)
            self._active_orders.add(order_id)
    
    def _update_status_metrics(self, status: OrderStatus) -> None:
        """Update metrics based on status."""
        if status == OrderStatus.FILLED:
            self._order_stats['total_filled'] += 1
        elif status == OrderStatus.CANCELLED:
            self._order_stats['total_cancelled'] += 1
        elif status == OrderStatus.REJECTED:
            self._order_stats['total_rejected'] += 1
    
    async def _cancel_all_active_orders(self) -> None:
        """Cancel all active orders during shutdown."""
        async with self._order_lock:
            order_ids = list(self._active_orders) + list(self._pending_orders)
            
            for order_id in order_ids:
                try:
                    await self.cancel_order(order_id)
                except Exception as e:
                    logger.error(f"Error cancelling order {order_id} during shutdown: {e}")


# Factory function for creating order managers (follows Protocol + Composition)
def create_order_manager(
    component_id: str = None,
    max_order_age_hours: int = 24,
    validation_enabled: bool = True
) -> OrderManager:
    """
    Factory function for creating order manager instances.
    
    Args:
        component_id: Unique component identifier
        max_order_age_hours: Maximum age for completed orders before cleanup
        validation_enabled: Whether to enable order validation
        
    Returns:
        Configured OrderManager instance
    """
    return OrderManager(
        component_id=component_id,
        max_order_age=timedelta(hours=max_order_age_hours),
        validation_enabled=validation_enabled
    )