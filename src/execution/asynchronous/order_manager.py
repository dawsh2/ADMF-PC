"""
Asynchronous order manager.

Real-time order tracking with async operations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque

from ..types import Order, Fill, OrderStatus

logger = logging.getLogger(__name__)


class AsyncOrderManager:
    """
    Asynchronous order manager.
    
    Provides real-time order tracking and status management
    with proper async patterns for concurrent access.
    """
    
    def __init__(
        self, 
        component_id: str,
        max_fill_history: int = 10000,
        cleanup_interval: int = 3600  # 1 hour
    ):
        self.component_id = component_id
        self.max_fill_history = max_fill_history
        self.cleanup_interval = cleanup_interval
        
        self.logger = logger.getChild(component_id)
        
        # Async locks for thread safety
        self._order_lock = asyncio.Lock()
        self._fill_lock = asyncio.Lock()
        
        # Order tracking
        self._orders: Dict[str, Order] = {}
        self._order_status: Dict[str, OrderStatus] = {}
        self._broker_order_ids: Dict[str, str] = {}  # order_id -> broker_order_id
        self._broker_to_internal: Dict[str, str] = {}  # broker_order_id -> order_id
        
        # Fill tracking
        self._fills: deque = deque(maxlen=max_fill_history)
        self._fills_by_order: Dict[str, List[Fill]] = defaultdict(list)
        self._processed_fill_ids: Set[str] = set()
        
        # Background tasks
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the order manager."""
        if self._running:
            return
        
        self.logger.info(f"Starting order manager: {self.component_id}")
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_old_data())
        self._running = True
        
        self.logger.info("Order manager started")
    
    async def stop(self) -> None:
        """Stop the order manager."""
        if not self._running:
            return
        
        self.logger.info("Stopping order manager")
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self._running = False
        self.logger.info("Order manager stopped")
    
    async def track_order(self, order: Order, broker_order_id: str) -> None:
        """Track a new order."""
        async with self._order_lock:
            self._orders[order.order_id] = order
            self._order_status[order.order_id] = OrderStatus.SUBMITTED
            self._broker_order_ids[order.order_id] = broker_order_id
            self._broker_to_internal[broker_order_id] = order.order_id
        
        self.logger.debug(f"Tracking order: {order.order_id} -> broker ID: {broker_order_id}")
    
    async def update_order_status(self, order_id: str, status: str) -> None:
        """Update order status."""
        async with self._order_lock:
            if order_id in self._order_status:
                # Convert string status to enum
                try:
                    status_enum = OrderStatus(status.lower())
                    self._order_status[order_id] = status_enum
                    self.logger.debug(f"Order status updated: {order_id} -> {status}")
                except ValueError:
                    self.logger.warning(f"Invalid order status: {status}")
    
    async def process_fill(self, fill: Fill) -> None:
        """Process an order fill."""
        # Check for duplicate fills
        if fill.fill_id in self._processed_fill_ids:
            self.logger.debug(f"Duplicate fill ignored: {fill.fill_id}")
            return
        
        async with self._fill_lock:
            # Add to fill tracking
            self._fills.append(fill)
            self._fills_by_order[fill.order_id].append(fill)
            self._processed_fill_ids.add(fill.fill_id)
        
        # Update order status
        async with self._order_lock:
            order = self._orders.get(fill.order_id)
            if order:
                # Calculate total filled quantity
                total_filled = sum(
                    f.quantity for f in self._fills_by_order[fill.order_id]
                )
                
                if total_filled >= order.quantity:
                    self._order_status[fill.order_id] = OrderStatus.FILLED
                else:
                    self._order_status[fill.order_id] = OrderStatus.PARTIALLY_FILLED
        
        self.logger.debug(f"Fill processed: {fill.fill_id} for order {fill.order_id}")
    
    async def cancel_order(self, order_id: str) -> None:
        """Cancel an order."""
        async with self._order_lock:
            if order_id in self._order_status:
                self._order_status[order_id] = OrderStatus.CANCELLED
                self.logger.debug(f"Order cancelled: {order_id}")
    
    async def reject_order(self, order_id: str, reason: str) -> None:
        """Reject an order."""
        async with self._order_lock:
            if order_id in self._orders:
                self._order_status[order_id] = OrderStatus.REJECTED
        
        self.logger.warning(f"Order rejected: {order_id} - {reason}")
    
    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        async with self._order_lock:
            return self._order_status.get(order_id)
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        async with self._order_lock:
            return self._orders.get(order_id)
    
    async def get_broker_order_id(self, order_id: str) -> Optional[str]:
        """Get broker order ID for internal order ID."""
        async with self._order_lock:
            return self._broker_order_ids.get(order_id)
    
    async def get_internal_order_id(self, broker_order_id: str) -> Optional[str]:
        """Get internal order ID for broker order ID."""
        async with self._order_lock:
            return self._broker_to_internal.get(broker_order_id)
    
    async def get_fills_for_order(self, order_id: str) -> List[Fill]:
        """Get all fills for an order."""
        async with self._fill_lock:
            return self._fills_by_order[order_id].copy()
    
    async def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        async with self._order_lock:
            pending_orders = []
            
            for order_id, status in self._order_status.items():
                if status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                    order = self._orders.get(order_id)
                    if order:
                        pending_orders.append(order)
            
            return pending_orders
    
    async def get_filled_orders(self) -> List[Order]:
        """Get all filled orders."""
        async with self._order_lock:
            filled_orders = []
            
            for order_id, status in self._order_status.items():
                if status == OrderStatus.FILLED:
                    order = self._orders.get(order_id)
                    if order:
                        filled_orders.append(order)
            
            return filled_orders
    
    async def get_recent_fills(self, limit: int = 100) -> List[Fill]:
        """Get recent fills."""
        async with self._fill_lock:
            # Return most recent fills up to limit
            recent_fills = list(self._fills)
            return recent_fills[-limit:] if len(recent_fills) > limit else recent_fills
    
    async def get_all_orders(self) -> List[Order]:
        """Get all orders."""
        async with self._order_lock:
            return list(self._orders.values())
    
    async def get_stats(self) -> Dict[str, int]:
        """Get order manager statistics."""
        async with self._order_lock:
            status_counts = defaultdict(int)
            
            for status in self._order_status.values():
                status_counts[status.value] += 1
        
        async with self._fill_lock:
            total_fills = len(self._fills)
        
        return {
            'total_orders': len(self._orders),
            'total_fills': total_fills,
            'pending': status_counts[OrderStatus.PENDING.value],
            'submitted': status_counts[OrderStatus.SUBMITTED.value],
            'filled': status_counts[OrderStatus.FILLED.value],
            'partially_filled': status_counts[OrderStatus.PARTIALLY_FILLED.value],
            'cancelled': status_counts[OrderStatus.CANCELLED.value],
            'rejected': status_counts[OrderStatus.REJECTED.value],
        }
    
    async def clear_history(self, keep_pending: bool = True) -> None:
        """Clear order and fill history."""
        async with self._order_lock, self._fill_lock:
            if keep_pending:
                # Keep only pending orders
                pending_order_ids = {
                    order_id for order_id, status in self._order_status.items()
                    if status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
                }
                
                # Filter orders
                self._orders = {
                    order_id: order for order_id, order in self._orders.items()
                    if order_id in pending_order_ids
                }
                self._order_status = {
                    order_id: status for order_id, status in self._order_status.items()
                    if order_id in pending_order_ids
                }
                self._broker_order_ids = {
                    order_id: broker_id for order_id, broker_id in self._broker_order_ids.items()
                    if order_id in pending_order_ids
                }
                
                # Update reverse mapping
                self._broker_to_internal = {
                    broker_id: order_id for order_id, broker_id in self._broker_order_ids.items()
                }
                
                # Clear fills for non-pending orders
                self._fills_by_order = {
                    order_id: fills for order_id, fills in self._fills_by_order.items()
                    if order_id in pending_order_ids
                }
            else:
                # Clear everything
                self._orders.clear()
                self._order_status.clear()
                self._broker_order_ids.clear()
                self._broker_to_internal.clear()
                self._fills_by_order.clear()
            
            # Clear fills and processed IDs
            self._fills.clear()
            self._processed_fill_ids.clear()
        
        self.logger.info("Order history cleared")
    
    async def _cleanup_old_data(self) -> None:
        """Periodically clean up old data."""
        self.logger.debug("Starting data cleanup task")
        
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                if not self._running:
                    break
                
                # Clean up old processed fill IDs (keep last 24 hours worth)
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                async with self._fill_lock:
                    # This is a simple cleanup - in practice you might want
                    # to track fill timestamps for more precise cleanup
                    if len(self._processed_fill_ids) > 50000:  # Arbitrary limit
                        # Clear half of the old IDs
                        old_ids = list(self._processed_fill_ids)[:25000]
                        for fill_id in old_ids:
                            self._processed_fill_ids.discard(fill_id)
                        
                        self.logger.debug("Cleaned up old fill IDs")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in data cleanup: {e}")
        
        self.logger.debug("Data cleanup task stopped")