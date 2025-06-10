"""
Synchronous order manager.

Simple, fast order tracking without async complexity.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

from ..types import Order, Fill, OrderStatus

logger = logging.getLogger(__name__)


class SyncOrderManager:
    """
    Synchronous order manager.
    
    Provides fast order tracking and status management
    without async overhead.
    """
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.logger = logger.getChild(component_id)
        
        # Simple dictionaries for fast lookups
        self._orders: Dict[str, Order] = {}
        self._fills: Dict[str, Fill] = {}
        self._order_status: Dict[str, OrderStatus] = {}
        self._broker_order_ids: Dict[str, str] = {}  # order_id -> broker_order_id
        self._fills_by_order: Dict[str, List[Fill]] = defaultdict(list)
        
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize order manager."""
        if self._initialized:
            return
        
        self.logger.info(f"Order manager {self.component_id} initialized")
        self._initialized = True
    
    def track_pending_order(self, order: Order, broker_order_id: str) -> None:
        """Track a pending order."""
        self._orders[order.order_id] = order
        self._order_status[order.order_id] = OrderStatus.SUBMITTED
        self._broker_order_ids[order.order_id] = broker_order_id
        
        self.logger.debug(f"Tracking pending order: {order.order_id}")
    
    def process_fill(self, fill: Fill) -> None:
        """Process an order fill."""
        self._fills[fill.fill_id] = fill
        self._fills_by_order[fill.order_id].append(fill)
        
        # Update order status
        order = self._orders.get(fill.order_id)
        if order:
            # Check if order is fully filled
            total_filled = sum(f.quantity for f in self._fills_by_order[fill.order_id])
            
            if total_filled >= order.quantity:
                self._order_status[fill.order_id] = OrderStatus.FILLED
            else:
                self._order_status[fill.order_id] = OrderStatus.PARTIALLY_FILLED
        
        self.logger.debug(f"Processed fill: {fill.fill_id} for order {fill.order_id}")
    
    def cancel_order(self, order_id: str) -> None:
        """Cancel an order."""
        if order_id in self._order_status:
            self._order_status[order_id] = OrderStatus.CANCELLED
            self.logger.debug(f"Cancelled order: {order_id}")
    
    def reject_order(self, order_id: str, reason: str) -> None:
        """Reject an order."""
        if order_id in self._orders:
            self._order_status[order_id] = OrderStatus.REJECTED
        
        # Log rejection reason
        self.logger.warning(f"Rejected order {order_id}: {reason}")
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        return self._order_status.get(order_id)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    def get_fill(self, fill_id: str) -> Optional[Fill]:
        """Get fill by ID."""
        return self._fills.get(fill_id)
    
    def get_fills_for_order(self, order_id: str) -> List[Fill]:
        """Get all fills for an order."""
        return self._fills_by_order[order_id].copy()
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        pending_orders = []
        
        for order_id, status in self._order_status.items():
            if status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                order = self._orders.get(order_id)
                if order:
                    pending_orders.append(order)
        
        return pending_orders
    
    def get_filled_orders(self) -> List[Order]:
        """Get all filled orders."""
        filled_orders = []
        
        for order_id, status in self._order_status.items():
            if status == OrderStatus.FILLED:
                order = self._orders.get(order_id)
                if order:
                    filled_orders.append(order)
        
        return filled_orders
    
    def get_fills(self) -> List[Fill]:
        """Get all fills."""
        return list(self._fills.values())
    
    def get_orders(self) -> List[Order]:
        """Get all orders."""
        return list(self._orders.values())
    
    def get_broker_order_id(self, order_id: str) -> Optional[str]:
        """Get broker order ID for internal order ID."""
        return self._broker_order_ids.get(order_id)
    
    def get_stats(self) -> Dict[str, int]:
        """Get order manager statistics."""
        status_counts = defaultdict(int)
        
        for status in self._order_status.values():
            status_counts[status.value] += 1
        
        return {
            'total_orders': len(self._orders),
            'total_fills': len(self._fills),
            'pending': status_counts[OrderStatus.PENDING.value],
            'submitted': status_counts[OrderStatus.SUBMITTED.value],
            'filled': status_counts[OrderStatus.FILLED.value],
            'partially_filled': status_counts[OrderStatus.PARTIALLY_FILLED.value],
            'cancelled': status_counts[OrderStatus.CANCELLED.value],
            'rejected': status_counts[OrderStatus.REJECTED.value],
        }
    
    def clear_history(self) -> None:
        """Clear order and fill history."""
        self._orders.clear()
        self._fills.clear()
        self._order_status.clear()
        self._broker_order_ids.clear()
        self._fills_by_order.clear()
        
        self.logger.info("Order history cleared")
    
    def shutdown(self) -> None:
        """Shutdown order manager."""
        if not self._initialized:
            return
        
        stats = self.get_stats()
        self.logger.info(f"Order manager {self.component_id} shutdown. Final stats: {stats}")
        self._initialized = False