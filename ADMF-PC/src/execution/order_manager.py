"""Order lifecycle management."""

import asyncio
from typing import Dict, Optional, List, Set
from datetime import datetime, timedelta
import uuid
from collections import defaultdict

from .protocols import (
    Order, OrderStatus, OrderProcessor, Fill,
    OrderType, OrderSide
)
from ..core.logging.structured import get_logger


logger = get_logger(__name__)


class OrderManager:
    """Manages order lifecycle and state transitions."""
    
    def __init__(self, max_order_age: timedelta = timedelta(days=1)):
        """Initialize order manager."""
        self.orders: Dict[str, Order] = {}
        self.order_status: Dict[str, OrderStatus] = {}
        self.order_fills: Dict[str, List[Fill]] = defaultdict(list)
        self.pending_orders: Set[str] = set()
        self.active_orders: Set[str] = set()
        self.max_order_age = max_order_age
        self._lock = asyncio.Lock()
        
        logger.info("Initialized OrderManager")
    
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> Order:
        """Create new order."""
        order_id = str(uuid.uuid4())
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            metadata=metadata or {}
        )
        
        async with self._lock:
            self.orders[order_id] = order
            self.order_status[order_id] = OrderStatus.PENDING
            self.pending_orders.add(order_id)
        
        logger.info(
            f"Created order: {order_id} - "
            f"{side.name} {quantity} {symbol} @ {order_type.name}"
        )
        
        return order
    
    async def submit_order(self, order_id: str) -> bool:
        """Submit order for execution."""
        async with self._lock:
            if order_id not in self.orders:
                logger.error(f"Order not found: {order_id}")
                return False
            
            if self.order_status[order_id] != OrderStatus.PENDING:
                logger.error(
                    f"Cannot submit order {order_id} with status "
                    f"{self.order_status[order_id]}"
                )
                return False
            
            self.order_status[order_id] = OrderStatus.SUBMITTED
            self.pending_orders.discard(order_id)
            self.active_orders.add(order_id)
            
            logger.info(f"Order submitted: {order_id}")
            return True
    
    async def update_order_status(
        self,
        order_id: str,
        status: OrderStatus
    ) -> bool:
        """Update order status."""
        async with self._lock:
            if order_id not in self.orders:
                logger.error(f"Order not found: {order_id}")
                return False
            
            old_status = self.order_status[order_id]
            self.order_status[order_id] = status
            
            # Update tracking sets
            if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                self.active_orders.discard(order_id)
                self.pending_orders.discard(order_id)
            elif status == OrderStatus.SUBMITTED:
                self.pending_orders.discard(order_id)
                self.active_orders.add(order_id)
            
            logger.info(f"Order {order_id} status: {old_status} -> {status}")
            return True
    
    async def add_fill(self, order_id: str, fill: Fill) -> bool:
        """Add fill to order."""
        async with self._lock:
            if order_id not in self.orders:
                logger.error(f"Order not found: {order_id}")
                return False
            
            self.order_fills[order_id].append(fill)
            
            # Check if order is fully filled
            order = self.orders[order_id]
            total_filled = sum(f.quantity for f in self.order_fills[order_id])
            
            if total_filled >= order.quantity:
                await self.update_order_status(order_id, OrderStatus.FILLED)
            else:
                await self.update_order_status(order_id, OrderStatus.PARTIAL)
            
            logger.info(
                f"Fill added to order {order_id}: "
                f"{fill.quantity} @ {fill.price}"
            )
            return True
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        async with self._lock:
            if order_id not in self.orders:
                logger.error(f"Order not found: {order_id}")
                return False
            
            status = self.order_status[order_id]
            if status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                logger.warning(
                    f"Cannot cancel order {order_id} with status {status}"
                )
                return False
            
            await self.update_order_status(order_id, OrderStatus.CANCELLED)
            logger.info(f"Order cancelled: {order_id}")
            return True
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        return self.order_status.get(order_id)
    
    async def get_order_fills(self, order_id: str) -> List[Fill]:
        """Get fills for order."""
        return self.order_fills.get(order_id, [])
    
    async def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        async with self._lock:
            return [
                self.orders[order_id]
                for order_id in self.active_orders
                if order_id in self.orders
            ]
    
    async def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        async with self._lock:
            return [
                self.orders[order_id]
                for order_id in self.pending_orders
                if order_id in self.orders
            ]
    
    async def validate_order(self, order: Order) -> bool:
        """Validate order before processing."""
        # Basic validation
        if order.quantity <= 0:
            logger.error(f"Invalid quantity: {order.quantity}")
            return False
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None or order.price <= 0:
                logger.error(f"Invalid limit price: {order.price}")
                return False
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None or order.stop_price <= 0:
                logger.error(f"Invalid stop price: {order.stop_price}")
                return False
        
        return True
    
    async def cleanup_old_orders(self) -> int:
        """Clean up old completed orders."""
        async with self._lock:
            now = datetime.now()
            orders_to_remove = []
            
            for order_id, order in self.orders.items():
                status = self.order_status[order_id]
                if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    if now - order.created_at > self.max_order_age:
                        orders_to_remove.append(order_id)
            
            for order_id in orders_to_remove:
                del self.orders[order_id]
                del self.order_status[order_id]
                if order_id in self.order_fills:
                    del self.order_fills[order_id]
            
            if orders_to_remove:
                logger.info(f"Cleaned up {len(orders_to_remove)} old orders")
            
            return len(orders_to_remove)
    
    def get_order_summary(self) -> Dict[str, int]:
        """Get order summary statistics."""
        summary = defaultdict(int)
        for status in self.order_status.values():
            summary[status.name] += 1
        
        return {
            "total_orders": len(self.orders),
            "active_orders": len(self.active_orders),
            "pending_orders": len(self.pending_orders),
            **dict(summary)
        }