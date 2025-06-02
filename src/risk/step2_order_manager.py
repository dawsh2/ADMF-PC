"""
File: src/risk/step2_order_manager.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#order-management
Step: 2 - Add Risk Container
Dependencies: core.logging, risk.models

Order management for Step 2 risk container.
Creates and tracks orders from trading signals.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime
from decimal import Decimal
import uuid

from ..core.logging.structured import ContainerLogger
from .models import TradingSignal, Order, OrderSide, OrderType


class OrderManager:
    """
    Manages order creation and tracking for Step 2.
    
    Creates risk-adjusted orders from trading signals and tracks
    order lifecycle for portfolio management.
    
    Architecture Context:
        - Part of: Step 2 - Add Risk Container
        - Implements: Protocol-based order management without inheritance
        - Provides: Signal-to-order transformation with risk metadata
        - Dependencies: Structured logging for audit trail
    
    Example:
        manager = OrderManager("risk_001")
        order = manager.create_order(signal, size, current_prices)
    """
    
    def __init__(self, container_id: str):
        """
        Initialize order manager.
        
        Args:
            container_id: Container ID for logging context
        """
        self.container_id = container_id
        
        # Order tracking
        self.created_orders: List[Order] = []
        self.order_count = 0
        
        # Logging
        self.logger = ContainerLogger("OrderManager", container_id, "order_manager")
        
        self.logger.info(
            "OrderManager initialized",
            container_id=container_id
        )
    
    def create_order(
        self, 
        signal: TradingSignal, 
        size: Decimal, 
        current_prices: Dict[str, float]
    ) -> Optional[Order]:
        """
        Create order from trading signal.
        
        Args:
            signal: Trading signal to convert
            size: Position size in shares/units
            current_prices: Current market prices
            
        Returns:
            Created order or None if creation failed
        """
        self.logger.trace(
            "Creating order from signal",
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=signal.side.value,
            size=float(size)
        )
        
        # Validate inputs
        if size <= 0:
            self.logger.warning(
                "Cannot create order with zero or negative size",
                signal_id=signal.signal_id,
                size=float(size)
            )
            return None
        
        current_price = current_prices.get(signal.symbol)
        if not current_price:
            self.logger.warning(
                "No current price available for order creation",
                symbol=signal.symbol
            )
            return None
        
        # Generate order ID
        order_id = self._generate_order_id()
        
        # Create order
        order = Order(
            order_id=order_id,
            symbol=signal.symbol,
            side=signal.side,
            order_type=OrderType.MARKET,  # Step 2 uses market orders for simplicity
            quantity=size,
            price=None,  # Market order - no price specified
            source_signal=signal,
            risk_checks_passed=["position_size", "risk_limits"],  # Populated by risk container
            timestamp=datetime.now(),
            metadata={
                'signal_strength': float(signal.strength),
                'signal_type': signal.signal_type.value,
                'strategy_id': signal.strategy_id,
                'container_id': self.container_id,
                'current_price': current_price
            }
        )
        
        # Add signal metadata to order
        if signal.metadata:
            order.metadata.update(signal.metadata)
        
        # Track order
        self.created_orders.append(order)
        self.order_count += 1
        
        self.logger.info(
            "Order created",
            order_id=order_id,
            symbol=signal.symbol,
            side=signal.side.value,
            quantity=float(size),
            signal_id=signal.signal_id,
            order_count=self.order_count
        )
        
        return order
    
    def _generate_order_id(self) -> str:
        """
        Generate unique order ID.
        
        Returns:
            Unique order identifier
        """
        # Use container ID and counter for readable IDs
        return f"ORD_{self.container_id}_{self.order_count + 1:04d}"
    
    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.
        
        Args:
            order_id: Order ID to find
            
        Returns:
            Order if found, None otherwise
        """
        for order in self.created_orders:
            if order.order_id == order_id:
                return order
        return None
    
    def get_orders_for_symbol(self, symbol: str) -> List[Order]:
        """
        Get all orders for a specific symbol.
        
        Args:
            symbol: Symbol to filter by
            
        Returns:
            List of orders for the symbol
        """
        return [order for order in self.created_orders if order.symbol == symbol]
    
    def get_recent_orders(self, limit: int = 10) -> List[Order]:
        """
        Get most recent orders.
        
        Args:
            limit: Maximum number of orders to return
            
        Returns:
            List of recent orders
        """
        return self.created_orders[-limit:] if self.created_orders else []
    
    def get_order_stats(self) -> Dict[str, Any]:
        """
        Get order statistics.
        
        Returns:
            Dictionary containing order statistics
        """
        if not self.created_orders:
            return {
                'total_orders': 0,
                'orders_by_side': {},
                'orders_by_symbol': {},
                'avg_order_size': 0
            }
        
        # Count by side
        side_counts = {}
        for order in self.created_orders:
            side = order.side.value
            side_counts[side] = side_counts.get(side, 0) + 1
        
        # Count by symbol
        symbol_counts = {}
        for order in self.created_orders:
            symbol = order.symbol
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        # Calculate average size
        total_quantity = sum(float(order.quantity) for order in self.created_orders)
        avg_size = total_quantity / len(self.created_orders)
        
        return {
            'total_orders': len(self.created_orders),
            'orders_by_side': side_counts,
            'orders_by_symbol': symbol_counts,
            'avg_order_size': avg_size,
            'most_recent_order': self.created_orders[-1].order_id if self.created_orders else None
        }
    
    def reset(self) -> None:
        """
        Reset order manager state.
        
        This method supports backtesting scenarios where order managers
        need to be reset between test runs.
        """
        orders_cleared = len(self.created_orders)
        self.created_orders.clear()
        self.order_count = 0
        
        self.logger.info(
            "OrderManager reset",
            orders_cleared=orders_cleared
        )
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current order manager state.
        
        Returns:
            Dictionary containing order manager state
        """
        return {
            'container_id': self.container_id,
            'order_count': self.order_count,
            'orders_tracked': len(self.created_orders),
            'order_stats': self.get_order_stats()
        }