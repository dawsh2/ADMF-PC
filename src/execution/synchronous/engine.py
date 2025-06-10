"""
Synchronous execution engine.

High-performance simulation without async overhead.
"""

import logging
from typing import Optional, Dict, Any, List
from decimal import Decimal
from datetime import datetime

from ..types import Order, Fill, ExecutionStats
from ..sync_protocols import SyncBroker
from .order_manager import SyncOrderManager
from ...data.protocols import DataProvider

logger = logging.getLogger(__name__)


class SyncExecutionEngine:
    """
    Synchronous execution engine.
    
    Provides high-performance order execution simulation without
    the overhead of async/await patterns.
    """
    
    def __init__(
        self,
        component_id: str,
        broker: SyncBroker,
        order_manager: Optional[SyncOrderManager] = None,
        data_provider: Optional[DataProvider] = None
    ):
        self.component_id = component_id
        self.broker = broker
        self.order_manager = order_manager or SyncOrderManager(f"{component_id}_orders")
        self.data_provider = data_provider
        
        self.stats = ExecutionStats()
        self.logger = logger.getChild(component_id)
        
        # Simple state tracking - no redundant market data cache!
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize execution engine."""
        if self._initialized:
            return
        
        self.order_manager.initialize()
        self.logger.info(f"Sync execution engine {self.component_id} initialized")
        self._initialized = True
    
    def execute_order(self, order: Order) -> Optional[Fill]:
        """
        Execute order synchronously.
        
        Orders are typically filled immediately using market simulation models.
        """
        if not self._initialized:
            self.initialize()
        
        self.logger.debug(f"Executing order: {order.order_id}")
        
        # Validate order
        is_valid, error_msg = self.broker.validate_order(order)
        if not is_valid:
            self.logger.warning(f"Order validation failed: {error_msg}")
            self.order_manager.reject_order(order.order_id, error_msg or "Validation failed")
            self.stats.orders_rejected += 1
            return None
        
        try:
            # Submit to broker
            broker_order_id = self.broker.submit_order(order)
            self.stats.orders_submitted += 1
            
            # Try immediate execution - delegate to broker
            fills = self.broker.get_pending_fills()
            
            # Find fill for this order
            order_fill = None
            for fill in fills:
                if fill.order_id == order.order_id:
                    order_fill = fill
                    break
            
            if order_fill:
                self.order_manager.process_fill(order_fill)
                self.stats.orders_filled += 1
                self.stats.total_commission += order_fill.commission
                self.logger.debug(f"Order filled: {order.order_id} -> {order_fill.fill_id}")
            else:
                self.order_manager.track_pending_order(order, broker_order_id)
                self.logger.debug(f"Order pending: {order.order_id}")
            
            return order_fill
            
        except Exception as e:
            self.logger.error(f"Order execution failed: {e}", exc_info=True)
            self.order_manager.reject_order(order.order_id, str(e))
            self.stats.orders_rejected += 1
            return None
    
    def process_market_data(self, market_data: Dict[str, Any]) -> List[Fill]:
        """
        Process market data and execute pending orders.
        
        This is called when new market data arrives to trigger
        execution of pending orders.
        """
        if not self._initialized:
            self.initialize()
        
        # Delegate market data processing entirely to broker - no redundant caching!
        new_fills = self.broker.process_market_data(market_data)
        
        # Process any new fills
        for fill in new_fills:
            self.order_manager.process_fill(fill)
            self.stats.orders_filled += 1
            self.stats.total_commission += fill.commission
            self.logger.debug(f"Market data triggered fill: {fill.fill_id}")
        
        return new_fills
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        if not self._initialized:
            return False
        
        try:
            # Try to cancel with broker
            cancelled = self.broker.cancel_order(order_id)
            
            if cancelled:
                self.order_manager.cancel_order(order_id)
                self.stats.orders_cancelled += 1
                self.logger.debug(f"Order cancelled: {order_id}")
            
            return cancelled
            
        except Exception as e:
            self.logger.error(f"Order cancellation failed: {e}")
            return False
    
    def get_execution_stats(self) -> ExecutionStats:
        """Get execution statistics."""
        return self.stats
    
    def get_order_status(self, order_id: str) -> Optional[str]:
        """Get order status."""
        return self.order_manager.get_order_status(order_id)
    
    def get_pending_orders(self) -> List[Order]:
        """Get pending orders."""
        return self.order_manager.get_pending_orders()
    
    def get_fills(self) -> List[Fill]:
        """Get all fills."""
        return self.order_manager.get_fills()
    
    def shutdown(self) -> None:
        """Shutdown execution engine."""
        if not self._initialized:
            return
        
        self.order_manager.shutdown()
        self.logger.info(f"Sync execution engine {self.component_id} shutdown")
        self._initialized = False
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.stats = ExecutionStats()
        self.logger.debug("Execution statistics reset")
    
    def get_market_data_cache(self) -> Dict[str, Any]:
        """Get current market data cache."""
        return self._market_data_cache.copy()
    
    def clear_market_data_cache(self) -> None:
        """Clear market data cache."""
        self._market_data_cache.clear()
        self.logger.debug("Market data cache cleared")