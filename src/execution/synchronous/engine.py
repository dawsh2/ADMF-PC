"""
Synchronous execution engine.

High-performance simulation without async overhead.
"""

import logging
from typing import Optional, Dict, Any, List
from decimal import Decimal
from datetime import datetime

from ..types import Order, Fill, ExecutionStats, FillStatus
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
            
            # For market orders, execute immediately using price from order
            if order.order_type.value.upper() == 'MARKET':
                # Use price from order (set by portfolio) or fallback to broker's market data
                market_price = float(order.price) if order.price else self.broker._get_market_price(order.symbol)
                
                if market_price:
                    # Apply slippage
                    slippage = self.broker.slippage_model.calculate_slippage(
                        order, market_price
                    )
                    execution_price = market_price + slippage
                    
                    # Calculate commission
                    commission = self.broker.commission_model.calculate_commission(
                        order.quantity, execution_price
                    )
                    
                    # Create fill
                    fill = Fill(
                        fill_id=f"FILL_{order.order_id}_{datetime.now().strftime('%H%M%S')}",
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        price=Decimal(str(execution_price)),
                        commission=Decimal(str(commission)),
                        executed_at=datetime.now(),
                        status=FillStatus.FILLED
                    )
                    
                    # Store fill in broker
                    self.broker._fills.append(fill)
                    self.order_manager.process_fill(fill)
                    self.stats.orders_filled += 1
                    self.stats.total_commission += fill.commission
                    self.logger.debug(f"Order filled: {order.order_id} -> {fill.fill_id}")
                    return fill
                else:
                    self.logger.warning(f"No market price available for {order.symbol}, order rejected")
                    self.order_manager.reject_order(order.order_id, f"No market price available for {order.symbol}")
                    self.stats.orders_rejected += 1
                    return None
            else:
                # Limit/stop orders - track as pending
                self.order_manager.track_pending_order(order, broker_order_id)
                self.logger.debug(f"Non-market order pending: {order.order_id}")
                return None
            
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
        
        # Update broker's market data cache first
        for symbol, data in market_data.items():
            self.broker._market_data[symbol] = data
            self.logger.info(f"📊 Updated market data for {symbol}: close={data.get('close')}")
        
        # Delegate market data processing to broker for pending orders
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
    
    def on_order(self, event: Any) -> None:
        """Handle ORDER event from portfolio.
        
        This is the event handler called when portfolio publishes ORDER events.
        """
        from ...core.events.types import Event
        
        if not isinstance(event, Event) or event.event_type != "ORDER":
            return
        
        payload = event.payload
        self.logger.info(f"🎯 Execution received ORDER: {payload.get('side')} {payload.get('quantity')} {payload.get('symbol')}")
        
        # Convert event payload to Order object
        from ..types import Order, OrderSide, OrderType
        
        # Handle side - could be enum value or string
        side_value = payload['side']
        if isinstance(side_value, str) and side_value.upper() in ['BUY', 'SELL']:
            side = OrderSide[side_value.upper()]
        else:
            side = OrderSide(side_value)
        
        # Handle order type - could be enum value or string
        order_type_value = payload['order_type']
        if isinstance(order_type_value, str) and order_type_value.upper() in ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']:
            order_type = OrderType[order_type_value.upper()]
        else:
            order_type = OrderType(order_type_value)
        
        # Handle price - could be string or None
        price = None
        if payload.get('price'):
            price = Decimal(str(payload['price']))
        
        order = Order(
            order_id=payload['order_id'],
            symbol=payload['symbol'],
            side=side,
            quantity=Decimal(str(payload['quantity'])),
            order_type=order_type,
            price=price,
            created_at=datetime.fromisoformat(payload['created_at'])
        )
        
        # Execute the order
        fill = self.execute_order(order)
        
        if fill:
            self.logger.info(f"  ✅ Order FILLED: {fill.quantity} @ ${fill.price} (commission: ${fill.commission})")
            
            # Publish FILL event
            if hasattr(self, '_container') and self._container:
                from ...core.events.types import EventType
                fill_event = Event(
                    event_type=EventType.FILL.value,
                    timestamp=datetime.now(),
                    payload=fill.to_dict(),
                    source_id="execution",
                    container_id=self._container.container_id
                )
                self._container.publish_event(fill_event, target_scope="parent")
                self.logger.info(f"  📮 Fill event published to portfolio")
        else:
            self.logger.warning(f"  ❌ Order execution failed")
    
    def set_container(self, container: Any) -> None:
        """Set container reference for event publishing."""
        self._container = container
    
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