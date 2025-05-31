"""Main execution engine implementation."""

import asyncio
from typing import Dict, Optional, Any, List
from datetime import datetime

from .protocols import (
    ExecutionEngine, Order, Fill, OrderStatus,
    OrderType, OrderSide
)
from .backtest_broker import BacktestBroker
from .order_manager import OrderManager
from .market_simulation import MarketSimulator
from .execution_context import ExecutionContext
from ..core.events.types import Event, EventType
import logging


logger = logging.getLogger(__name__)


class DefaultExecutionEngine:
    """Default implementation of execution engine."""
    
    def __init__(
        self,
        broker: Optional[BacktestBroker] = None,
        order_manager: Optional[OrderManager] = None,
        market_simulator: Optional[MarketSimulator] = None,
        context: Optional[ExecutionContext] = None
    ):
        """Initialize execution engine."""
        self.broker = broker or BacktestBroker()
        self.order_manager = order_manager or OrderManager()
        self.market_simulator = market_simulator or MarketSimulator()
        self.context = context or ExecutionContext()
        self._initialized = False
        self._shutdown = False
        
        # Market data cache
        self._market_data: Dict[str, Dict[str, float]] = {}
        
        logger.info("Initialized DefaultExecutionEngine")
    
    async def initialize(self) -> None:
        """Initialize execution engine."""
        if self._initialized:
            logger.warning("ExecutionEngine already initialized")
            return
        
        # Initialize components
        self.context.reset()
        
        self._initialized = True
        logger.info("ExecutionEngine initialized")
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process incoming event."""
        if not self._initialized:
            logger.error("ExecutionEngine not initialized")
            return None
        
        if self._shutdown:
            logger.warning("ExecutionEngine is shutdown")
            return None
        
        try:
            # Handle different event types
            if event.type == EventType.ORDER:
                return await self._process_order_event(event)
            elif event.type == EventType.MARKET_DATA:
                await self._update_market_data(event)
                return None
            elif event.type == EventType.CANCEL:
                return await self._process_cancel_event(event)
            else:
                logger.debug(f"Ignoring event type: {event.type}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing event: {e}", exc_info=True)
            return None
    
    async def _process_order_event(self, event: Event) -> Optional[Event]:
        """Process ORDER event."""
        order_data = event.data
        
        # Create order from event data
        order = self.order_manager.create_order(
            symbol=order_data["symbol"],
            side=OrderSide[order_data["side"]],
            quantity=order_data["quantity"],
            order_type=OrderType[order_data.get("order_type", "MARKET")],
            price=order_data.get("price"),
            stop_price=order_data.get("stop_price"),
            metadata=order_data.get("metadata", {})
        )
        
        # Validate order
        if not self.order_manager.validate_order(order):
            self.order_manager.update_order_status(
                order.order_id,
                OrderStatus.REJECTED
            )
            self.context.record_order_status("rejected")
            return None
        
        # Execute order
        fill = self.execute_order(order)
        
        if fill:
            # Create FILL event
            fill_event = Event(
                type=EventType.FILL,
                data={
                    "fill_id": fill.fill_id,
                    "order_id": fill.order_id,
                    "symbol": fill.symbol,
                    "side": fill.side.name,
                    "quantity": fill.quantity,
                    "price": fill.price,
                    "commission": fill.commission,
                    "slippage": fill.slippage,
                    "fill_type": fill.fill_type.name,
                    "executed_at": fill.executed_at.isoformat(),
                    "metadata": fill.metadata
                },
                metadata={
                    "source": "execution_engine",
                    "original_order_id": order.order_id
                }
            )
            return fill_event
        
        return None
    
    async def _process_cancel_event(self, event: Event) -> Optional[Event]:
        """Process CANCEL event."""
        order_id = event.data.get("order_id")
        if not order_id:
            logger.error("Cancel event missing order_id")
            return None
        
        # Cancel with broker
        success = self.broker.cancel_order(order_id)
        
        if success:
            # Update order manager
            self.order_manager.cancel_order(order_id)
            self.context.record_order_status("cancelled")
            
            # Create CANCELLED event
            return Event(
                type=EventType.CANCELLED,
                data={
                    "order_id": order_id,
                    "cancelled_at": datetime.now().isoformat()
                },
                metadata={
                    "source": "execution_engine"
                }
            )
        
        return None
    
    async def _update_market_data(self, event: Event) -> None:
        """Update market data from event."""
        symbol = event.data.get("symbol")
        if not symbol:
            return
        
        # Store raw market data and normalize price
        price = event.data.get("price") or event.data.get("close", 0)
        self._market_data[symbol] = {
            "price": price,
            "close": event.data.get("close", price),
            "open": event.data.get("open", price),
            "high": event.data.get("high", price),
            "low": event.data.get("low", price),
            "volume": event.data.get("volume", 0),
            "bid": event.data.get("bid", 0),
            "ask": event.data.get("ask", 0),
            "timestamp": event.timestamp
        }
        
        # Update broker positions (commented out - method not implemented in refactored broker)
        # prices = {
        #     symbol: data["price"]
        #     for symbol, data in self._market_data.items()
        #     if data["price"] > 0
        # }
        # self.broker.update_position_prices(prices)
    
    def execute_order(self, order: Order) -> Optional[Fill]:
        """Execute order through broker."""
        logger.info(f"🔧 DefaultExecutionEngine.execute_order() called for {order.order_id}")
        
        # with self.context.transaction(f"execute_{order.order_id}"):
        if True:  # Temporarily disable transaction context to test for deadlock
            try:
                # Add to active orders
                self.context.add_active_order(order.order_id)
                
                # Add order to order manager (it wasn't created through the order manager)
                with self.order_manager._lock:
                    self.order_manager.orders[order.order_id] = order
                    self.order_manager.order_status[order.order_id] = OrderStatus.PENDING
                    self.order_manager.pending_orders.add(order.order_id)
                
                # Submit to broker
                self.order_manager.submit_order(order.order_id)
                broker_order_id = self.broker.submit_order(order)
                logger.info(f"   Broker order ID: {broker_order_id}")
                
                # Get market data
                market_data = self._market_data.get(order.symbol, {})
                # Try price first, then close, then default
                market_price = market_data.get("price") or market_data.get("close", 100.0)
                volume = market_data.get("volume", 1000000)  # Default volume
                spread = abs(market_data.get("ask", 0) - market_data.get("bid", 0))
                if spread == 0:
                    spread = 0.01  # Default spread
                
                logger.info(f"   Market data for {order.symbol}: price={market_price}, volume={volume}, spread={spread}")
                logger.info(f"   Raw market data: {market_data}")
                
                # Simulate fill
                logger.info(f"   Calling market_simulator.simulate_fill()")
                fill = self.market_simulator.simulate_fill(
                    order,
                    market_price,
                    volume,
                    spread
                )
                
                if fill:
                    # Execute fill with broker
                    broker_result = self.broker.execute_fill(fill)
                    logger.info(f"   Broker execute_fill result: {broker_result}")
                    if broker_result:
                        logger.info(f"   ✅ Broker returned True, proceeding with order completion")
                        
                        # Update order manager
                        logger.info(f"   📝 Adding fill to order manager")
                        try:
                            fill_added = self.order_manager.add_fill(order.order_id, fill)
                            logger.info(f"   📝 Fill added to order manager: {fill_added}")
                        except Exception as e:
                            logger.error(f"   ❌ Error adding fill to order manager: {e}")
                            raise
                        
                        # Record metrics
                        logger.info(f"   📊 Recording fill metrics")
                        try:
                            self.context.record_fill(
                                order.order_id,
                                fill.quantity,
                                fill.commission,
                                fill.slippage
                            )
                            logger.info(f"   📊 Fill metrics recorded successfully")
                        except Exception as e:
                            logger.error(f"   ❌ Error recording fill metrics: {e}")
                            raise
                        
                        logger.info(
                            f"Order executed: {order.order_id} - "
                            f"Fill: {fill.fill_id}"
                        )
                        
                        return fill
                    else:
                        logger.error(f"Broker failed to execute fill: {fill.fill_id}")
                else:
                    logger.info(f"Order not filled: {order.order_id}")
                
                # Remove from active orders
                self.context.remove_active_order(order.order_id)
                
                return None
                
            except Exception as e:
                logger.error(
                    f"Error executing order {order.order_id}: {e}",
                    exc_info=True
                )
                self.context.remove_active_order(order.order_id)
                self.order_manager.update_order_status(
                    order.order_id,
                    OrderStatus.REJECTED
                )
                self.context.record_order_status("rejected")
                return None
    
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        metrics = self.context.get_metrics()
        broker_info = self.broker.get_account_info()
        order_summary = self.order_manager.get_order_summary()
        broker_summary = self.broker.get_execution_summary()
        
        return {
            "metrics": metrics,
            "account": broker_info,
            "orders": order_summary,
            "execution": broker_summary,
            "active_orders": len(self.context.get_active_orders()),
            "market_data_symbols": len(self._market_data)
        }
    
    async def shutdown(self) -> None:
        """Shutdown execution engine."""
        if self._shutdown:
            logger.warning("ExecutionEngine already shutdown")
            return
        
        self._shutdown = True
        
        # Cancel all active orders
        active_orders = self.order_manager.get_active_orders()
        for order in active_orders:
            self.broker.cancel_order(order.order_id)
            self.order_manager.cancel_order(order.order_id)
        
        # Clean up old orders
        cleaned = self.order_manager.cleanup_old_orders()
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old orders during shutdown")
        
        logger.info("ExecutionEngine shutdown complete")