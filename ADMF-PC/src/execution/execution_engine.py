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
from ..core.logging.structured import get_logger


logger = get_logger(__name__)


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
        await self.context.reset()
        
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
        order = await self.order_manager.create_order(
            symbol=order_data["symbol"],
            side=OrderSide[order_data["side"]],
            quantity=order_data["quantity"],
            order_type=OrderType[order_data.get("order_type", "MARKET")],
            price=order_data.get("price"),
            stop_price=order_data.get("stop_price"),
            metadata=order_data.get("metadata", {})
        )
        
        # Validate order
        if not await self.order_manager.validate_order(order):
            await self.order_manager.update_order_status(
                order.order_id,
                OrderStatus.REJECTED
            )
            await self.context.record_order_status("rejected")
            return None
        
        # Execute order
        fill = await self.execute_order(order)
        
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
        success = await self.broker.cancel_order(order_id)
        
        if success:
            # Update order manager
            await self.order_manager.cancel_order(order_id)
            await self.context.record_order_status("cancelled")
            
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
        
        self._market_data[symbol] = {
            "price": event.data.get("price", 0),
            "volume": event.data.get("volume", 0),
            "bid": event.data.get("bid", 0),
            "ask": event.data.get("ask", 0),
            "timestamp": event.timestamp
        }
        
        # Update broker positions
        prices = {
            symbol: data["price"]
            for symbol, data in self._market_data.items()
            if data["price"] > 0
        }
        await self.broker.update_position_prices(prices)
    
    async def execute_order(self, order: Order) -> Optional[Fill]:
        """Execute order through broker."""
        async with self.context.transaction(f"execute_{order.order_id}"):
            try:
                # Add to active orders
                await self.context.add_active_order(order.order_id)
                
                # Submit to broker
                await self.order_manager.submit_order(order.order_id)
                broker_order_id = await self.broker.submit_order(order)
                
                # Get market data
                market_data = self._market_data.get(order.symbol, {})
                market_price = market_data.get("price", 100.0)  # Default price
                volume = market_data.get("volume", 1000000)  # Default volume
                spread = abs(market_data.get("ask", 0) - market_data.get("bid", 0))
                if spread == 0:
                    spread = 0.01  # Default spread
                
                # Simulate fill
                fill = await self.market_simulator.simulate_fill(
                    order,
                    market_price,
                    volume,
                    spread
                )
                
                if fill:
                    # Execute fill with broker
                    if await self.broker.execute_fill(fill):
                        # Update order manager
                        await self.order_manager.add_fill(order.order_id, fill)
                        
                        # Record metrics
                        await self.context.record_fill(
                            order.order_id,
                            fill.quantity,
                            fill.commission,
                            fill.slippage
                        )
                        
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
                await self.context.remove_active_order(order.order_id)
                
                return None
                
            except Exception as e:
                logger.error(
                    f"Error executing order {order.order_id}: {e}",
                    exc_info=True
                )
                await self.context.remove_active_order(order.order_id)
                await self.order_manager.update_order_status(
                    order.order_id,
                    OrderStatus.REJECTED
                )
                await self.context.record_order_status("rejected")
                return None
    
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        metrics = await self.context.get_metrics()
        broker_info = await self.broker.get_account_info()
        order_summary = self.order_manager.get_order_summary()
        broker_summary = self.broker.get_execution_summary()
        
        return {
            "metrics": metrics,
            "account": broker_info,
            "orders": order_summary,
            "execution": broker_summary,
            "active_orders": len(await self.context.get_active_orders()),
            "market_data_symbols": len(self._market_data)
        }
    
    async def shutdown(self) -> None:
        """Shutdown execution engine."""
        if self._shutdown:
            logger.warning("ExecutionEngine already shutdown")
            return
        
        self._shutdown = True
        
        # Cancel all active orders
        active_orders = await self.order_manager.get_active_orders()
        for order in active_orders:
            await self.broker.cancel_order(order.order_id)
            await self.order_manager.cancel_order(order.order_id)
        
        # Clean up old orders
        cleaned = await self.order_manager.cleanup_old_orders()
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old orders during shutdown")
        
        logger.info("ExecutionEngine shutdown complete")