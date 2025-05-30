"""
Improved execution engine with proper dependency injection and error handling.

This engine follows the core system's patterns and integrates seamlessly
with the Risk module and backtest container architecture.
"""

import asyncio
from typing import Dict, Optional, Any, List, Set
from datetime import datetime
from decimal import Decimal
import logging

from ..core.components.protocols import Component, Lifecycle, EventCapable
from ..core.events.types import Event, EventType
from .protocols import (
    ExecutionEngine, Order, Fill, OrderStatus,
    Broker, OrderProcessor, MarketSimulator
)
from .execution_context import ExecutionContext

logger = logging.getLogger(__name__)


class ImprovedExecutionEngine(Component, Lifecycle, EventCapable):
    """
    Improved execution engine with proper dependency injection.
    
    This engine eliminates hard dependencies and follows the core system's
    architectural patterns for lifecycle management and event processing.
    """
    
    def __init__(
        self,
        component_id: str,
        broker: Broker,
        order_manager: OrderProcessor,
        market_simulator: MarketSimulator,
        execution_context: ExecutionContext
    ):
        """Initialize execution engine with proper dependency injection.
        
        Args:
            component_id: Unique component identifier
            broker: Broker implementation (injected)
            order_manager: Order manager (injected)
            market_simulator: Market simulator (injected)
            execution_context: Execution context (injected)
        """
        self._component_id = component_id
        self._broker = broker
        self._order_manager = order_manager
        self._market_simulator = market_simulator
        self._context = execution_context
        
        # Event bus will be set by container
        self._event_bus = None
        
        # Market data cache
        self._market_data: Dict[str, Dict[str, Any]] = {}
        self._market_data_lock = asyncio.Lock()
        
        # Execution state
        self._active_orders: Set[str] = set()
        self._execution_stats = {
            'events_processed': 0,
            'orders_executed': 0,
            'fills_generated': 0,
            'errors_encountered': 0
        }
        
        # Lifecycle state
        self._initialized = False
        self._running = False
        self._shutdown = False
        
        logger.info(f"ImprovedExecutionEngine created - ID: {component_id}")
    
    @property
    def component_id(self) -> str:
        """Component identifier."""
        return self._component_id
    
    @property
    def event_bus(self):
        """Get event bus."""
        return self._event_bus
    
    @event_bus.setter
    def event_bus(self, value):
        """Set event bus."""
        self._event_bus = value
    
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize the execution engine."""
        self.event_bus = context.get('event_bus')
        
        # Initialize dependencies if they support lifecycle
        if hasattr(self._broker, 'initialize'):
            self._broker.initialize(context)
        if hasattr(self._order_manager, 'initialize'):
            self._order_manager.initialize(context)
        if hasattr(self._market_simulator, 'initialize'):
            self._market_simulator.initialize(context)
        
        self._initialized = True
        logger.info(f"ExecutionEngine initialized - ID: {self._component_id}")
    
    def start(self) -> None:
        """Start the execution engine."""
        if not self._initialized:
            raise RuntimeError("ExecutionEngine not initialized")
        
        # Start dependencies if they support lifecycle
        if hasattr(self._broker, 'start'):
            self._broker.start()
        if hasattr(self._order_manager, 'start'):
            self._order_manager.start()
        if hasattr(self._market_simulator, 'start'):
            self._market_simulator.start()
        
        self._running = True
        logger.info(f"ExecutionEngine started - ID: {self._component_id}")
    
    def stop(self) -> None:
        """Stop the execution engine."""
        self._running = False
        self._shutdown = True
        
        # Cancel all active orders
        asyncio.create_task(self._cancel_all_active_orders())
        
        # Stop dependencies if they support lifecycle
        if hasattr(self._broker, 'stop'):
            self._broker.stop()
        if hasattr(self._order_manager, 'stop'):
            self._order_manager.stop()
        if hasattr(self._market_simulator, 'stop'):
            self._market_simulator.stop()
        
        logger.info(f"ExecutionEngine stopped - ID: {self._component_id}")
    
    def reset(self) -> None:
        """Reset the execution engine."""
        # Reset internal state
        self._market_data.clear()
        self._active_orders.clear()
        self._execution_stats = {
            'events_processed': 0,
            'orders_executed': 0,
            'fills_generated': 0,
            'errors_encountered': 0
        }
        
        # Reset dependencies if they support it
        if hasattr(self._broker, 'reset'):
            self._broker.reset()
        if hasattr(self._order_manager, 'reset'):
            self._order_manager.reset()
        if hasattr(self._context, 'reset'):
            asyncio.create_task(self._context.reset())
        
        logger.info(f"ExecutionEngine reset - ID: {self._component_id}")
    
    def teardown(self) -> None:
        """Teardown the execution engine."""
        # Stop if running
        if self._running:
            self.stop()
        
        # Teardown dependencies if they support it
        if hasattr(self._broker, 'teardown'):
            self._broker.teardown()
        if hasattr(self._order_manager, 'teardown'):
            self._order_manager.teardown()
        
        logger.info(f"ExecutionEngine torn down - ID: {self._component_id}")
    
    def initialize_events(self) -> None:
        """Initialize event subscriptions."""
        # Event subscriptions would be managed by parent container
        pass
    
    def teardown_events(self) -> None:
        """Clean up event subscriptions."""
        # Event cleanup would be managed by parent container
        pass
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process incoming event with comprehensive error handling."""
        if not self._running or self._shutdown:
            logger.warning("ExecutionEngine not running, ignoring event")
            return None
        
        self._execution_stats['events_processed'] += 1
        
        try:
            # Handle different event types
            if event.event_type == EventType.ORDER:
                return await self._process_order_event(event)
            elif event.event_type == EventType.MARKET_DATA:
                await self._process_market_data_event(event)
                return None
            elif event.event_type == EventType.CANCEL:
                return await self._process_cancel_event(event)
            else:
                logger.debug(f"Ignoring unsupported event type: {event.event_type}")
                return None
                
        except Exception as e:
            self._execution_stats['errors_encountered'] += 1
            logger.error(
                f"Error processing event: {e}",
                exc_info=True,
                extra={'event_type': event.event_type, 'event_data': event.payload}
            )
            return self._create_error_event(event, str(e))
    
    async def execute_order(self, order: Order) -> Optional[Fill]:
        """Execute order with comprehensive validation and error handling."""
        if not self._running or self._shutdown:
            logger.warning("ExecutionEngine not running, cannot execute order")
            return None
        
        async with self._context.transaction(f"execute_{order.order_id}"):
            try:
                # Validate order
                validation_result = await self._validate_order(order)
                if not validation_result.is_valid:
                    await self._reject_order(order, validation_result.reason)
                    return None
                
                # Add to active orders
                self._active_orders.add(order.order_id)
                await self._context.add_active_order(order.order_id)
                
                # Submit to broker
                try:
                    broker_order_id = await self._broker.submit_order(order)
                    logger.debug(f"Order submitted to broker - ID: {broker_order_id}")
                except Exception as e:
                    await self._reject_order(order, f"Broker submission failed: {e}")
                    return None
                
                # Update order manager
                if hasattr(self._order_manager, 'submit_order'):
                    await self._order_manager.submit_order(order.order_id)
                
                # Simulate execution if broker supports it
                fill = await self._simulate_execution(order)
                
                if fill:
                    # Process fill
                    await self._process_fill(order, fill)
                    self._execution_stats['orders_executed'] += 1
                    self._execution_stats['fills_generated'] += 1
                    
                    logger.info(
                        f"Order executed successfully - ID: {order.order_id}, "
                        f"Fill: {fill.quantity} @ {fill.price}"
                    )
                    
                    return fill
                else:
                    logger.info(f"Order not filled - ID: {order.order_id}")
                
                return None
                
            except Exception as e:
                logger.error(f"Error executing order {order.order_id}: {e}", exc_info=True)
                await self._reject_order(order, f"Execution error: {e}")
                return None
            
            finally:
                # Remove from active orders
                self._active_orders.discard(order.order_id)
                await self._context.remove_active_order(order.order_id)
    
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""
        # Get broker stats
        broker_stats = {}
        if hasattr(self._broker, 'get_execution_summary'):
            broker_stats = self._broker.get_execution_summary()
        elif hasattr(self._broker, 'get_account_info'):
            broker_stats = await self._broker.get_account_info()
        
        # Get order manager stats
        order_stats = {}
        if hasattr(self._order_manager, 'get_order_summary'):
            order_stats = self._order_manager.get_order_summary()
        
        # Get context metrics
        context_metrics = await self._context.get_metrics()
        
        return {
            "engine_stats": self._execution_stats,
            "broker_stats": broker_stats,
            "order_stats": order_stats,
            "context_metrics": context_metrics,
            "active_orders": len(self._active_orders),
            "market_data_symbols": len(self._market_data),
            "component_id": self._component_id,
            "running": self._running
        }
    
    async def shutdown(self) -> None:
        """Graceful shutdown of execution engine."""
        if self._shutdown:
            logger.warning("ExecutionEngine already shutdown")
            return
        
        logger.info("Initiating ExecutionEngine shutdown")
        
        # Cancel all active orders
        await self._cancel_all_active_orders()
        
        # Generate final statistics
        final_stats = await self.get_execution_stats()
        logger.info(f"ExecutionEngine shutdown complete - Final stats: {final_stats}")
        
        # Call stop
        self.stop()
    
    # Private methods
    
    async def _process_order_event(self, event: Event) -> Optional[Event]:
        """Process ORDER event."""
        try:
            order_data = event.payload
            
            # Create order from event data
            order = self._create_order_from_event(order_data)
            
            # Execute order
            fill = await self.execute_order(order)
            
            if fill:
                # Create FILL event
                return self._create_fill_event(fill)
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing ORDER event: {e}", exc_info=True)
            return self._create_error_event(event, str(e))
    
    async def _process_market_data_event(self, event: Event) -> None:
        """Process MARKET_DATA event."""
        try:
            market_data = event.payload
            
            async with self._market_data_lock:
                # Update market data cache
                symbol = market_data.get("symbol")
                if symbol:
                    self._market_data[symbol] = {
                        "price": market_data.get("price", 0),
                        "volume": market_data.get("volume", 0),
                        "bid": market_data.get("bid", 0),
                        "ask": market_data.get("ask", 0),
                        "timestamp": market_data.get("timestamp", datetime.now().isoformat())
                    }
                
                # Update broker if it supports market data updates
                if hasattr(self._broker, 'process_pending_orders'):
                    await self._broker.process_pending_orders({"prices": {
                        sym: data["price"] for sym, data in self._market_data.items()
                    }})
                elif hasattr(self._broker, 'update_market_data'):
                    await self._broker.update_market_data(market_data)
            
        except Exception as e:
            logger.error(f"Error processing MARKET_DATA event: {e}", exc_info=True)
    
    async def _process_cancel_event(self, event: Event) -> Optional[Event]:
        """Process CANCEL event."""
        try:
            order_id = event.payload.get("order_id")
            if not order_id:
                logger.error("CANCEL event missing order_id")
                return self._create_error_event(event, "Missing order_id")
            
            # Cancel with broker
            success = await self._broker.cancel_order(order_id)
            
            if success:
                # Update order manager
                if hasattr(self._order_manager, 'cancel_order'):
                    await self._order_manager.cancel_order(order_id)
                
                await self._context.record_order_status("cancelled")
                
                # Remove from active orders
                self._active_orders.discard(order_id)
                await self._context.remove_active_order(order_id)
                
                # Create CANCELLED event
                return Event(
                    event_type=EventType.CANCELLED,
                    source_id=self._component_id,
                    payload={
                        "order_id": order_id,
                        "cancelled_at": datetime.now().isoformat()
                    }
                )
            else:
                return self._create_error_event(event, f"Failed to cancel order {order_id}")
            
        except Exception as e:
            logger.error(f"Error processing CANCEL event: {e}", exc_info=True)
            return self._create_error_event(event, str(e))
    
    def _create_order_from_event(self, order_data: Dict[str, Any]) -> Order:
        """Create Order from event data."""
        from .protocols import OrderSide, OrderType
        
        return Order(
            order_id=order_data.get("order_id", f"ord_{datetime.now().timestamp()}"),
            symbol=order_data["symbol"],
            side=OrderSide[order_data["side"]],
            order_type=OrderType[order_data.get("order_type", "MARKET")],
            quantity=order_data["quantity"],
            price=order_data.get("price"),
            stop_price=order_data.get("stop_price"),
            time_in_force=order_data.get("time_in_force", "DAY"),
            created_at=datetime.now(),
            metadata=order_data.get("metadata", {})
        )
    
    def _create_fill_event(self, fill: Fill) -> Event:
        """Create FILL event from fill."""
        return Event(
            event_type=EventType.FILL,
            source_id=self._component_id,
            payload={
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
            }
        )
    
    def _create_error_event(self, original_event: Event, error_message: str) -> Event:
        """Create ERROR event."""
        return Event(
            event_type=EventType.ERROR,
            source_id=self._component_id,
            payload={
                "error_message": error_message,
                "original_event_type": original_event.event_type.name,
                "original_event_data": original_event.payload,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    async def _validate_order(self, order: Order) -> 'ValidationResult':
        """Validate order comprehensively."""
        # Use order manager validation if available
        if hasattr(self._order_manager, 'validate_order'):
            try:
                is_valid = await self._order_manager.validate_order(order)
                if not is_valid:
                    return ValidationResult(False, "Order manager validation failed")
            except Exception as e:
                return ValidationResult(False, f"Validation error: {e}")
        
        # Basic validation
        if order.quantity <= 0:
            return ValidationResult(False, "Invalid quantity")
        
        if not order.symbol:
            return ValidationResult(False, "Missing symbol")
        
        # Market data validation
        if order.symbol not in self._market_data:
            logger.warning(f"No market data for symbol: {order.symbol}")
            # Allow order but note the warning
        
        return ValidationResult(True, "Valid")
    
    async def _simulate_execution(self, order: Order) -> Optional[Fill]:
        """Simulate order execution."""
        try:
            # Get market data
            market_data = self._market_data.get(order.symbol, {})
            market_price = market_data.get("price", 100.0)  # Default price
            volume = market_data.get("volume", 1000000)  # Default volume
            
            # Calculate spread
            bid = market_data.get("bid", market_price * 0.999)
            ask = market_data.get("ask", market_price * 1.001)
            spread = abs(ask - bid)
            
            # Use market simulator if available
            if hasattr(self._market_simulator, 'simulate_fill'):
                return await self._market_simulator.simulate_fill(
                    order, market_price, volume, spread
                )
            
            # Fallback to simple execution
            return await self._simple_execution(order, market_price)
            
        except Exception as e:
            logger.error(f"Error simulating execution for order {order.order_id}: {e}")
            return None
    
    async def _simple_execution(self, order: Order, market_price: float) -> Fill:
        """Simple execution fallback."""
        from .protocols import FillType, FillStatus
        import uuid
        
        # Simple fill at market price
        commission = order.quantity * market_price * 0.001  # 0.1% commission
        slippage = market_price * 0.0005  # 0.05% slippage
        
        if order.side.name == "SELL":
            slippage = -slippage
        
        fill_price = market_price + slippage
        
        return Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage,
            fill_type=FillType.FULL,
            status=FillStatus.FILLED,
            executed_at=datetime.now(),
            metadata={"execution_type": "simple"}
        )
    
    async def _process_fill(self, order: Order, fill: Fill) -> None:
        """Process fill and update tracking."""
        # Update order manager
        if hasattr(self._order_manager, 'add_fill'):
            await self._order_manager.add_fill(order.order_id, fill)
        
        # Record metrics
        await self._context.record_fill(
            order.order_id,
            fill.quantity,
            fill.commission,
            fill.slippage
        )
        
        # Update order status
        if hasattr(self._order_manager, 'update_order_status'):
            await self._order_manager.update_order_status(
                order.order_id,
                OrderStatus.FILLED
            )
    
    async def _reject_order(self, order: Order, reason: str) -> None:
        """Reject order with proper tracking."""
        logger.warning(f"Order rejected - ID: {order.order_id}, Reason: {reason}")
        
        # Update order manager
        if hasattr(self._order_manager, 'update_order_status'):
            await self._order_manager.update_order_status(
                order.order_id,
                OrderStatus.REJECTED
            )
        
        # Record metrics
        await self._context.record_order_status("rejected")
        self._execution_stats['errors_encountered'] += 1
    
    async def _cancel_all_active_orders(self) -> None:
        """Cancel all active orders."""
        if not self._active_orders:
            return
        
        logger.info(f"Cancelling {len(self._active_orders)} active orders")
        
        # Create list to avoid modification during iteration
        order_ids = list(self._active_orders)
        
        for order_id in order_ids:
            try:
                await self._broker.cancel_order(order_id)
                if hasattr(self._order_manager, 'cancel_order'):
                    await self._order_manager.cancel_order(order_id)
                await self._context.record_order_status("cancelled")
            except Exception as e:
                logger.error(f"Error cancelling order {order_id}: {e}")
        
        self._active_orders.clear()


class ValidationResult:
    """Result of order validation."""
    
    def __init__(self, is_valid: bool, reason: str):
        self.is_valid = is_valid
        self.reason = reason


def create_execution_engine(
    component_id: str,
    broker: Broker,
    order_manager: OrderProcessor,
    market_simulator: MarketSimulator,
    execution_context: ExecutionContext
) -> ImprovedExecutionEngine:
    """Factory function to create execution engine with proper dependencies."""
    return ImprovedExecutionEngine(
        component_id=component_id,
        broker=broker,
        order_manager=order_manager,
        market_simulator=market_simulator,
        execution_context=execution_context
    )
