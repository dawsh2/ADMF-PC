"""Canonical execution engine implementation for ADMF-PC.

This unified execution engine follows Protocol + Composition principles
with proper dependency injection and lifecycle management.
"""

import asyncio
from typing import Dict, Optional, Any, List, Set
from datetime import datetime
import uuid
from decimal import Decimal

from ..core.logging.structured import StructuredLogger, LogContext

from ..core.components.protocols import Component, Lifecycle, EventCapable
from ..core.types.events import Event, EventType
from .protocols import (
    ExecutionEngine, Order, Fill, OrderStatus, FillType, FillStatus,
    OrderType, OrderSide, Broker, OrderProcessor, MarketSimulator as MarketSimulatorProtocol
)
# from .context import ExecutionContext  # Removed - redundant bookkeeping


# Structured logging will be initialized per component instance


class DefaultExecutionEngine(Component, Lifecycle, EventCapable):
    """Canonical execution engine implementation using Protocol + Composition.
    
    This execution engine:
    - Uses dependency injection for clean separation of concerns
    - Follows Component/Lifecycle protocols for ADMF-PC compliance
    - Supports event-driven execution patterns
    - Handles both backtesting and live trading through configuration
    """
    
    def __init__(
        self,
        component_id: str = None,
        broker: Optional[Broker] = None,
        order_manager: Optional[OrderProcessor] = None,
        market_simulator: Optional[MarketSimulatorProtocol] = None,
        # context removed - redundant bookkeeping handled by event tracing
        mode: str = "backtest"
    ):
        """Initialize execution engine with dependency injection.
        
        Args:
            component_id: Unique component identifier
            broker: Broker implementation (injected)
            order_manager: Order manager (injected) 
            market_simulator: Market simulator (injected)
            context: Execution context (injected)
            mode: Operating mode ('backtest' or 'live')
        """
        self._component_id = component_id or f"execution_engine_{uuid.uuid4().hex[:8]}"
        self._broker = broker
        self._order_manager = order_manager
        self._market_simulator = market_simulator
        # Context removed - bookkeeping via event tracing
        self._mode = mode
        
        # State tracking
        self._initialized = False
        self._running = False
        self._shutdown = False
        
        # Market data cache with thread safety
        self._market_data: Dict[str, Dict[str, Any]] = {}
        self._market_data_lock = asyncio.Lock()
        
        # Event tracking and statistics
        self._processed_events = 0
        self._active_orders: Set[str] = set()
        self._execution_stats = {
            'events_processed': 0,
            'orders_executed': 0,
            'fills_generated': 0,
            'fills_processed': 0,
            'orders_rejected': 0,
            'total_commission': 0.0,
            'errors_encountered': 0
        }
        
        logger.info(
            f"ExecutionEngine initialized in {mode} mode",
            component_id=self._component_id
        )
    
    @property
    def component_id(self) -> str:
        """Get component ID."""
        return self._component_id
    
    # Lifecycle methods
    async def initialize(self) -> None:
        """Initialize execution engine and dependencies."""
        if self._initialized:
            logger.warning(f"ExecutionEngine {self._component_id} already initialized")
            return
        
        # Validate dependencies
        if self._broker is None:
            raise ValueError("Broker must be injected for execution engine to function")
        
        if self._order_manager is None:
            raise ValueError("Order manager must be injected for execution engine to function")
        
        # Initialize dependencies if they support it
        if hasattr(self._broker, 'initialize'):
            await self._broker.initialize()
        
        if hasattr(self._order_manager, 'initialize'):
            await self._order_manager.initialize()
        
        if hasattr(self._market_simulator, 'initialize'):
            await self._market_simulator.initialize()
        
        self._initialized = True
        logger.info(f"ExecutionEngine {self._component_id} initialized successfully")
    
    async def start(self) -> None:
        """Start execution engine."""
        if not self._initialized:
            await self.initialize()
        
        if self._running:
            logger.warning(f"ExecutionEngine {self._component_id} already running")
            return
        
        # Start dependencies if they support it
        if hasattr(self._broker, 'start'):
            await self._broker.start()
        
        if hasattr(self._order_manager, 'start'):
            await self._order_manager.start()
        
        if hasattr(self._market_simulator, 'start'):
            await self._market_simulator.start()
        
        self._running = True
        logger.info(f"ExecutionEngine {self._component_id} started")
    
    async def stop(self) -> None:
        """Stop execution engine."""
        if not self._running:
            logger.warning(f"ExecutionEngine {self._component_id} not running")
            return
        
        self._running = False
        self._shutdown = True
        
        # Cancel all active orders
        await self._cancel_all_active_orders()
        
        # Stop dependencies if they support it
        if hasattr(self._broker, 'stop'):
            await self._broker.stop()
        
        if hasattr(self._order_manager, 'stop'):
            await self._order_manager.stop()
        
        if hasattr(self._market_simulator, 'stop'):
            await self._market_simulator.stop()
        
        logger.info(f"ExecutionEngine {self._component_id} stopped")
    
    async def reset(self) -> None:
        """Reset execution engine state."""
        # Reset internal state
        async with self._market_data_lock:
            self._market_data.clear()
        self._active_orders.clear()
        self._execution_stats = {
            'events_processed': 0,
            'orders_executed': 0,
            'fills_generated': 0,
            'fills_processed': 0,
            'orders_rejected': 0,
            'total_commission': 0.0,
            'errors_encountered': 0
        }
        
        # Reset dependencies if they support it
        if hasattr(self._broker, 'reset'):
            await self._broker.reset()
        if hasattr(self._order_manager, 'reset'):
            await self._order_manager.reset()
        # Context reset removed - handled by state initialization
        
        logger.info(f"ExecutionEngine {self._component_id} reset")
    
    # EventCapable methods
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process incoming events with comprehensive error handling."""
        if not self._running or self._shutdown:
            logger.warning("ExecutionEngine not running, ignoring event")
            return None
        
        self._processed_events += 1
        self._execution_stats['events_processed'] += 1
        
        logger.debug(
            f"Processing event",
            component_id=self._component_id,
            event_type=event.type.value if hasattr(event.type, 'value') else str(event.type),
            event_count=self._processed_events
        )
        
        try:
            if event.type == EventType.ORDER:
                return await self._handle_order_event(event)
            elif event.type == EventType.BAR:
                await self._handle_market_data_event(event)
                return None
            elif event.type == EventType.TICK:
                await self._handle_market_data_event(event)
                return None
            elif event.type == EventType.CANCEL:
                return await self._handle_cancel_event(event)
            else:
                logger.debug(f"Unhandled event type: {event.type}")
                return None
        except Exception as e:
            self._execution_stats['errors_encountered'] += 1
            logger.error(
                f"Error processing event: {e}",
                exc_info=True,
                extra={'event_type': str(event.type), 'event_data': event.payload}
            )
            return self._create_error_event(event, str(e))
    
    async def _handle_order_event(self, event: Event) -> Optional[Event]:
        """Handle order events."""
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
            logger.error(f"Error handling order event: {e}", component_id=self._component_id)
            return self._create_error_event(event, str(e))
    
    async def _handle_market_data_event(self, event: Event) -> None:
        """Handle market data events."""
        try:
            market_data = event.payload
            
            async with self._market_data_lock:
                # Update market data cache
                if isinstance(market_data, dict):
                    symbol = market_data.get("symbol")
                    if symbol:
                        self._market_data[symbol] = {
                            "price": market_data.get("price", 0),
                            "volume": market_data.get("volume", 0),
                            "bid": market_data.get("bid", 0),
                            "ask": market_data.get("ask", 0),
                            "timestamp": market_data.get("timestamp", datetime.now().isoformat())
                        }
                    
                    # Handle bulk price updates
                    prices = market_data.get('prices', {})
                    for sym, price in prices.items():
                        if sym not in self._market_data:
                            self._market_data[sym] = {}
                        self._market_data[sym].update({
                            'price': price,
                            'timestamp': datetime.now().isoformat()
                        })
                
                # Process pending orders with broker
                if self._broker:
                    if hasattr(self._broker, 'process_pending_orders'):
                        await self._broker.process_pending_orders({"prices": {
                            sym: data["price"] for sym, data in self._market_data.items()
                        }})
                    elif hasattr(self._broker, 'process_market_data'):
                        await self._broker.process_market_data(market_data)
                    
        except Exception as e:
            logger.error(f"Error handling market data event: {e}", component_id=self._component_id)
    
    async def execute_order(self, order: Order) -> Optional[Fill]:
        """Execute order with comprehensive validation and error handling."""
        if not self._running or self._shutdown:
            logger.warning("ExecutionEngine not running, cannot execute order")
            return None
        
        try:
            # Validate order
            validation_result = await self._validate_order(order)
            if not validation_result.is_valid:
                await self._reject_order(order, validation_result.reason)
                return None
            
            # Add to active orders
            self._active_orders.add(order.order_id)
            # Context tracking removed - redundant with _active_orders
            
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
            
            # Simulate execution
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
            # Context tracking removed
    
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
        # Context metrics removed - will be handled by event tracer
        context_metrics = {}
        
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


    async def _handle_cancel_event(self, event: Event) -> Optional[Event]:
        """Handle order cancellation events."""
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
                
                # Context tracking removed
                
                # Remove from active orders
                self._active_orders.discard(order_id)
                # Context tracking removed
                
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
                "original_event_type": original_event.type.name if hasattr(original_event.type, 'name') else str(original_event.type),
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
        
        # Record metrics - handled by event tracing system
        self._execution_stats['fills_processed'] += 1
        self._execution_stats['total_commission'] += fill.commission
        
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
        
        # Record metrics - handled by event tracing system
        self._execution_stats['orders_rejected'] += 1
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
                # Context tracking removed
            except Exception as e:
                logger.error(f"Error cancelling order {order_id}: {e}")
        
        self._active_orders.clear()


class ValidationResult:
    """Result of order validation."""
    
    def __init__(self, is_valid: bool, reason: str):
        self.is_valid = is_valid
        self.reason = reason


# Factory function for creating execution engines (follows Protocol + Composition)
def create_execution_engine(
    component_id: str = None,
    broker: Optional[Broker] = None,
    order_manager: Optional[OrderProcessor] = None,
    market_simulator: Optional[MarketSimulatorProtocol] = None,
    mode: str = "backtest"
) -> DefaultExecutionEngine:
    """
    Factory function for creating execution engine instances.
    
    Args:
        component_id: Unique component identifier
        broker: Broker implementation (injected)
        order_manager: Order manager (injected)
        market_simulator: Market simulator (injected)
        mode: Operating mode ('backtest' or 'live')
        
    Returns:
        Configured DefaultExecutionEngine instance
    """
    return DefaultExecutionEngine(
        component_id=component_id,
        broker=broker,
        order_manager=order_manager,
        market_simulator=market_simulator,
        mode=mode
    )
