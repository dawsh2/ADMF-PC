"""
Clean async execution engine following the architecture principles.

- Async at the boundaries (broker I/O, order monitoring)
- Sync at the core (strategy signals, portfolio state)
- No complex bridges - clean event queue pattern
- Natural async patterns without over-engineering
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import threading

from ..types import Order, Fill, Position, ExecutionStats
from ...core.events.types import Event, EventType

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for execution engine."""
    component_id: str
    order_poll_interval: float = 1.0
    fill_poll_interval: float = 0.5
    position_update_interval: float = 5.0
    max_pending_orders: int = 100


class CleanAsyncExecutionEngine:
    """
    Clean async execution engine.
    
    Follows the architecture principles:
    - Async for I/O (broker communication)
    - Sync for logic (order validation, portfolio updates)
    - Simple event queue for sync/async boundary
    - No complex threading or bridges
    """
    
    def __init__(self, config: ExecutionConfig, broker):
        self.config = config
        self.broker = broker
        self.logger = logger.getChild(config.component_id)
        
        # Simple event queue - the clean boundary
        self.event_queue: asyncio.Queue[Event] = asyncio.Queue()
        
        # Sync state (no I/O here)
        self.stats = ExecutionStats()
        self._running = False
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
        # Portfolio reference (set by container)
        self.portfolio = None
        
    def set_portfolio(self, portfolio):
        """Set portfolio reference (called by container during wiring)."""
        self.portfolio = portfolio
        self.logger.info("Portfolio connected to execution engine")
    
    async def start(self) -> None:
        """Start the execution engine."""
        if self._running:
            return
        
        self.logger.info("Starting clean async execution engine")
        
        # Connect to broker (async I/O)
        await self.broker.connect()
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._order_processing_loop()),
            asyncio.create_task(self._fill_monitoring_loop()),
            asyncio.create_task(self._position_sync_loop()),
        ]
        
        # Add WebSocket order stream if available
        if hasattr(self.broker, 'has_trade_stream') and self.broker.has_trade_stream:
            self._tasks.append(asyncio.create_task(self._order_stream_loop()))
            self.logger.info("WebSocket order updates enabled")
        
        self._running = True
        self.logger.info("Execution engine started successfully")
    
    async def stop(self) -> None:
        """Stop the execution engine."""
        if not self._running:
            return
        
        self.logger.info("Stopping execution engine")
        self._running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Disconnect from broker
        await self.broker.disconnect()
        
        self.logger.info("Execution engine stopped")
    
    def emit_event(self, event: Event) -> None:
        """
        Thread-safe event emission (called from sync code).
        
        This is the clean boundary between sync and async.
        """
        asyncio.run_coroutine_threadsafe(
            self.event_queue.put(event),
            asyncio.get_event_loop()
        )
    
    async def _order_processing_loop(self) -> None:
        """Process orders from the event queue."""
        self.logger.debug("Order processing loop started")
        
        while self._running:
            try:
                # Wait for order events
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=self.config.order_poll_interval
                )
                
                if event.event_type == EventType.ORDER.value:
                    await self._process_order_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing order: {e}")
        
        self.logger.debug("Order processing loop stopped")
    
    async def _process_order_event(self, event: Event) -> None:
        """Process a single order event."""
        try:
            # Extract order from event
            order = self._event_to_order(event)
            if not order:
                return
            
            # Validate order (async I/O)
            is_valid, error = await self.broker.validate_order(order)
            if not is_valid:
                self.logger.warning(f"Order rejected: {error}")
                self.stats.orders_rejected += 1
                self._emit_order_rejected(order, error)
                return
            
            # Submit to broker (async I/O)
            broker_id = await self.broker.submit_order(order)
            
            self.stats.orders_submitted += 1
            self.logger.info(
                f"Order submitted: {order.symbol} {order.side.value} "
                f"{order.quantity} @ {order.order_type.value}"
            )
            
            # Emit acknowledgment event
            self._emit_order_acknowledged(order, broker_id)
            
        except Exception as e:
            self.logger.error(f"Failed to submit order: {e}")
            self.stats.orders_rejected += 1
    
    async def _fill_monitoring_loop(self) -> None:
        """
        Monitor for fills from broker.
        
        This serves as a fallback when WebSocket is not available,
        or to catch any missed updates.
        """
        # Reduce polling frequency if WebSocket is active
        has_websocket = hasattr(self.broker, 'has_trade_stream') and self.broker.has_trade_stream
        poll_interval = self.config.fill_poll_interval * 10 if has_websocket else self.config.fill_poll_interval
        
        self.logger.debug(f"Fill monitoring loop started (polling every {poll_interval}s)")
        
        while self._running:
            try:
                # Skip if WebSocket is active and working
                if has_websocket:
                    # Just do periodic checks for missed fills
                    await asyncio.sleep(poll_interval)
                    
                    # Only check if we have pending orders
                    if not hasattr(self.broker, '_pending_orders') or not self.broker._pending_orders:
                        continue
                
                # Check for fills (async I/O)
                fills = await self.broker.get_recent_fills()
                
                for fill in fills:
                    await self._process_fill(fill)
                
                # Wait before next check
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                self.logger.error(f"Error monitoring fills: {e}")
                await asyncio.sleep(poll_interval)
        
        self.logger.debug("Fill monitoring loop stopped")
    
    async def _process_fill(self, fill: Fill) -> None:
        """Process a fill from broker."""
        try:
            self.stats.orders_filled += 1
            self.stats.total_commission += fill.commission
            
            self.logger.info(
                f"Fill received: {fill.symbol} {fill.side.value} "
                f"{fill.quantity} @ {fill.price}"
            )
            
            # Update portfolio if available (sync operation)
            if self.portfolio:
                self._update_portfolio_with_fill(fill)
            
            # Emit fill event
            self._emit_fill_event(fill)
            
        except Exception as e:
            self.logger.error(f"Error processing fill: {e}")
    
    async def _position_sync_loop(self) -> None:
        """Periodically sync positions with broker."""
        self.logger.debug("Position sync loop started")
        
        while self._running:
            try:
                # Get positions from broker (async I/O)
                positions = await self.broker.get_positions()
                
                # Update portfolio if available (sync operation)
                if self.portfolio and positions:
                    self._sync_portfolio_positions(positions)
                
                # Get account info
                account = await self.broker.get_account_info()
                if account:
                    self._emit_account_update(account)
                
                # Wait before next sync
                await asyncio.sleep(self.config.position_update_interval)
                
            except Exception as e:
                self.logger.error(f"Error syncing positions: {e}")
                await asyncio.sleep(self.config.position_update_interval)
        
        self.logger.debug("Position sync loop stopped")
    
    async def _order_stream_loop(self) -> None:
        """
        Process real-time order updates via WebSocket.
        
        This provides instant notifications instead of polling.
        """
        self.logger.debug("Order stream loop started (WebSocket)")
        
        while self._running:
            try:
                # Stream order updates
                async for update in self.broker.stream_order_updates():
                    self.logger.debug(f"WebSocket update: {update.event.value} for {update.symbol}")
                    
                    # Handle different event types
                    if update.event.value in ['fill', 'partial_fill']:
                        # Convert to Fill and process
                        fill = self.broker.trade_update_to_fill(update)
                        if fill:
                            await self._process_fill(fill)
                            # Remove from pending
                            if update.event.value == 'fill':
                                self.broker._pending_orders.discard(fill.order_id)
                    
                    elif update.event.value == 'rejected':
                        self.logger.warning(f"Order rejected: {update.symbol}")
                        self.stats.orders_rejected += 1
                        # Remove from pending
                        internal_id = self.broker._alpaca_to_internal.get(update.order_id)
                        if internal_id:
                            self.broker._pending_orders.discard(internal_id)
                    
                    elif update.event.value == 'canceled':
                        self.logger.info(f"Order canceled: {update.symbol}")
                        self.stats.orders_cancelled += 1
                        # Remove from pending
                        internal_id = self.broker._alpaca_to_internal.get(update.order_id)
                        if internal_id:
                            self.broker._pending_orders.discard(internal_id)
                    
                    elif update.event.value == 'new':
                        self.logger.debug(f"Order acknowledged: {update.symbol}")
                
                # If stream ends, wait before retry
                self.logger.warning("Order stream ended, retrying in 5 seconds...")
                await asyncio.sleep(5.0)
                
            except Exception as e:
                self.logger.error(f"Error in order stream: {e}")
                await asyncio.sleep(5.0)
        
        self.logger.debug("Order stream loop stopped")
    
    # Sync helper methods (no I/O)
    
    def _event_to_order(self, event: Event) -> Optional[Order]:
        """Convert event to Order object (sync)."""
        try:
            payload = event.payload
            return Order(
                order_id=payload['order_id'],
                symbol=payload['symbol'],
                side=payload['side'],
                order_type=payload['order_type'],
                quantity=payload['quantity'],
                price=payload.get('price'),
                stop_price=payload.get('stop_price'),
                time_in_force=payload.get('time_in_force', 'day'),
                created_at=event.timestamp
            )
        except Exception as e:
            self.logger.error(f"Failed to parse order event: {e}")
            return None
    
    def _update_portfolio_with_fill(self, fill: Fill) -> None:
        """Update portfolio with fill (sync)."""
        try:
            # Convert Fill to FILL event for portfolio
            fill_event = Event(
                event_type=EventType.FILL.value,
                timestamp=fill.executed_at,
                source_id=self.config.component_id,
                payload={
                    'fill_id': fill.fill_id,
                    'order_id': fill.order_id,
                    'symbol': fill.symbol,
                    'side': fill.side.value,
                    'quantity': float(fill.quantity),
                    'price': float(fill.price),
                    'commission': float(fill.commission),
                    'executed_at': fill.executed_at.isoformat()
                }
            )
            
            # Portfolio has sync event processing
            self.portfolio.process_event(fill_event)
            
        except Exception as e:
            self.logger.error(f"Failed to update portfolio: {e}")
    
    def _sync_portfolio_positions(self, broker_positions: Dict[str, Position]) -> None:
        """Sync portfolio with broker positions (sync)."""
        # This is optional - depends on your portfolio implementation
        # Could emit position update events or directly update
        pass
    
    # Event emission helpers (sync)
    
    def _emit_order_acknowledged(self, order: Order, broker_id: str) -> None:
        """Emit order acknowledgment event."""
        # Create ACK event
        # This would go to your event bus for other components
        pass
    
    def _emit_order_rejected(self, order: Order, reason: str) -> None:
        """Emit order rejection event."""
        # Create REJECT event
        pass
    
    def _emit_fill_event(self, fill: Fill) -> None:
        """Emit fill event."""
        # Already handled in _update_portfolio_with_fill
        pass
    
    def _emit_account_update(self, account: Dict[str, Any]) -> None:
        """Emit account update event."""
        # Create ACCOUNT event if needed
        pass
    
    # Public interface for sync code
    
    def submit_order(self, order: Order) -> None:
        """
        Submit order (called from sync code).
        
        Non-blocking - returns immediately.
        """
        if not self._running:
            self.logger.warning("Engine not running, order rejected")
            return
        
        # Create ORDER event
        event = Event(
            event_type=EventType.ORDER.value,
            timestamp=datetime.now(timezone.utc),
            source_id="strategy",
            payload={
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side,
                'order_type': order.order_type,
                'quantity': order.quantity,
                'price': order.price,
                'stop_price': order.stop_price,
                'time_in_force': order.time_in_force
            }
        )
        
        # Emit to queue (thread-safe)
        self.emit_event(event)
    
    def get_stats(self) -> ExecutionStats:
        """Get execution statistics (sync)."""
        return self.stats
    
    def is_running(self) -> bool:
        """Check if engine is running (sync)."""
        return self._running


class AsyncExecutionAdapter:
    """
    Simple adapter for sync code to use async execution engine.
    
    This is the only "bridge" needed - and it's very simple.
    """
    
    def __init__(self, engine: CleanAsyncExecutionEngine):
        self.engine = engine
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start async engine in background thread."""
        import threading
        
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            # Start engine
            self._loop.run_until_complete(self.engine.start())
            
            # Keep running
            self._loop.run_forever()
        
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        
        # Wait for engine to be ready
        import time
        time.sleep(0.5)
    
    def stop(self) -> None:
        """Stop async engine."""
        if self._loop:
            # Schedule stop
            future = asyncio.run_coroutine_threadsafe(
                self.engine.stop(),
                self._loop
            )
            future.result(timeout=5)
            
            # Stop loop
            self._loop.call_soon_threadsafe(self._loop.stop)
    
    def submit_order(self, order: Order) -> None:
        """Submit order from sync code."""
        self.engine.submit_order(order)
    
    def process_event(self, event: Event) -> None:
        """Process event from sync event bus."""
        if event.event_type == EventType.ORDER.value:
            # Extract order and submit
            payload = event.payload
            order = Order(
                order_id=payload['order_id'],
                symbol=payload['symbol'],
                side=payload['side'],
                order_type=payload['order_type'],
                quantity=payload['quantity'],
                price=payload.get('price'),
                stop_price=payload.get('stop_price'),
                time_in_force=payload.get('time_in_force', 'day'),
                created_at=event.timestamp
            )
            self.submit_order(order)
    
    # Event bus integration
    def on_order(self, event: Event):
        """Handle ORDER events from event bus."""
        self.process_event(event)


# Factory function
def create_async_execution_engine(
    component_id: str,
    broker,
    portfolio=None,
    **kwargs
) -> AsyncExecutionAdapter:
    """
    Create async execution engine with sync adapter.
    
    Args:
        component_id: Unique component identifier
        broker: Async broker instance
        portfolio: Portfolio state (optional)
        **kwargs: Additional configuration
    
    Returns:
        AsyncExecutionAdapter for sync code integration
    """
    config = ExecutionConfig(
        component_id=component_id,
        **kwargs
    )
    
    engine = CleanAsyncExecutionEngine(config, broker)
    
    if portfolio:
        engine.set_portfolio(portfolio)
    
    adapter = AsyncExecutionAdapter(engine)
    
    return adapter