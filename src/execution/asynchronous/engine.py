"""
Asynchronous live trading execution engine.

Real broker integration with proper async patterns for I/O operations.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..types import Order, Fill, ExecutionStats
from ..async_protocols import AsyncBroker, OrderMonitor
from .order_manager import AsyncOrderManager

logger = logging.getLogger(__name__)


class AsyncExecutionEngine:
    """
    Asynchronous execution engine for live trading.
    
    Handles real broker communication with proper async patterns
    for network I/O, order monitoring, and real-time updates.
    """
    
    def __init__(
        self,
        component_id: str,
        broker: AsyncBroker,
        order_manager: Optional[AsyncOrderManager] = None,
        order_monitor: Optional[OrderMonitor] = None
    ):
        self.component_id = component_id
        self.broker = broker
        self.order_manager = order_manager or AsyncOrderManager(f"{component_id}_orders")
        self.order_monitor = order_monitor
        
        self.stats = ExecutionStats()
        self.logger = logger.getChild(component_id)
        
        # Async control
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._fill_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.order_poll_interval = 1.0  # seconds
        self.fill_poll_interval = 0.5   # seconds
    
    async def start(self) -> None:
        """Start the live execution engine."""
        if self._running:
            self.logger.warning("Engine already running")
            return
        
        self.logger.info(f"Starting live execution engine: {self.component_id}")
        
        try:
            # Connect to broker
            await self.broker.connect()
            
            # Start order manager
            await self.order_manager.start()
            
            # Start order monitor if available
            if self.order_monitor:
                await self.order_monitor.start_monitoring()
            
            # Start background tasks
            self._monitor_task = asyncio.create_task(self._monitor_orders())
            self._fill_task = asyncio.create_task(self._process_fills())
            
            self._running = True
            self.logger.info("Live execution engine started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start live execution engine: {e}", exc_info=True)
            await self._cleanup()
            raise
    
    async def stop(self) -> None:
        """Stop the live execution engine."""
        if not self._running:
            return
        
        self.logger.info("Stopping live execution engine")
        
        await self._cleanup()
        self._running = False
        
        self.logger.info("Live execution engine stopped")
    
    async def _cleanup(self) -> None:
        """Clean up resources."""
        # Cancel background tasks
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if self._fill_task and not self._fill_task.done():
            self._fill_task.cancel()
            try:
                await self._fill_task
            except asyncio.CancelledError:
                pass
        
        # Stop order monitor
        if self.order_monitor:
            try:
                await self.order_monitor.stop_monitoring()
            except Exception as e:
                self.logger.error(f"Error stopping order monitor: {e}")
        
        # Stop order manager
        try:
            await self.order_manager.stop()
        except Exception as e:
            self.logger.error(f"Error stopping order manager: {e}")
        
        # Disconnect from broker
        try:
            await self.broker.disconnect()
        except Exception as e:
            self.logger.error(f"Error disconnecting broker: {e}")
    
    async def submit_order(self, order: Order) -> Optional[str]:
        """
        Submit order for execution.
        
        Returns broker order ID if successful, None if failed.
        Live trading returns order ID immediately; fill comes later.
        """
        if not self._running:
            self.logger.error("Engine not running")
            return None
        
        self.logger.debug(f"Submitting order: {order.order_id}")
        
        # Validate order
        is_valid, error_msg = await self.broker.validate_order(order)
        if not is_valid:
            self.logger.warning(f"Order validation failed: {error_msg}")
            await self.order_manager.reject_order(order.order_id, error_msg or "Validation failed")
            self.stats.orders_rejected += 1
            return None
        
        try:
            # Submit to broker
            broker_order_id = await self.broker.submit_order(order)
            
            # Track order
            await self.order_manager.track_order(order, broker_order_id)
            
            # Track with order monitor if available
            if self.order_monitor:
                await self.order_monitor.track_order(order, broker_order_id)
            
            self.stats.orders_submitted += 1
            self.logger.debug(f"Order submitted: {order.order_id} -> broker ID: {broker_order_id}")
            
            return broker_order_id
            
        except Exception as e:
            self.logger.error(f"Order submission failed: {e}", exc_info=True)
            await self.order_manager.reject_order(order.order_id, str(e))
            self.stats.orders_rejected += 1
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        if not self._running:
            return False
        
        try:
            # Get broker order ID
            broker_order_id = await self.order_manager.get_broker_order_id(order_id)
            if not broker_order_id:
                self.logger.warning(f"No broker order ID found for: {order_id}")
                return False
            
            # Cancel with broker
            cancelled = await self.broker.cancel_order(broker_order_id)
            
            if cancelled:
                await self.order_manager.cancel_order(order_id)
                self.stats.orders_cancelled += 1
                self.logger.debug(f"Order cancelled: {order_id}")
            
            return cancelled
            
        except Exception as e:
            self.logger.error(f"Order cancellation failed: {e}")
            return False
    
    async def _monitor_orders(self) -> None:
        """Continuously monitor order status updates."""
        self.logger.debug("Starting order monitoring task")
        
        while self._running:
            try:
                # Get order updates from monitor
                if self.order_monitor:
                    updates = await self.order_monitor.get_order_updates()
                    
                    for update in updates:
                        await self._process_order_update(update)
                
                # Sleep before next check
                await asyncio.sleep(self.order_poll_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in order monitoring: {e}")
                await asyncio.sleep(self.order_poll_interval)
        
        self.logger.debug("Order monitoring task stopped")
    
    async def _process_fills(self) -> None:
        """Continuously process new fills from broker."""
        self.logger.debug("Starting fill processing task")
        
        while self._running:
            try:
                # Get recent fills from broker
                new_fills = await self.broker.get_recent_fills()
                
                for fill in new_fills:
                    await self._process_fill(fill)
                
                # Sleep before next check
                await asyncio.sleep(self.fill_poll_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing fills: {e}")
                await asyncio.sleep(self.fill_poll_interval)
        
        self.logger.debug("Fill processing task stopped")
    
    async def _process_order_update(self, update: Dict[str, Any]) -> None:
        """Process an order status update."""
        try:
            order_id = update.get('order_id')
            status = update.get('status')
            
            if order_id and status:
                await self.order_manager.update_order_status(order_id, status)
                self.logger.debug(f"Order status updated: {order_id} -> {status}")
        
        except Exception as e:
            self.logger.error(f"Error processing order update: {e}")
    
    async def _process_fill(self, fill: Fill) -> None:
        """Process a new fill."""
        try:
            await self.order_manager.process_fill(fill)
            self.stats.orders_filled += 1
            self.stats.total_commission += fill.commission
            
            self.logger.debug(
                f"Fill processed: {fill.fill_id} for order {fill.order_id} "
                f"({fill.quantity} @ {fill.price})"
            )
        
        except Exception as e:
            self.logger.error(f"Error processing fill: {e}")
    
    async def get_execution_stats(self) -> ExecutionStats:
        """Get execution statistics."""
        return self.stats
    
    async def get_order_status(self, order_id: str) -> Optional[str]:
        """Get order status."""
        return await self.order_manager.get_order_status(order_id)
    
    async def get_pending_orders(self) -> List[Order]:
        """Get pending orders."""
        return await self.order_manager.get_pending_orders()
    
    async def get_recent_fills(self, limit: int = 100) -> List[Fill]:
        """Get recent fills."""
        return await self.order_manager.get_recent_fills(limit)
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get current positions from broker."""
        try:
            return await self.broker.get_positions()
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information from broker."""
        try:
            return await self.broker.get_account_info()
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running
    
    async def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.stats = ExecutionStats()
        self.logger.debug("Execution statistics reset")