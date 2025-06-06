"""
Execution Container for EVENT_FLOW_ARCHITECTURE

This container:
1. Receives ORDER events from portfolio containers
2. Simulates market execution (fills, partial fills, rejections)
3. Broadcasts FILL events back to portfolios
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging
import random

from ..events import Event, EventType
from ..types.trading import Order, Fill
from .protocols import ContainerRole
from .container import Container, ContainerConfig

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStats:
    """Track execution statistics."""
    orders_received: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    total_volume: float = 0.0
    total_commission: float = 0.0


class ExecutionContainer(Container):
    """
    Execution container that simulates market execution.
    
    In the EVENT_FLOW_ARCHITECTURE, execution:
    - Receives ORDER events from all portfolios
    - Simulates realistic market conditions
    - Generates FILL events for executed orders
    - Handles partial fills and rejections
    """
    
    def __init__(self,
                 execution_config: Dict[str, Any],
                 container_id: Optional[str] = None):
        """
        Initialize execution container.
        
        Args:
            execution_config: Execution configuration
                - slippage: Price slippage model parameters
                - commission: Commission structure
                - fill_probability: Probability of order filling
                - partial_fill_probability: Probability of partial fills
            container_id: Optional container ID override
        """
        # Initialize base container
        super().__init__(ContainerConfig(
            role=ContainerRole.EXECUTION,
            name='execution',
            container_id=container_id or 'execution',
            config=execution_config,
            capabilities={'order.execution', 'fill.generation', 'market.simulation'}
        ))
        
        self.execution_config = execution_config
        
        # Set random seed for deterministic execution
        if 'random_seed' in execution_config:
            random.seed(execution_config['random_seed'])
            logger.info(f"Set random seed to {execution_config['random_seed']} for deterministic execution")
        
        # Execution stats
        self.stats = ExecutionStats()
        
        # Market data cache (for realistic pricing)
        self.last_prices: Dict[str, float] = {}
        
        # Order tracking
        self.pending_orders: List[Dict[str, Any]] = []
        self.order_history: List[Dict[str, Any]] = []
        
        logger.info(f"Created ExecutionContainer: {container_id}")
    
    async def initialize(self) -> None:
        """Initialize the execution container."""
        await super().initialize()
        
        # Subscribe to ORDER events
        self.event_bus.subscribe(EventType.ORDER, self._on_order_received)
        
        # Subscribe to BAR events for market prices
        self.event_bus.subscribe(EventType.BAR, self._on_bar_received)
        
        logger.info(f"Initialized execution container")
    
    def _on_order_received(self, event: Event) -> None:
        """Process incoming ORDER event."""
        self.stats.orders_received += 1
        
        try:
            order = event.payload.get('order', {})
            if not order:
                return
            
            logger.info(f"Execution received order: {order.get('order_id')} - {order.get('side')} {order.get('quantity')} {order.get('symbol')}")
            
            # Simulate order execution
            fill = self._execute_order(order)
            
            if fill:
                self.stats.orders_filled += 1
                self.stats.total_volume += fill['quantity'] * fill['price']
                self.stats.total_commission += fill.get('commission', 0)
                
                logger.info(f"Execution ABOUT TO publish FILL event for {order.get('order_id')}")
                # Broadcast FILL event
                self.event_bus.publish(Event(
                    event_type=EventType.FILL,
                    payload={'fill': fill},
                    source_id=self.container_id
                ))
                logger.info(f"Execution FINISHED publishing FILL event for {order.get('order_id')}")
                
                logger.info(f"Executed order {order.get('order_id')}: filled {fill['quantity']} @ {fill['price']}")
            else:
                self.stats.orders_rejected += 1
                logger.debug(f"Order {order.get('order_id')} rejected")
                
        except Exception as e:
            logger.error(f"Error processing order: {e}")
    
    def _on_bar_received(self, event: Event) -> None:
        """Update market prices from BAR events."""
        try:
            bar = event.payload.get('bar')
            symbol = event.payload.get('symbol')
            
            if bar and symbol:
                # Update last known price
                if hasattr(bar, 'close'):
                    self.last_prices[symbol] = bar.close
                elif isinstance(bar, dict):
                    self.last_prices[symbol] = bar.get('close', bar.get('price', 0))
                    
        except Exception as e:
            logger.error(f"Error updating market prices: {e}")
    
    def _execute_order(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Simulate order execution with realistic market conditions.
        
        Returns fill dict or None if order rejected.
        """
        symbol = order.get('symbol')
        side = order.get('side')
        quantity = order.get('quantity', 0)
        order_type = order.get('order_type', 'market')
        
        # Get current market price
        base_price = self._get_market_price(symbol, order)
        if base_price <= 0:
            logger.warning(f"No valid price for {symbol}")
            return None
        
        # Check fill probability
        fill_prob = self.execution_config.get('fill_probability', 0.98)
        if random.random() > fill_prob:
            logger.debug(f"Order randomly rejected (fill_prob={fill_prob})")
            return None
        
        # Calculate execution price with slippage
        exec_price = self._calculate_execution_price(base_price, side, quantity)
        
        # Check for partial fills
        partial_prob = self.execution_config.get('partial_fill_probability', 0.1)
        if random.random() < partial_prob:
            # Partial fill - reduce quantity
            fill_ratio = random.uniform(0.5, 0.95)
            quantity = int(quantity * fill_ratio)
            logger.debug(f"Partial fill: {quantity} of {order.get('quantity')}")
        
        # Calculate commission
        commission = self._calculate_commission(quantity, exec_price)
        
        # Create fill
        fill = {
            'fill_id': f"fill_{order.get('order_id')}_{datetime.now().timestamp()}",
            'order_id': order.get('order_id'),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': exec_price,
            'commission': commission,
            'timestamp': datetime.now(),
            'portfolio_id': order.get('portfolio_id'),
            'metadata': {
                'execution_type': 'simulated',
                'slippage': exec_price - base_price
            }
        }
        
        return fill
    
    def _get_market_price(self, symbol: str, order: Dict[str, Any]) -> float:
        """Get current market price for symbol."""
        # First check if order has a reference price
        if 'price' in order and order['price'] > 0:
            return order['price']
        
        # Then check our price cache
        if symbol in self.last_prices:
            return self.last_prices[symbol]
        
        # Default
        return 100.0
    
    def _calculate_execution_price(self, base_price: float, side: str, quantity: float) -> float:
        """Calculate execution price with slippage."""
        slippage_config = self.execution_config.get('slippage', {})
        
        # Base slippage (as percentage)
        base_slippage = slippage_config.get('base', 0.0001)  # 0.01% default
        
        # Volume-based slippage (larger orders have more impact)
        volume_factor = slippage_config.get('volume_factor', 0.00001)
        volume_slippage = volume_factor * quantity
        
        # Random component
        random_factor = slippage_config.get('random_factor', 0.0001)
        random_slippage = random.uniform(-random_factor, random_factor)
        
        # Total slippage
        total_slippage = base_slippage + volume_slippage + random_slippage
        
        # Apply slippage (adverse for buyer/seller)
        if side == 'buy':
            exec_price = base_price * (1 + total_slippage)
        else:
            exec_price = base_price * (1 - total_slippage)
        
        return round(exec_price, 2)
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate trading commission."""
        commission_config = self.execution_config.get('commission', {})
        
        # Per-share commission
        per_share = commission_config.get('per_share', 0.005)
        share_commission = quantity * per_share
        
        # Minimum commission
        min_commission = commission_config.get('minimum', 1.0)
        
        # Maximum commission
        max_commission = commission_config.get('maximum', 5.0)
        
        # Calculate total
        commission = max(min_commission, min(share_commission, max_commission))
        
        return round(commission, 2)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'orders_received': self.stats.orders_received,
            'orders_filled': self.stats.orders_filled,
            'orders_rejected': self.stats.orders_rejected,
            'fill_rate': self.stats.orders_filled / max(1, self.stats.orders_received),
            'total_volume': self.stats.total_volume,
            'total_commission': self.stats.total_commission,
            'avg_commission': self.stats.total_commission / max(1, self.stats.orders_filled)
        }


def create_execution_container(
    execution_config: Optional[Dict[str, Any]] = None
) -> ExecutionContainer:
    """
    Factory function to create an ExecutionContainer.
    
    This is the preferred way to create execution containers
    in the EVENT_FLOW_ARCHITECTURE.
    """
    default_config = {
        'fill_probability': 0.98,
        'partial_fill_probability': 0.05,
        'slippage': {
            'base': 0.0001,  # 0.01% base slippage
            'volume_factor': 0.00001,
            'random_factor': 0.0001
        },
        'commission': {
            'per_share': 0.005,
            'minimum': 1.0,
            'maximum': 5.0
        }
    }
    
    if execution_config:
        default_config.update(execution_config)
    
    return ExecutionContainer(default_config)