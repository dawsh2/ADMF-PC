"""
Portfolio Container for EVENT_FLOW_ARCHITECTURE

This container:
1. Receives FEATURES events from Symbol-Timeframe containers
2. Uses stateless strategy services to generate signals
3. Maintains portfolio state (positions, cash, P&L)
4. Generates ORDER events for execution
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..events import Event, EventType
from ..types.trading import Order, Position, Signal
from .protocols import ContainerRole
from .container import Container, ContainerConfig
from ..tracing import trace, TracePoint

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Current state of the portfolio."""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    pending_orders: List[Order] = field(default_factory=list)
    historical_values: List[float] = field(default_factory=list)
    last_update: Optional[datetime] = None
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value


class PortfolioContainer(Container):
    """
    Portfolio container that processes FEATURES events and manages positions.
    
    In the EVENT_FLOW_ARCHITECTURE, portfolios:
    - Subscribe to FEATURES events from symbol containers
    - Use stateless strategy services to generate signals
    - Maintain position and cash state
    - Generate ORDER events for execution
    """
    
    def __init__(self,
                 combo_id: str,
                 strategy_params: Dict[str, Any],
                 risk_params: Dict[str, Any],
                 initial_capital: float = 100000,
                 container_id: Optional[str] = None):
        """
        Initialize portfolio container.
        
        Args:
            combo_id: Unique identifier for this parameter combination
            strategy_params: Strategy configuration
            risk_params: Risk management configuration
            initial_capital: Starting cash amount
            container_id: Optional container ID override
        """
        # Initialize base container
        super().__init__(ContainerConfig(
            role=ContainerRole.PORTFOLIO,
            name=f'portfolio_{combo_id}',
            container_id=container_id or f'portfolio_{combo_id}',
            config={
                'combo_id': combo_id,
                'strategy_params': strategy_params,
                'risk_params': risk_params,
                'initial_capital': initial_capital
            },
            capabilities={'portfolio.management', 'signal.processing', 'order.generation'}
        ))
        
        self.combo_id = combo_id
        self.strategy_params = strategy_params
        self.risk_params = risk_params
        
        # Portfolio state
        self.portfolio_state = PortfolioState(cash=initial_capital)
        
        # Metrics tracking
        self.metrics = {
            'trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_value': initial_capital
        }
        
        # Event tracking for testing
        self._signals_received = 0  # Now tracking SIGNALS instead of FEATURES
        self._signals_generated = 0
        self._orders_created = 0
        
        logger.info(f"Created PortfolioContainer: {container_id} with ${initial_capital}")
    
    async def initialize(self) -> None:
        """Initialize the portfolio container."""
        await super().initialize()
        
        # Subscribe to SIGNAL events (from stateless dispatcher)
        self.event_bus.subscribe(EventType.SIGNAL, self._on_signal_received)
        
        # Subscribe to FILL events
        self.event_bus.subscribe(EventType.FILL, self._on_fill_received)
        
        logger.info(f"Initialized portfolio {self.combo_id}")
    
    
    def _on_signal_received(self, event: Event) -> None:
        """Process SIGNAL event from stateless dispatcher."""
        self._signals_received += 1
        
        try:
            # Check if this signal is targeted at this portfolio
            target_combo_id = event.payload.get('combo_id')
            if target_combo_id != self.combo_id:
                return  # Signal not for this portfolio
                
            signal = event.payload.get('signal', {})
            bar = event.payload.get('bar')
            symbol = event.payload.get('symbol')
            correlation_id = event.payload.get('correlation_id')
            
            if not all([signal, bar, symbol]):
                logger.warning(f"Portfolio {self.combo_id} received incomplete SIGNAL event")
                return
            
            logger.info(f"Portfolio {self.combo_id} received SIGNAL: {signal.get('direction')} for {symbol}")
            
            # Process the signal (signal already generated by stateless service)
            if signal.get('direction') != 'flat':
                self._signals_generated += 1
                self._process_signal(signal, bar, correlation_id)
            
            # Update portfolio valuation
            self._update_valuation(bar)
            
        except Exception as e:
            logger.error(f"Error processing signal in portfolio {self.combo_id}: {e}")
    
    def _process_signal(self, signal: Dict[str, Any], bar: Dict[str, Any], correlation_id: Optional[str] = None) -> None:
        """Process a trading signal and potentially generate an order."""
        symbol = signal.get('symbol', bar.get('symbol'))
        direction = signal.get('direction')  # 'long' or 'short'
        strength = signal.get('strength', 1.0)
        
        logger.debug(f"Processing signal: {direction} for {symbol}, strength={strength}")
        
        # Calculate position size based on risk parameters
        position_size = self._calculate_position_size(symbol, direction, strength, bar)
        logger.debug(f"Calculated position size: {position_size}")
        
        if position_size > 0:
            # Create order
            order = self._create_order(symbol, direction, position_size, bar)
            logger.debug(f"Created order: {order['order_id']} for {order['quantity']} shares")
            
            # Trace order creation
            correlation_id = trace(TracePoint.ORDER_CREATE, "portfolio_container.py", {
                'symbol': symbol,
                'direction': direction,
                'quantity': position_size,
                'order_id': order['order_id']
            }, correlation_id)
            
            # Convert portfolio state for risk service
            portfolio_state_dict = {
                'cash': self.portfolio_state.cash,
                'total_value': self.portfolio_state.total_value,
                'positions': {
                    symbol: {
                        'quantity': pos.quantity,
                        'avg_price': pos.avg_price,
                        'current_price': pos.current_price,
                        'market_value': pos.market_value
                    }
                    for symbol, pos in self.portfolio_state.positions.items()
                },
                'metrics': self.metrics
            }
            
            # Add to pending orders (will be confirmed when we get FILL)
            self.portfolio_state.pending_orders.append(order)
            self._orders_created += 1
            
            # Emit ORDER_REQUEST event to isolated bus for risk validation
            try:
                logger.info(f"Portfolio {self.combo_id} publishing ORDER_REQUEST event for {direction} {symbol}")
                order_request_event = Event(
                    event_type=EventType.ORDER_REQUEST,
                    payload={
                        'order': order,
                        'portfolio_state': portfolio_state_dict,
                        'risk_params': self.risk_params,
                        'market_data': bar
                    },
                    source_id=self.container_id
                )
                # Publish to isolated event bus only
                self.event_bus.publish(order_request_event)
                logger.info(f"Portfolio {self.combo_id} published ORDER_REQUEST to isolated bus")
            except Exception as e:
                logger.error(f"Failed to publish ORDER_REQUEST event: {e}")
    
    def _calculate_position_size(self, symbol: str, direction: str, strength: float, bar: Dict[str, Any]) -> float:
        """Calculate position size based on risk parameters."""
        # Simple position sizing based on risk parameters
        max_position_pct = self.risk_params.get('max_position_percent', self.risk_params.get('max_position_size', 0.1))
        
        # Scale by signal strength
        position_pct = max_position_pct * strength
        
        # Calculate dollar amount
        position_value = self.portfolio_state.total_value * position_pct
        
        # Convert to shares (assuming we have price in bar)
        price = bar.get('close', 100)  # Default price if not available
        shares = int(position_value / price)
        
        # Ensure at least 1 share if we have a signal and can afford it
        if shares == 0 and position_value > 0 and self.portfolio_state.cash >= price:
            shares = 1
            logger.debug(f"Position sizing adjusted to minimum 1 share for {self.combo_id}")
        
        logger.debug(f"Position sizing for {self.combo_id}: max_pct={max_position_pct}, strength={strength}, "
                    f"position_pct={position_pct}, position_value=${position_value:.2f}, price=${price:.2f}, shares={shares}")
        
        # Check if we have enough cash
        required_cash = shares * price
        if required_cash > self.portfolio_state.cash:
            # Adjust to available cash
            shares = int(self.portfolio_state.cash / price)
        
        return shares
    
    def _create_order(self, symbol: str, direction: str, quantity: float, bar: Dict[str, Any]) -> Dict[str, Any]:
        """Create an order dictionary."""
        # Convert direction to side for risk validator compatibility
        side = 'buy' if direction == 'long' else 'sell'
        
        return {
            'order_id': f"{self.combo_id}_{symbol}_{datetime.now().timestamp()}",
            'symbol': symbol,
            'direction': direction,  # Keep for internal use
            'side': side,  # For risk validator
            'quantity': quantity,
            'order_type': 'market',
            'price': bar.get('close', 100),  # For market orders, use current price as reference
            'timestamp': datetime.now(),
            'portfolio_id': self.combo_id,
            'metadata': {
                'strategy': self.strategy_params.get('type'),
                'signal_strength': 1.0
            }
        }
    
    def _on_fill_received(self, event: Event) -> None:
        """Process FILL event and update portfolio state."""
        try:
            fill = event.payload.get('fill', {})
            logger.info(f"Portfolio {self.combo_id} STARTED processing fill for {fill.get('symbol', 'unknown')}")
            
            # Check if this fill is for our portfolio
            if fill.get('portfolio_id') != self.combo_id:
                return
            
            # Update positions
            symbol = fill.get('symbol')
            quantity = fill.get('quantity', 0)
            price = fill.get('price', 0)
            side = fill.get('side')  # 'buy' or 'sell'
            
            # Update cash
            if side == 'buy':
                self.portfolio_state.cash -= quantity * price
            else:  # sell
                self.portfolio_state.cash += quantity * price
            
            # Update or create position
            if symbol in self.portfolio_state.positions:
                position = self.portfolio_state.positions[symbol]
                old_quantity = position.quantity
                
                if side == 'buy':
                    position.quantity += quantity
                else:  # sell
                    position.quantity -= quantity
                
                # Update average price (only for adding to position)
                if abs(position.quantity) > abs(old_quantity):
                    # Position increased
                    position.avg_price = ((position.avg_price * abs(old_quantity)) + 
                                        (price * quantity)) / abs(position.quantity)
                
                # Remove position if flat
                if position.quantity == 0:
                    del self.portfolio_state.positions[symbol]
                    logger.info(f"Position closed for {symbol}")
            else:
                # Create new position
                self.portfolio_state.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity if side == 'buy' else -quantity,
                    avg_price=price,
                    current_price=price,
                    market_value=quantity * price
                )
            
            # Remove from pending orders
            self.portfolio_state.pending_orders = [o for o in self.portfolio_state.pending_orders 
                                       if o.get('order_id') != fill.get('order_id')]
            
            # Update metrics
            self._update_trade_metrics(fill)
            
            logger.info(f"Portfolio {self.combo_id} processed fill for {symbol}: "
                       f"{side} {quantity} @ ${price}")
            
        except Exception as e:
            logger.error(f"Error processing fill in portfolio {self.combo_id}: {e}")
    
    def _update_valuation(self, bar: Dict[str, Any]) -> None:
        """Update portfolio valuation with latest prices."""
        symbol = bar.get('symbol')
        price = bar.get('close', 0)
        
        # Update position prices
        if symbol in self.portfolio_state.positions:
            position = self.portfolio_state.positions[symbol]
            position.current_price = price
            position.market_value = position.quantity * price
        
        # Track portfolio value
        current_value = self.portfolio_state.total_value
        self.portfolio_state.historical_values.append(current_value)
        self.portfolio_state.last_update = datetime.now()
        
        # Update drawdown
        if current_value > self.metrics['peak_value']:
            self.metrics['peak_value'] = current_value
        else:
            drawdown = (self.metrics['peak_value'] - current_value) / self.metrics['peak_value']
            if drawdown > self.metrics['max_drawdown']:
                self.metrics['max_drawdown'] = drawdown
    
    def _update_trade_metrics(self, fill: Dict[str, Any]) -> None:
        """Update trade-related metrics."""
        self.metrics['trades'] += 1
        
        # This is simplified - real implementation would track P&L per trade
        # For now, just count trades
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current portfolio state information."""
        state = {}
        if hasattr(super(), 'get_state_info'):
            state = super().get_state_info()
        state.update({
            'combo_id': self.combo_id,
            'cash': self.portfolio_state.cash,
            'total_value': self.portfolio_state.total_value,
            'positions': len(self.portfolio_state.positions),
            'pending_orders': len(self.portfolio_state.pending_orders),
            'metrics': self.metrics.copy(),
            '_signals_received': self._signals_received,
            '_signals_generated': self._signals_generated,
            '_orders_created': self._orders_created
        })
        return state
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get portfolio performance metrics."""
        initial_capital = self.config.config.get('initial_capital', 100000)
        current_value = self.portfolio_state.total_value
        
        return {
            'total_value': current_value,
            'total_return': (current_value - initial_capital) / initial_capital,
            'trades': self.metrics['trades'],
            'max_drawdown': self.metrics['max_drawdown'],
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price,
                    'pnl': (pos.current_price - pos.avg_price) * pos.quantity
                }
                for symbol, pos in self.portfolio_state.positions.items()
            }
        }
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(self.portfolio_state.historical_values) < 2:
            return 0.0
        
        # Calculate returns
        returns = []
        for i in range(1, len(self.portfolio_state.historical_values)):
            prev_val = self.portfolio_state.historical_values[i-1]
            curr_val = self.portfolio_state.historical_values[i]
            if prev_val > 0:
                returns.append((curr_val - prev_val) / prev_val)
        
        if not returns:
            return 0.0
        
        # Simple Sharpe calculation
        import numpy as np
        returns_array = np.array(returns)
        if returns_array.std() == 0:
            return 0.0
        
        # Annualized Sharpe (assuming daily returns)
        return (returns_array.mean() / returns_array.std()) * np.sqrt(252)
    
    async def close_all_positions(self, final_prices: Dict[str, float]) -> None:
        """Close all open positions at given prices (for end of backtest)."""
        logger.info(f"Closing all positions for portfolio {self.combo_id}")
        
        for symbol, position in list(self.portfolio_state.positions.items()):
            if position.quantity != 0:
                # Generate closing order
                close_quantity = -position.quantity  # Opposite of current position
                price = final_prices.get(symbol, position.current_price)
                
                logger.info(f"Closing position: {symbol} {position.quantity} shares @ ${price:.2f}")
                
                # Create closing order
                order = {
                    'order_id': f'{self.combo_id}_{symbol}_close_{datetime.now().timestamp()}',
                    'symbol': symbol,
                    'quantity': abs(close_quantity),
                    'side': 'buy' if close_quantity > 0 else 'sell',
                    'order_type': 'market',
                    'price': price,
                    'timestamp': datetime.now(),
                    'portfolio_id': self.combo_id  # Important for fill routing
                }
                
                # Publish ORDER_REQUEST event for closing (also needs risk validation)
                order_request_event = Event(
                    event_type=EventType.ORDER_REQUEST,
                    payload={
                        'order': order,
                        'portfolio_state': {
                            'cash': self.portfolio_state.cash,
                            'total_value': self.portfolio_state.total_value,
                            'positions': {
                                s: {
                                    'quantity': p.quantity,
                                    'avg_price': p.avg_price,
                                    'current_price': p.current_price,
                                    'market_value': p.market_value
                                }
                                for s, p in self.portfolio_state.positions.items()
                            },
                            'metrics': self.metrics
                        },
                        'risk_params': self.risk_params,
                        'market_data': {'symbol': symbol, 'close': price}
                    },
                    source_id=self.container_id
                )
                # Publish to isolated event bus only
                self.event_bus.publish(order_request_event)
                
                self._orders_created += 1
    
    # Methods for compatibility with current architecture
    
    def has_pending_orders(self) -> bool:
        """Check if there are pending orders."""
        return len(self.portfolio_state.pending_orders) > 0
    
    async def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get list of pending orders."""
        return self.portfolio_state.pending_orders.copy()
    
    async def process_signal(self, signal: Dict[str, Any]) -> None:
        """Process a signal (for compatibility)."""
        # In new architecture, signals come through FEATURES events
        # This method is for backward compatibility
        pass
    
    async def update_market_prices(self, bar: Dict[str, Any]) -> None:
        """Update market prices (for compatibility)."""
        self._update_valuation(bar)
    
    async def reject_order(self, order: Dict[str, Any], reason: str) -> None:
        """Reject an order."""
        self.portfolio_state.pending_orders = [o for o in self.portfolio_state.pending_orders 
                                   if o.get('order_id') != order.get('order_id')]
        logger.info(f"Portfolio {self.combo_id} rejected order: {reason}")