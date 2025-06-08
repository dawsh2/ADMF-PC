"""
Container components for modular container composition.

This module provides reusable components that can be added to containers
to build specific functionality through composition.
"""

from typing import Protocol, Dict, Any, Optional, List, Callable
from abc import abstractmethod
import logging
from dataclasses import dataclass
from datetime import datetime

from ..types.events import Event, EventType
from ..types.trading import Bar, Signal, Order, Fill, Position

logger = logging.getLogger(__name__)


class ContainerComponent(Protocol):
    """Base protocol for all container components."""
    
    @abstractmethod
    def initialize(self, container: 'Container') -> None:
        """Initialize component with parent container reference."""
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Start the component."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the component."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get component state."""
        pass


@dataclass
class DataStreamer:
    """Component for streaming market data."""
    symbol: str
    timeframe: str
    data_source: Optional[Any] = None
    
    def initialize(self, container: 'Container') -> None:
        self.container = container
        self.current_index = 0
        
    def start(self) -> None:
        logger.info(f"Starting data streamer for {self.symbol}_{self.timeframe}")
        
    def stop(self) -> None:
        logger.info(f"Stopping data streamer for {self.symbol}_{self.timeframe}")
        
    def stream_next_bar(self) -> Optional[Bar]:
        """Stream next bar and publish BAR event with enhanced metadata."""
        if self.data_source and self.current_index < len(self.data_source):
            bar = self.data_source[self.current_index]
            self.current_index += 1
            
            # Publish enhanced BAR event with timing metadata
            event = Event(
                event_type=EventType.BAR,
                payload={
                    'bar': bar,
                    'symbol': self.symbol,
                    'timeframe': self.timeframe,
                    'bar_close_time': bar.timestamp,  # When bar period ended
                    'is_complete': True,  # Historical bars are complete
                    # Include OHLCV data directly for convenience
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                }
            )
            self.container.event_bus.publish(event)
            return bar
        return None
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'current_index': self.current_index
        }


@dataclass
class FeatureCalculator:
    """Component for calculating technical indicators."""
    indicators: List[Dict[str, Any]]
    lookback_window: int = 100
    
    def initialize(self, container: 'Container') -> None:
        self.container = container
        self.price_history = []
        self.feature_cache = {}
        
    def start(self) -> None:
        # Subscribe to BAR events
        self.container.event_bus.subscribe(EventType.BAR, self.on_bar)
        logger.info(f"Started feature calculator with {len(self.indicators)} indicators")
        
    def stop(self) -> None:
        logger.info("Stopping feature calculator")
        
    def on_bar(self, event: Event) -> None:
        """Calculate features when new bar arrives."""
        bar = event.payload.get('bar')
        if not bar:
            return
            
        # Update price history
        self.price_history.append(bar)
        if len(self.price_history) > self.lookback_window:
            self.price_history.pop(0)
            
        # Calculate features
        features = self.calculate_features()
        
        # Publish FEATURES event
        features_event = Event(
            event_type=EventType.FEATURES,
            payload={
                'features': features,
                'symbol': event.payload.get('symbol'),
                'bar': bar
            }
        )
        # Publish to parent (symbol_timeframe container will handle routing to root)
        self.container.publish_event(features_event, target_scope="parent")
        
    def calculate_features(self) -> Dict[str, float]:
        """Calculate all configured indicators."""
        features = {}
        
        # Placeholder - actual implementation would calculate real indicators
        for indicator in self.indicators:
            indicator_name = indicator['name']
            features[indicator_name] = 0.0  # Replace with actual calculation
            
        return features
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'indicators': self.indicators,
            'history_length': len(self.price_history),
            'features_calculated': len(self.feature_cache)
        }


@dataclass
class PortfolioState:
    """Component for managing portfolio state."""
    initial_capital: float = 100000.0
    
    def initialize(self, container: 'Container') -> None:
        self.container = container
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Order] = []
        self.order_history: List[Order] = []
        self.fill_history: List[Fill] = []
        
    def start(self) -> None:
        # Subscribe to FILL events
        self.container.event_bus.subscribe(EventType.FILL, self.on_fill)
        logger.info(f"Started portfolio state with capital ${self.initial_capital:,.2f}")
        
    def stop(self) -> None:
        logger.info("Stopping portfolio state")
        
    def on_fill(self, event: Event) -> None:
        """Update positions when fill is received."""
        fill = event.payload.get('fill')
        if not fill:
            return
            
        # Update position
        symbol = fill.get('symbol')
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol, quantity=0, avg_price=0)
            
        position = self.positions[symbol]
        # Update position logic here
        
        self.fill_history.append(fill)
        
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total = self.cash
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total += position.quantity * current_prices[symbol]
        return total
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'cash': self.cash,
            'positions': len(self.positions),
            'pending_orders': len(self.pending_orders),
            'total_fills': len(self.fill_history)
        }


@dataclass
class SignalProcessor:
    """Component for processing trading signals."""
    
    def initialize(self, container: 'Container') -> None:
        self.container = container
        self.signal_count = 0
        # Get strategy assignments from container config
        self.strategy_assignments = self.container.config.config.get('strategy_assignments', [])
        
    def start(self) -> None:
        # Subscribe to SIGNAL events
        self.container.event_bus.subscribe(EventType.SIGNAL, self.on_signal)
        logger.info("Started signal processor")
        
    def stop(self) -> None:
        logger.info("Stopping signal processor")
        
    def on_signal(self, event: Event) -> None:
        """Process incoming signals."""
        signal = event.payload.get('signal')
        if not signal:
            return
            
        self.signal_count += 1
        
        # Check if this signal is from a strategy we're subscribed to
        strategy_id = event.payload.get('strategy_id')
        if strategy_id not in self.strategy_assignments:
            return
            
        # Process signal and potentially generate order
        # This would interact with PortfolioState component
        logger.debug(f"Processing signal #{self.signal_count}: {signal}")
        
    def get_state(self) -> Dict[str, Any]:
        return {
            'signals_processed': self.signal_count
        }


@dataclass
class OrderGenerator:
    """Component for generating orders from signals."""
    
    def initialize(self, container: 'Container') -> None:
        self.container = container
        self.orders_generated = 0
        
    def start(self) -> None:
        logger.info("Started order generator")
        
    def stop(self) -> None:
        logger.info("Stopping order generator")
        
    def generate_order(self, signal: Dict[str, Any], portfolio_state: PortfolioState) -> Optional[Order]:
        """Generate order from signal considering portfolio state.
        
        Multi-asset signal convention examples:
        1. Single asset signal:
           signal = {
               'symbol': 'AAPL',
               'direction': 'BUY',
               'strategy_id': 'momentum_v1',
               'bar_data': {'AAPL_1m': {...}},
               'features': {...}
           }
        
        2. Pairs trading signal (multiple orders):
           signal = {
               'strategy_id': 'pairs_trade',
               'orders': [
                   {'symbol': 'AAPL', 'direction': 'BUY', 'ratio': 1},
                   {'symbol': 'MSFT', 'direction': 'SELL', 'ratio': 1.2}
               ],
               'bar_data': {'AAPL_5m': {...}, 'MSFT_5m': {...}},
               'features': {...}
           }
        """
        # Order generation logic
        self.orders_generated += 1
        
        # TODO: Implement actual order creation logic
        order = None  # Replace with actual order creation
        
        # Publish ORDER_REQUEST event with enriched context
        order_event = Event(
            event_type=EventType.ORDER_REQUEST,
            payload={
                'order': order,
                'portfolio_id': self.container.metadata.container_id,
                'strategy_id': signal.get('strategy_id'),
                'classification': signal.get('classification'),  # For risk validation
                'bar_data': signal.get('bar_data', {})  # Context for risk assessment
            }
        )
        # Always publish to parent - routing will handle the rest
        self.container.publish_event(order_event, target_scope="parent")
        
        return order
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'orders_generated': self.orders_generated
        }


@dataclass
class RiskValidator:
    """Component for validating orders against risk limits."""
    max_position_size: float = 0.1
    max_portfolio_risk: float = 0.02
    
    def initialize(self, container: 'Container') -> None:
        self.container = container
        self.validations_performed = 0
        self.rejections = 0
        
    def start(self) -> None:
        # Subscribe to ORDER_REQUEST events
        self.container.event_bus.subscribe(EventType.ORDER_REQUEST, self.on_order_request)
        logger.info("Started risk validator")
        
    def stop(self) -> None:
        logger.info("Stopping risk validator")
        
    def on_order_request(self, event: Event) -> None:
        """Validate order against risk limits."""
        order = event.payload.get('order')
        if not order:
            return
            
        self.validations_performed += 1
        
        # Perform risk validation
        approved = self.validate_order(order)
        
        if approved:
            # Publish approved ORDER event
            approved_event = Event(
                event_type=EventType.ORDER,
                payload=event.payload
            )
            # Publish locally - routes will handle distribution
            self.container.event_bus.publish(approved_event)
        else:
            self.rejections += 1
            logger.warning(f"Order rejected by risk validator: {order}")
            # Could also publish RISK_REJECTED event
            rejected_event = Event(
                event_type=EventType.RISK_REJECTED,
                payload=event.payload
            )
            self.container.event_bus.publish(rejected_event)
    
    def validate_order(self, order: Order) -> bool:
        """Validate order against risk limits.
        
        TODO: Enhanced risk validation examples using enriched signal data:
        
        1. Use bar_data from signal (passed in ORDER_REQUEST payload):
           bar_data = self.current_event.payload.get('bar_data', {})
           current_price = bar_data.get(f"{order.symbol}_1m", {}).get('close')
           
        2. Classification-aware risk:
           classification = self.current_event.payload.get('classification')
           if classification == 'high_volatility':
               max_position_size = self.max_position_size * 0.5  # Reduce in volatile markets
               
        3. Multi-asset correlation checks:
           # Check if multiple positions are correlated
           bar_data = self.current_event.payload.get('bar_data', {})
           if 'SPY_1m' in bar_data and 'QQQ_1m' in bar_data:
               # Both tech-heavy ETFs, check combined exposure
               pass
        """
        # Placeholder validation logic
        return True
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'validations': self.validations_performed,
            'rejections': self.rejections,
            'rejection_rate': self.rejections / max(1, self.validations_performed)
        }


@dataclass
class ExecutionEngine:
    """Component for executing orders and generating fills."""
    slippage_model: Optional[Any] = None
    commission_model: Optional[Any] = None
    
    def initialize(self, container: 'Container') -> None:
        self.container = container
        self.orders_processed = 0
        self.fills_generated = 0
        
    def start(self) -> None:
        # Subscribe to ORDER events
        self.container.event_bus.subscribe(EventType.ORDER, self.on_order)
        logger.info("Started execution engine")
        
    def stop(self) -> None:
        logger.info("Stopping execution engine")
        
    def on_order(self, event: Event) -> None:
        """Execute order and generate fill."""
        order = event.payload.get('order')
        if not order:
            return
            
        self.orders_processed += 1
        
        # Execute order (simplified)
        fill = self.execute_order(order)
        
        if fill:
            self.fills_generated += 1
            # Publish FILL event
            fill_event = Event(
                event_type=EventType.FILL,
                payload={'fill': fill, 'portfolio_id': event.payload.get('portfolio_id')}
            )
            self.container.event_bus.publish(fill_event)
    
    def execute_order(self, order: Order) -> Optional[Fill]:
        """Execute order with slippage and commission models."""
        # Placeholder execution logic
        return Fill(
            order_id=order.id,
            symbol=order.symbol,
            quantity=order.quantity,
            price=100.0,  # Placeholder
            commission=0.0,
            timestamp=datetime.now()
        )
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'orders_processed': self.orders_processed,
            'fills_generated': self.fills_generated,
            'fill_rate': self.fills_generated / max(1, self.orders_processed)
        }


# Import our signal generation/replay components
# These would normally be in separate files but adding here for completeness
try:
    from .components.signal_generator import SignalGeneratorComponent
    from .components.signal_streamer import SignalStreamerComponent, BoundaryAwareReplay
except ImportError:
    # If not in separate files, they should be added here
    pass
