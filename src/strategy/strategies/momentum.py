"""
Momentum trading strategy implementation.

This demonstrates a pure protocol-based strategy with NO inheritance.
The strategy can be enhanced with capabilities through the ComponentFactory.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from ..protocols import Strategy, SignalDirection


class MomentumStrategy:
    """
    Momentum-based trading strategy.
    
    This is a simple class with no inheritance. It implements the
    Strategy protocol methods, making it compatible with the system.
    
    Features:
    - Price momentum calculation
    - RSI-based signals
    - No inheritance required
    """
    
    def __init__(self, 
                 lookback_period: int = 20,
                 momentum_threshold: float = 0.02,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70):
        """
        Initialize momentum strategy.
        
        Args:
            lookback_period: Period for momentum calculation
            momentum_threshold: Minimum momentum for signal
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
        """
        # Parameters
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # State
        self.price_history: List[float] = []
        self.rsi_values: List[float] = []
        self.last_signal_time: Optional[datetime] = None
        self.signal_cooldown = 3600  # 1 hour in seconds
        
        # Internal calculation state
        self._gains: List[float] = []
        self._losses: List[float] = []
        
    @property
    def name(self) -> str:
        """Strategy name for identification."""
        return "momentum_strategy"
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal from market data.
        
        This method implements the Strategy protocol.
        """
        # Extract price
        price = market_data.get('close', market_data.get('price'))
        timestamp = market_data.get('timestamp', datetime.now())
        symbol = market_data.get('symbol', 'UNKNOWN')
        
        if price is None:
            return None
        
        # Update price history
        self.price_history.append(price)
        if len(self.price_history) > self.lookback_period * 2:
            self.price_history.pop(0)
        
        # Need enough data
        if len(self.price_history) < self.lookback_period:
            return None
        
        # Check cooldown
        if self.last_signal_time:
            time_since_last = (timestamp - self.last_signal_time).total_seconds()
            if time_since_last < self.signal_cooldown:
                return None
        
        # Calculate momentum
        momentum = self._calculate_momentum()
        
        # Calculate RSI
        rsi = self._calculate_rsi(price)
        
        # Generate signal based on momentum and RSI
        signal = None
        
        if momentum > self.momentum_threshold and rsi < self.rsi_overbought:
            # Bullish momentum, not overbought
            signal = {
                'symbol': symbol,
                'direction': SignalDirection.BUY,
                'strength': min(momentum / (self.momentum_threshold * 2), 1.0),
                'timestamp': timestamp,
                'metadata': {
                    'momentum': momentum,
                    'rsi': rsi,
                    'reason': 'Positive momentum with room to run'
                }
            }
            
        elif momentum < -self.momentum_threshold and rsi > self.rsi_oversold:
            # Bearish momentum, not oversold
            signal = {
                'symbol': symbol,
                'direction': SignalDirection.SELL,
                'strength': min(abs(momentum) / (self.momentum_threshold * 2), 1.0),
                'timestamp': timestamp,
                'metadata': {
                    'momentum': momentum,
                    'rsi': rsi,
                    'reason': 'Negative momentum with room to fall'
                }
            }
            
        elif rsi < self.rsi_oversold and momentum > 0:
            # Oversold with positive momentum - potential reversal
            signal = {
                'symbol': symbol,
                'direction': SignalDirection.BUY,
                'strength': 0.5,  # Lower confidence for reversal
                'timestamp': timestamp,
                'metadata': {
                    'momentum': momentum,
                    'rsi': rsi,
                    'reason': 'Oversold reversal signal'
                }
            }
            
        elif rsi > self.rsi_overbought and momentum < 0:
            # Overbought with negative momentum - potential reversal
            signal = {
                'symbol': symbol,
                'direction': SignalDirection.SELL,
                'strength': 0.5,  # Lower confidence for reversal
                'timestamp': timestamp,
                'metadata': {
                    'momentum': momentum,
                    'rsi': rsi,
                    'reason': 'Overbought reversal signal'
                }
            }
        
        if signal:
            self.last_signal_time = timestamp
        
        return signal
    
    def _calculate_momentum(self) -> float:
        """Calculate price momentum."""
        if len(self.price_history) < self.lookback_period:
            return 0.0
        
        # Simple rate of change
        current_price = self.price_history[-1]
        past_price = self.price_history[-self.lookback_period]
        
        if past_price == 0:
            return 0.0
        
        return (current_price - past_price) / past_price
    
    def _calculate_rsi(self, current_price: float) -> float:
        """Calculate RSI indicator."""
        if len(self.price_history) < 2:
            return 50.0  # Neutral
        
        # Calculate price change
        prev_price = self.price_history[-2] if len(self.price_history) > 1 else current_price
        change = current_price - prev_price
        
        # Track gains and losses
        gain = max(0, change)
        loss = max(0, -change)
        
        self._gains.append(gain)
        self._losses.append(loss)
        
        # Limit history
        if len(self._gains) > self.rsi_period:
            self._gains.pop(0)
            self._losses.pop(0)
        
        # Need enough data
        if len(self._gains) < self.rsi_period:
            return 50.0
        
        # Calculate average gain/loss
        avg_gain = sum(self._gains) / len(self._gains)
        avg_loss = sum(self._losses) / len(self._losses)
        
        if avg_loss == 0:
            return 100.0  # No losses = RSI 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.price_history.clear()
        self.rsi_values.clear()
        self._gains.clear()
        self._losses.clear()
        self.last_signal_time = None
    
    # Note: Optimization methods are added by OptimizationCapability
    # when the strategy is created through ComponentFactory.
    # This keeps the strategy class clean and focused on its core purpose.


# Example of creating the strategy with capabilities
def create_momentum_strategy(config: Dict[str, Any] = None) -> Any:
    """
    Factory function to create momentum strategy with capabilities.
    
    This would typically use ComponentFactory to add capabilities.
    """
    # Default configuration
    default_config = {
        'lookback_period': 20,
        'momentum_threshold': 0.02,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    }
    
    if config:
        default_config.update(config)
    
    # Create strategy instance
    strategy = MomentumStrategy(**default_config)
    
    # In real usage, this would use ComponentFactory:
    # from core.components import ComponentFactory
    # 
    # strategy = ComponentFactory().create_component({
    #     'class': 'MomentumStrategy',
    #     'params': default_config,
    #     'capabilities': ['strategy', 'lifecycle', 'events', 'optimization']
    # })
    
    return strategy