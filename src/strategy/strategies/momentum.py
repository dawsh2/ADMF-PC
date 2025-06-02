"""
Momentum trading strategy implementation.

This demonstrates a pure protocol-based strategy with NO inheritance.
The strategy can be enhanced with capabilities through the ComponentFactory.
"""

from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from decimal import Decimal
import uuid

from ..protocols import Strategy, SignalDirection
from ...risk.protocols import Signal, SignalType, OrderSide


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
                 momentum_threshold: float = 0.0005,  # Lowered from 0.02 to 0.0005 for minute data
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 signal_cooldown: float = 300):
        """
        Initialize momentum strategy.
        
        Args:
            lookback_period: Period for momentum calculation
            momentum_threshold: Minimum momentum for signal
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            signal_cooldown: Signal cooldown in seconds
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
        self.signal_cooldown = signal_cooldown  # Configurable cooldown
        
        # Internal calculation state
        self._gains: List[float] = []
        self._losses: List[float] = []
        
    @property
    def name(self) -> str:
        """Strategy name for identification."""
        return "momentum_strategy"
    
    def get_required_indicators(self) -> Set[str]:
        """
        Return indicators required by this strategy.
        
        This allows the IndicatorContainer to compute shared indicators
        instead of the strategy calculating them internally.
        """
        return {
            f"SMA_{self.lookback_period}",  # For momentum calculation
            "RSI"  # For RSI-based signals
        }
    
    def generate_signals(self, strategy_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading signals from market data and indicators.
        
        This method is called by StrategyContainer with combined
        market data and computed indicators.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🎯 MomentumStrategy.generate_signals() called!")
        
        market_data = strategy_input.get('market_data', {})
        indicators = strategy_input.get('indicators', {})
        timestamp = strategy_input.get('timestamp', datetime.now())
        
        logger.info(f"   Market data: {list(market_data.keys())} symbols")
        logger.info(f"   Indicators: {indicators}")
        logger.info(f"   Timestamp: {timestamp}")
        
        signals = []
        
        # Process each symbol in market data
        for symbol, data in market_data.items():
            logger.info(f"   📊 Processing symbol: {symbol}, data: {data}")
            price = data.get('close', data.get('price'))
            logger.info(f"   💰 Extracted price: {price}")
            if price is None:
                logger.info(f"   ❌ No price found for {symbol}, skipping")
                continue
            
            # Get indicators for this symbol
            symbol_indicators = indicators.get(symbol, {})
            
            # Update price history for momentum calculation
            self.price_history.append(price)
            if len(self.price_history) > self.lookback_period * 2:
                self.price_history.pop(0)
            
            # Need enough data for momentum
            if len(self.price_history) < self.lookback_period:
                # Debug logging for price history
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Not enough price history: {len(self.price_history)}/{self.lookback_period}")
                continue
            
            # Check cooldown
            if self.last_signal_time:
                time_since_last = (timestamp - self.last_signal_time).total_seconds()
                if time_since_last < self.signal_cooldown:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Signal in cooldown: {time_since_last}s < {self.signal_cooldown}s")
                    continue
            
            # Get indicators - use shared indicators if available, fallback to internal calculation
            rsi = symbol_indicators.get('RSI')
            if rsi is None:
                rsi = self._calculate_rsi(price)
            
            # Calculate momentum (still internal for now - could be moved to indicators)
            momentum = self._calculate_momentum()
            
            # Debug logging for signal conditions
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"🎯 SIGNAL ANALYSIS: momentum={momentum:.6f}, rsi={rsi:.2f}, threshold={self.momentum_threshold}")
            logger.info(f"   RSI bounds: oversold={self.rsi_oversold}, overbought={self.rsi_overbought}")
            logger.info(f"   Price history length: {len(self.price_history)}, Current price: {price}")
            logger.info(f"   Momentum conditions: momentum > threshold? {momentum > self.momentum_threshold}, momentum < -threshold? {momentum < -self.momentum_threshold}")
            logger.info(f"   RSI conditions: rsi < overbought? {rsi < self.rsi_overbought}, rsi > oversold? {rsi > self.rsi_oversold}")
            
            # Log the actual signal decision logic
            if momentum > self.momentum_threshold and rsi < self.rsi_overbought:
                logger.info(f"   💡 BULLISH signal triggered!")
            elif momentum < -self.momentum_threshold and rsi > self.rsi_oversold:
                logger.info(f"   💡 BEARISH signal triggered!")
            elif rsi < self.rsi_oversold and momentum > 0:
                logger.info(f"   💡 OVERSOLD REVERSAL signal triggered!")
            elif rsi > self.rsi_overbought and momentum < 0:
                logger.info(f"   💡 OVERBOUGHT REVERSAL signal triggered!")
            else:
                logger.info(f"   ❌ No signal conditions met")
            
            # Generate signal based on momentum and RSI
            signal = None
            
            if momentum > self.momentum_threshold and rsi < self.rsi_overbought:
                # Bullish momentum, not overbought
                signal = Signal(
                    signal_id=str(uuid.uuid4()),
                    strategy_id=self.name,
                    symbol=symbol,
                    signal_type=SignalType.ENTRY,
                    side=OrderSide.BUY,
                    strength=Decimal(str(min(momentum / (self.momentum_threshold * 2), 1.0))),
                    timestamp=timestamp,
                    metadata={
                        'momentum': float(momentum),
                        'rsi': float(rsi),
                        'reason': 'Positive momentum with room to run'
                    }
                )
                
            elif momentum < -self.momentum_threshold and rsi > self.rsi_oversold:
                # Bearish momentum, not oversold
                signal = Signal(
                    signal_id=str(uuid.uuid4()),
                    strategy_id=self.name,
                    symbol=symbol,
                    signal_type=SignalType.ENTRY,
                    side=OrderSide.SELL,
                    strength=Decimal(str(min(abs(momentum) / (self.momentum_threshold * 2), 1.0))),
                    timestamp=timestamp,
                    metadata={
                        'momentum': float(momentum),
                        'rsi': float(rsi),
                        'reason': 'Negative momentum with room to fall'
                    }
                )
                
            elif rsi < self.rsi_oversold and momentum > 0:
                # Oversold with positive momentum - potential reversal
                signal = Signal(
                    signal_id=str(uuid.uuid4()),
                    strategy_id=self.name,
                    symbol=symbol,
                    signal_type=SignalType.ENTRY,
                    side=OrderSide.BUY,
                    strength=Decimal('0.5'),  # Lower confidence for reversal
                    timestamp=timestamp,
                    metadata={
                        'momentum': float(momentum),
                        'rsi': float(rsi),
                        'reason': 'Oversold reversal signal'
                    }
                )
                
            elif rsi > self.rsi_overbought and momentum < 0:
                # Overbought with negative momentum - potential reversal
                signal = Signal(
                    signal_id=str(uuid.uuid4()),
                    strategy_id=self.name,
                    symbol=symbol,
                    signal_type=SignalType.ENTRY,
                    side=OrderSide.SELL,
                    strength=Decimal('0.5'),  # Lower confidence for reversal
                    timestamp=timestamp,
                    metadata={
                        'momentum': float(momentum),
                        'rsi': float(rsi),
                        'reason': 'Overbought reversal signal'
                    }
                )
            
            if signal:
                signals.append(signal)
                self.last_signal_time = timestamp
        
        return signals
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Backward compatibility method for single signal generation.
        
        This delegates to generate_signals for consistency.
        """
        strategy_input = {
            'market_data': {'UNKNOWN': market_data},
            'indicators': {},
            'timestamp': market_data.get('timestamp', datetime.now())
        }
        
        signals = self.generate_signals(strategy_input)
        return signals[0] if signals else None
    
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