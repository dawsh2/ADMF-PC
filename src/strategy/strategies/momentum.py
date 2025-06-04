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
from ...core.logging.structured import StructuredLogger, LogContext


class MomentumStrategy:
    """
    Stateless momentum-based trading strategy.
    
    This strategy consumes features from FeatureHub and makes pure
    decisions based on current feature values. No state is maintained.
    
    Features:
    - Pure decision logic based on SMA momentum
    - RSI-based signal filtering
    - Completely stateless - no internal state storage
    - Protocol + Composition compliant
    """
    
    def __init__(self, 
                 momentum_threshold: float = 0.02,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 component_id: str = None):
        """
        Initialize momentum strategy.
        
        Args:
            momentum_threshold: Minimum momentum for signal generation
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
        """
        # Configuration only - no state!
        self.momentum_threshold = momentum_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # Logging setup
        self.component_id = component_id or f"momentum_strategy_{uuid.uuid4().hex[:8]}"
        context = LogContext(
            container_id="strategy_container",
            component_id=self.component_id,
            correlation_id=None
        )
        self.logger = StructuredLogger(__name__, context)
        
    @property
    def name(self) -> str:
        """Strategy name for identification."""
        return "momentum_strategy"
    
    def get_required_features(self) -> Set[str]:
        """
        Return features required by this strategy.
        
        This allows the FeatureHub to compute shared features
        instead of duplicating calculations.
        """
        return {
            "sma_fast",   # Fast SMA for momentum
            "sma_slow",   # Slow SMA for momentum  
            "rsi"         # RSI for signal filtering
        }
    
    def generate_signals(self, strategy_input: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals from market data and features.
        
        This is a STATELESS function that makes decisions based purely
        on current feature values from FeatureHub.
        """
        self.logger.info("Stateless MomentumStrategy.generate_signals() called")
        
        market_data = strategy_input.get('market_data', {})
        features = strategy_input.get('features', {})
        timestamp = strategy_input.get('timestamp', datetime.now())
        
        signals = []
        
        # Process each symbol - pure stateless decision logic
        for symbol, data in market_data.items():
            price = data.get('close', data.get('price'))
            if price is None:
                continue
            
            # Get features for this symbol from FeatureHub
            symbol_features = features.get(symbol, {})
            
            # Extract required features
            sma_fast = symbol_features.get('sma_fast')
            sma_slow = symbol_features.get('sma_slow')
            rsi = symbol_features.get('rsi')
            
            # Skip if required features not available
            if any(x is None for x in [sma_fast, sma_slow, rsi]):
                self.logger.debug("Missing features for %s", symbol)
                continue
            
            # Pure stateless momentum calculation
            momentum = (sma_fast - sma_slow) / sma_slow if sma_slow != 0 else 0.0
            
            self.logger.debug(
                "Signal analysis for %s: momentum=%.6f, rsi=%.2f", 
                symbol, momentum, rsi
            )
            
            # Stateless signal generation logic
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
                        'sma_fast': float(sma_fast),
                        'sma_slow': float(sma_slow),
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
                        'sma_fast': float(sma_fast),
                        'sma_slow': float(sma_slow),
                        'reason': 'Negative momentum with room to fall'
                    }
                )
            
            if signal:
                signals.append(signal)
                self.logger.info("Generated %s signal for %s", signal.side.name, symbol)
        
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
    
    def reset(self) -> None:
        """
        Reset strategy state.
        
        Since this strategy is stateless, reset does nothing.
        All state is managed by FeatureHub.
        """
        pass  # No state to reset!
    
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