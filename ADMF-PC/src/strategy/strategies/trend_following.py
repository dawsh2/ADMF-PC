"""
Trend following trading strategy.

This demonstrates a pure protocol-based strategy with NO inheritance.
The strategy can be enhanced with capabilities through the ComponentFactory.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..protocols import Strategy, SignalDirection
from ..indicators import (
    SimpleMovingAverage,
    ExponentialMovingAverage,
    ATR,
    ADX
)
from ..features import TechnicalFeatureExtractor, PricePatternExtractor
from ..rules import CrossoverRule, ThresholdRule, CompositeRule
from ..rules import TrailingStopRule, VolatilityBasedRule


logger = logging.getLogger(__name__)


class TrendFollowingStrategy:
    """
    Trend following strategy that identifies and rides trends.
    
    Features:
    - Multiple timeframe trend confirmation
    - ADX for trend strength measurement
    - Dynamic position sizing based on trend strength
    - Trailing stops to protect profits
    - Pyramid positions in strong trends
    """
    
    def __init__(self,
                 fast_ma_period: int = 20,
                 slow_ma_period: int = 50,
                 trend_ma_period: int = 200,
                 adx_period: int = 14,
                 adx_threshold: float = 25,
                 atr_period: int = 14,
                 pyramid_enabled: bool = True,
                 max_pyramids: int = 3):
        # Parameters
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.trend_ma_period = trend_ma_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period
        self.pyramid_enabled = pyramid_enabled
        self.max_pyramids = max_pyramids
        
        # Initialize components
        self._initialize_components()
        
        # Track pyramid positions
        self.pyramid_count: Dict[str, int] = {}  # symbol -> count
        self.last_pyramid_price: Dict[str, float] = {}  # symbol -> price
        
        # State tracking
        self.state: Dict[str, Any] = {}
        self.positions: List[Dict[str, Any]] = []
    
    def _initialize_components(self):
        """Initialize strategy components."""
        # Indicators
        self.indicators = {
            'fast_ma': ExponentialMovingAverage(self.fast_ma_period),
            'slow_ma': ExponentialMovingAverage(self.slow_ma_period),
            'trend_ma': SimpleMovingAverage(self.trend_ma_period),
            'atr': ATR(self.atr_period),
            'adx': ADX(self.adx_period)
        }
        
        # Feature extractors
        self.feature_extractors = [
            TechnicalFeatureExtractor(name="technical"),
            PricePatternExtractor(lookback=10, name="patterns")
        ]
        
        # Entry rules
        self.entry_rules = self._create_entry_rules()
        
        # Exit rules
        self.exit_rules = self._create_exit_rules()
        
        # Position sizing
        self.position_sizer = VolatilityBasedRule(
            name="trend_sizer",
            target_risk=0.002,  # 0.2% risk
            volatility_measure="atr"
        )
    
    def _create_entry_rules(self) -> CompositeRule:
        """Create trend entry rules."""
        # MA crossover
        ma_cross = CrossoverRule(
            name="ma_crossover",
            fast_key="fast_ma",
            slow_key="slow_ma",
            signal_on="both",
            min_separation=0.001  # 0.1% minimum separation
        )
        
        # Trend filter
        trend_filter = ThresholdRule(
            name="trend_filter",
            value_key="trend_aligned",
            threshold=0.5,
            direction="above",
            signal_type="PASS"  # Just a filter
        )
        
        # ADX strength
        adx_filter = ThresholdRule(
            name="adx_filter",
            value_key="adx",
            threshold=self.adx_threshold,
            direction="above",
            signal_type="PASS"
        )
        
        return CompositeRule(
            name="trend_entry",
            rules=[ma_cross, trend_filter, adx_filter],
            logic="all"
        )
    
    def _create_exit_rules(self) -> List[Any]:
        """Create trend exit rules."""
        return [
            TrailingStopRule(
                name="trend_trail",
                trail_type="atr",
                trail_distance=3.0,  # 3x ATR
                activation_profit=0.02  # Activate after 2% profit
            )
        ]
    
    @property
    def name(self) -> str:
        """Strategy name for identification."""
        return "trend_following_strategy"
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signal from market data."""
        bar_data = market_data
        price = bar_data.get('close', bar_data.get('price'))
        timestamp = bar_data.get('timestamp', datetime.now())
        symbol = bar_data.get('symbol', 'default')
        
        # Update indicators
        for name, indicator in self.indicators.items():
            if name == 'atr':
                # ATR needs full bar data
                value = indicator.calculate(bar_data, timestamp)
            else:
                value = indicator.calculate(price, timestamp)
            
            if value is not None:
                self.state[name] = value
        
        # Extract features
        all_features = {}
        for extractor in self.feature_extractors:
            features = extractor.extract(bar_data)
            all_features.update(features)
        
        # Calculate trend alignment
        self._calculate_trend_alignment(price)
        
        # Check for entry signals
        if self._should_check_entry():
            rule_data = {**self.state, **all_features}
            signal = self.entry_rules.evaluate(rule_data)
            
            if signal:
                return self._generate_signal(signal, bar_data)
        
        # Check for pyramid opportunities
        if self.pyramid_enabled:
            pyramid_signal = self._check_pyramid_opportunity(symbol, price)
            if pyramid_signal:
                return pyramid_signal
        
        return None
    
    def _calculate_trend_alignment(self, price: float) -> None:
        """Calculate trend alignment score."""
        fast_ma = self.state.get('fast_ma')
        slow_ma = self.state.get('slow_ma')
        trend_ma = self.state.get('trend_ma')
        
        if all(v is not None for v in [fast_ma, slow_ma, trend_ma]):
            # Bullish alignment: price > fast > slow > trend
            bullish_score = 0
            if price > fast_ma:
                bullish_score += 0.25
            if fast_ma > slow_ma:
                bullish_score += 0.25
            if slow_ma > trend_ma:
                bullish_score += 0.25
            if price > trend_ma:
                bullish_score += 0.25
            
            # Bearish alignment: price < fast < slow < trend
            bearish_score = 0
            if price < fast_ma:
                bearish_score += 0.25
            if fast_ma < slow_ma:
                bearish_score += 0.25
            if slow_ma < trend_ma:
                bearish_score += 0.25
            if price < trend_ma:
                bearish_score += 0.25
            
            self.state['trend_aligned'] = max(bullish_score, bearish_score)
            self.state['trend_direction'] = 'bullish' if bullish_score > bearish_score else 'bearish'
        else:
            self.state['trend_aligned'] = 0
            self.state['trend_direction'] = 'neutral'
    
    def _should_check_entry(self) -> bool:
        """Check if we should look for entries."""
        # All indicators must be ready
        return all(ind.ready for ind in self.indicators.values())
    
    def _check_pyramid_opportunity(self, symbol: str, price: float) -> Optional[Dict[str, Any]]:
        """Check if we should add to winning position."""
        positions = self.state.get('positions', [])
        symbol_positions = [p for p in positions if p.get('symbol') == symbol]
        
        if not symbol_positions:
            return None
        
        # Check pyramid count
        current_pyramids = self.pyramid_count.get(symbol, 0)
        if current_pyramids >= self.max_pyramids:
            return None
        
        # Get position details
        position = symbol_positions[0]  # Assume one position per symbol
        entry_price = position.get('entry_price', 0)
        position_side = position.get('side', 'long')
        
        # Check if price has moved favorably
        atr = self.state.get('atr', price * 0.02)
        min_distance = atr * 2  # Require 2 ATR move
        
        last_pyramid = self.last_pyramid_price.get(symbol, entry_price)
        
        should_pyramid = False
        
        if position_side == 'long':
            if price > last_pyramid + min_distance:
                should_pyramid = True
        else:  # short
            if price < last_pyramid - min_distance:
                should_pyramid = True
        
        if should_pyramid:
            # Generate pyramid signal
            pyramid_signal = {
                'type': 'ADD',
                'strategy': self.name,
                'symbol': symbol,
                'reason': 'pyramid',
                'pyramid_level': current_pyramids + 1,
                'confidence': 0.7,  # Lower confidence for pyramids
                'timestamp': datetime.now()
            }
            
            # Calculate pyramid size (smaller than initial)
            context = {
                'equity': self.state.get('equity', 100000),
                'close': price,
                'atr': atr
            }
            
            base_size = self.position_sizer.calculate_size(pyramid_signal, context)
            pyramid_size = base_size * (0.5 ** current_pyramids)  # Halve each level
            
            pyramid_signal['size'] = pyramid_size
            
            # Update tracking
            self.pyramid_count[symbol] = current_pyramids + 1
            self.last_pyramid_price[symbol] = price
            
            logger.info(f"Generated pyramid signal for {symbol} at level {current_pyramids + 1}")
            
            return pyramid_signal
        
        return None
    
    def _generate_signal(self, signal: Dict[str, Any], bar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal."""
        symbol = bar_data.get('symbol', 'default')
        
        # Calculate position size
        context = {
            'equity': self.state.get('equity', 100000),
            'close': bar_data.get('close'),
            'atr': self.state.get('atr')
        }
        
        size = self.position_sizer.calculate_size(signal, context)
        
        # Adjust size based on trend strength (ADX)
        adx = self.state.get('adx', self.adx_threshold)
        strength_multiplier = min(adx / self.adx_threshold, 2.0)
        size *= strength_multiplier
        
        # Reset pyramid tracking for new position
        self.pyramid_count[symbol] = 0
        self.last_pyramid_price[symbol] = bar_data.get('close')
        
        # Create enhanced signal
        enhanced_signal = {
            **signal,
            'strategy': self.name,
            'size': size,
            'trend_strength': adx,
            'trend_alignment': self.state.get('trend_aligned'),
            'trend_direction': self.state.get('trend_direction'),
            'timestamp': bar_data.get('timestamp')
        }
        
        logger.info(f"Generated trend signal: {signal['type']} with ADX {adx:.1f}")
        
        return enhanced_signal
    
    # Note: Optimization methods are added by OptimizationCapability
    # when the strategy is created through ComponentFactory.
    # This keeps the strategy class clean and focused on its core purpose.
    
    def reset(self) -> None:
        """Reset strategy state."""
        # Clear state
        self.state.clear()
        self.positions.clear()
        
        # Clear pyramid tracking
        self.pyramid_count.clear()
        self.last_pyramid_price.clear()
        
        # Reset all components
        for indicator in self.indicators.values():
            if hasattr(indicator, 'reset'):
                indicator.reset()


# Example of creating the strategy with capabilities
def create_trend_following_strategy(config: Dict[str, Any] = None) -> Any:
    """
    Factory function to create trend following strategy with capabilities.
    
    This would typically use ComponentFactory to add capabilities.
    """
    # Default configuration
    default_config = {
        'fast_ma_period': 20,
        'slow_ma_period': 50,
        'trend_ma_period': 200,
        'adx_period': 14,
        'adx_threshold': 25,
        'atr_period': 14,
        'pyramid_enabled': True,
        'max_pyramids': 3
    }
    
    if config:
        default_config.update(config)
    
    # Create strategy instance
    strategy = TrendFollowingStrategy(**default_config)
    
    # In real usage, this would use ComponentFactory:
    # from core.components import ComponentFactory
    # 
    # strategy = ComponentFactory().create_component({
    #     'class': 'TrendFollowingStrategy',
    #     'params': default_config,
    #     'capabilities': ['strategy', 'lifecycle', 'events', 'optimization']
    # })
    
    return strategy