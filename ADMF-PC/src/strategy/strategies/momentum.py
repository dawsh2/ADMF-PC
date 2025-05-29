"""
Momentum-based trading strategy.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ...core.events import Event, EventType
from ..base import StrategyBase
from ..indicators import RSI, MACD, Momentum
from ..features import PriceReturnExtractor, TechnicalFeatureExtractor
from ..rules import CrossoverRule, ThresholdRule, CompositeRule
from ..rules import StopLossRule, TakeProfitRule, PercentEquityRule


logger = logging.getLogger(__name__)


class MomentumStrategy(StrategyBase):
    """
    Momentum strategy that trades based on price momentum and trend strength.
    
    Features:
    - Multiple momentum indicators (RSI, MACD, Rate of Change)
    - Trend confirmation
    - Dynamic position sizing based on signal strength
    - Regime-aware adjustments
    """
    
    def __init__(self, 
                 name: str = "momentum_strategy",
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 lookback_period: int = 20,
                 min_momentum_score: float = 0.6):
        super().__init__(name)
        
        # Parameters
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.lookback_period = lookback_period
        self.min_momentum_score = min_momentum_score
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize strategy components."""
        # Indicators
        self.indicators = {
            'rsi': RSI(self.rsi_period),
            'macd': MACD(self.macd_fast, self.macd_slow, self.macd_signal),
            'momentum': Momentum(self.lookback_period)
        }
        
        # Feature extractors
        self.feature_extractors = [
            PriceReturnExtractor([5, 10, 20], name="returns"),
            TechnicalFeatureExtractor(name="technical")
        ]
        
        # Signal rules
        self.entry_rules = self._create_entry_rules()
        self.exit_rules = self._create_exit_rules()
        
        # Position sizing
        self.position_sizer = PercentEquityRule(
            name="momentum_sizer",
            percent=0.02,  # 2% base size
            max_percent=0.05  # 5% max
        )
    
    def _create_entry_rules(self) -> CompositeRule:
        """Create entry signal rules."""
        # RSI rules
        rsi_buy = ThresholdRule(
            name="rsi_oversold",
            value_key="rsi",
            threshold=self.rsi_oversold,
            direction="below",
            signal_type="BUY"
        )
        
        rsi_sell = ThresholdRule(
            name="rsi_overbought",
            value_key="rsi",
            threshold=self.rsi_overbought,
            direction="above",
            signal_type="SELL"
        )
        
        # MACD crossover
        macd_cross = CrossoverRule(
            name="macd_signal_cross",
            fast_key="macd_line",
            slow_key="signal_line",
            signal_on="both"
        )
        
        # Combine rules
        return CompositeRule(
            name="momentum_entry",
            rules=[rsi_buy, rsi_sell, macd_cross],
            logic="any",
            min_rules=1
        )
    
    def _create_exit_rules(self) -> List[Any]:
        """Create exit rules."""
        return [
            StopLossRule(
                name="momentum_stop",
                stop_type="atr",
                stop_distance=2.0  # 2x ATR
            ),
            TakeProfitRule(
                name="momentum_target",
                target_type="risk_reward",
                risk_reward_ratio=3.0
            )
        ]
    
    def on_bar(self, event: Event) -> None:
        """Process new bar data."""
        bar_data = event.payload
        
        # Update indicators
        price = bar_data.get('close', bar_data.get('price'))
        timestamp = bar_data.get('timestamp', datetime.now())
        
        for name, indicator in self.indicators.items():
            value = indicator.calculate(price, timestamp)
            if value is not None:
                self.state[name] = value
        
        # Extract features
        all_features = {}
        for extractor in self.feature_extractors:
            features = extractor.extract(bar_data)
            all_features.update(features)
        
        # Calculate momentum score
        momentum_score = self._calculate_momentum_score(all_features)
        self.state['momentum_score'] = momentum_score
        
        # Check entry signals if we have momentum
        if momentum_score >= self.min_momentum_score:
            # Prepare data for rules
            rule_data = {
                **self.state,
                **all_features,
                'macd_line': getattr(self.indicators['macd'], 'macd_line', None),
                'signal_line': getattr(self.indicators['macd'], 'signal_line', None)
            }
            
            # Evaluate entry rules
            signal = self.entry_rules.evaluate(rule_data)
            
            if signal:
                self._generate_signal(signal, bar_data)
        
        # Check exits for existing positions
        self._check_exits(bar_data)
    
    def _calculate_momentum_score(self, features: Dict[str, float]) -> float:
        """Calculate overall momentum score."""
        scores = []
        
        # Price momentum
        for period in [5, 10, 20]:
            ret_key = f'returns_return_{period}'
            if ret_key in features:
                ret = features[ret_key]
                scores.append(1.0 if ret > 0 else 0.0)
        
        # RSI momentum
        if 'rsi' in self.state:
            rsi = self.state['rsi']
            if rsi > 50:
                scores.append((rsi - 50) / 50)
            else:
                scores.append((50 - rsi) / 50)
        
        # MACD momentum
        if hasattr(self.indicators['macd'], 'histogram'):
            hist = self.indicators['macd'].histogram
            if hist is not None:
                scores.append(1.0 if hist > 0 else 0.0)
        
        # Technical momentum features
        if 'technical_combined_momentum' in features:
            scores.append((features['technical_combined_momentum'] + 1) / 2)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _generate_signal(self, signal: Dict[str, Any], bar_data: Dict[str, Any]) -> None:
        """Generate trading signal."""
        # Calculate position size
        context = {
            'equity': self.state.get('equity', 100000),
            'close': bar_data.get('close'),
            'atr': self.state.get('atr', bar_data.get('close', 1) * 0.02)
        }
        
        size = self.position_sizer.calculate_size(signal, context)
        
        # Adjust size based on momentum score
        momentum_score = self.state.get('momentum_score', 0.5)
        size *= momentum_score
        
        # Create enhanced signal
        enhanced_signal = {
            **signal,
            'strategy': self.name,
            'size': size,
            'momentum_score': momentum_score,
            'timestamp': bar_data.get('timestamp')
        }
        
        # Emit signal event
        if self._events and self._events.event_bus:
            signal_event = Event('SIGNAL', enhanced_signal)
            self._events.event_bus.publish(signal_event)
        
        logger.info(f"Generated {signal['type']} signal with size {size:.2f}")
    
    def _check_exits(self, bar_data: Dict[str, Any]) -> None:
        """Check exit conditions for open positions."""
        positions = self.state.get('positions', [])
        
        for position in positions:
            context = {
                **bar_data,
                'atr': self.state.get('atr')
            }
            
            for exit_rule in self.exit_rules:
                exit_info = exit_rule.should_exit(position, context)
                
                if exit_info:
                    # Create exit signal
                    exit_signal = {
                        'type': 'EXIT',
                        'position_id': position.get('id'),
                        'strategy': self.name,
                        **exit_info
                    }
                    
                    # Emit exit signal
                    if self._events and self._events.event_bus:
                        exit_event = Event('SIGNAL', exit_signal)
                        self._events.event_bus.publish(exit_event)
                    
                    logger.info(f"Generated exit signal: {exit_info['reason']}")
                    break
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get optimization parameter space."""
        return {
            'rsi_period': {'type': 'int', 'min': 10, 'max': 20},
            'rsi_overbought': {'type': 'float', 'min': 65, 'max': 80},
            'rsi_oversold': {'type': 'float', 'min': 20, 'max': 35},
            'macd_fast': {'type': 'int', 'min': 8, 'max': 15},
            'macd_slow': {'type': 'int', 'min': 20, 'max': 30},
            'macd_signal': {'type': 'int', 'min': 7, 'max': 12},
            'lookback_period': {'type': 'int', 'min': 10, 'max': 30},
            'min_momentum_score': {'type': 'float', 'min': 0.5, 'max': 0.8}
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters."""
        # Update parameters
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Reinitialize components with new parameters
        self._initialize_components()
        
        logger.info(f"Updated parameters: {params}")
    
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        
        # Reset indicators
        for indicator in self.indicators.values():
            if hasattr(indicator, 'reset'):
                indicator.reset()
        
        # Reset feature extractors
        for extractor in self.feature_extractors:
            if hasattr(extractor, 'reset'):
                extractor.reset()
        
        # Reset rules
        self.entry_rules.reset()
        for rule in self.exit_rules:
            rule.reset()