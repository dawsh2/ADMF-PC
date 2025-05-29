"""
Mean reversion trading strategy.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
import logging

from ...core.events import Event
from ..base import StrategyBase
from ..indicators import BollingerBands, RSI, SimpleMovingAverage
from ..features import PriceFeatureExtractor, VolatilityExtractor
from ..rules import ThresholdRule, CompositeRule
from ..rules import StopLossRule, TakeProfitRule, VolatilityBasedRule


logger = logging.getLogger(__name__)


class MeanReversionStrategy(StrategyBase):
    """
    Mean reversion strategy that trades price deviations from equilibrium.
    
    Features:
    - Bollinger Bands for deviation measurement
    - RSI for oversold/overbought confirmation
    - Z-score based entry signals
    - Volatility-adjusted position sizing
    - Quick exits at mean
    """
    
    def __init__(self,
                 name: str = "mean_reversion_strategy",
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 z_score_threshold: float = 2.0,
                 rsi_period: int = 14,
                 rsi_threshold: float = 30,
                 mean_exit_threshold: float = 0.1,
                 max_holding_period: int = 10):
        super().__init__(name)
        
        # Parameters
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.z_score_threshold = z_score_threshold
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.mean_exit_threshold = mean_exit_threshold
        self.max_holding_period = max_holding_period
        
        # Initialize components
        self._initialize_components()
        
        # Track positions for mean reversion exits
        self.position_entry_data: Dict[str, Dict[str, Any]] = {}
    
    def _initialize_components(self):
        """Initialize strategy components."""
        # Indicators
        self.indicators = {
            'bb': BollingerBands(self.bb_period, self.bb_std),
            'rsi': RSI(self.rsi_period),
            'sma': SimpleMovingAverage(self.bb_period)
        }
        
        # Feature extractors
        self.feature_extractors = [
            PriceFeatureExtractor(name="price"),
            VolatilityExtractor([10, 20], name="volatility")
        ]
        
        # Entry rules
        self.entry_rules = self._create_entry_rules()
        
        # Exit rules
        self.exit_rules = self._create_exit_rules()
        
        # Position sizing
        self.position_sizer = VolatilityBasedRule(
            name="mean_reversion_sizer",
            target_risk=0.001,  # 0.1% risk per trade
            volatility_measure="atr"
        )
    
    def _create_entry_rules(self) -> CompositeRule:
        """Create entry rules for mean reversion."""
        # Buy when price is below lower band and RSI is oversold
        oversold_rule = ThresholdRule(
            name="oversold_entry",
            value_key="z_score",
            threshold=-self.z_score_threshold,
            direction="below",
            signal_type="BUY"
        )
        
        # Sell when price is above upper band and RSI is overbought
        overbought_rule = ThresholdRule(
            name="overbought_entry",
            value_key="z_score",
            threshold=self.z_score_threshold,
            direction="above",
            signal_type="SELL"
        )
        
        return CompositeRule(
            name="mean_reversion_entry",
            rules=[oversold_rule, overbought_rule],
            logic="any"
        )
    
    def _create_exit_rules(self) -> List[Any]:
        """Create exit rules."""
        return [
            StopLossRule(
                name="mean_reversion_stop",
                stop_type="percent",
                stop_distance=0.03  # 3% stop
            ),
            # Mean reversion specific exits handled in _check_mean_exits
        ]
    
    def on_bar(self, event: Event) -> None:
        """Process new bar data."""
        bar_data = event.payload
        price = bar_data.get('close', bar_data.get('price'))
        timestamp = bar_data.get('timestamp', datetime.now())
        
        # Update indicators
        for name, indicator in self.indicators.items():
            value = indicator.calculate(price, timestamp)
            if value is not None:
                self.state[name] = value
        
        # Extract features
        all_features = {}
        for extractor in self.feature_extractors:
            features = extractor.extract(bar_data)
            all_features.update(features)
        
        # Calculate z-score if we have BB data
        if hasattr(self.indicators['bb'], 'middle_band'):
            middle = self.indicators['bb'].middle_band
            std_dev = self.indicators['bb'].std_dev
            
            if middle is not None and std_dev is not None and std_dev > 0:
                z_score = (price - middle) / std_dev
                self.state['z_score'] = z_score
                self.state['distance_from_mean'] = abs(z_score)
        
        # Check for entry signals
        if self._should_check_entry():
            rule_data = {**self.state, **all_features}
            signal = self.entry_rules.evaluate(rule_data)
            
            if signal:
                self._generate_signal(signal, bar_data)
        
        # Check exits
        self._check_exits(bar_data)
        self._check_mean_exits(bar_data)
    
    def _should_check_entry(self) -> bool:
        """Check if we should look for entries."""
        # Need indicators ready
        if not all(ind.ready for ind in self.indicators.values()):
            return False
        
        # Check RSI confirmation
        rsi = self.state.get('rsi', 50)
        z_score = self.state.get('z_score', 0)
        
        # For long: need oversold RSI
        if z_score < -self.z_score_threshold * 0.8:
            return rsi < self.rsi_threshold
        
        # For short: need overbought RSI
        if z_score > self.z_score_threshold * 0.8:
            return rsi > (100 - self.rsi_threshold)
        
        return False
    
    def _generate_signal(self, signal: Dict[str, Any], bar_data: Dict[str, Any]) -> None:
        """Generate trading signal."""
        # Calculate position size
        context = {
            'equity': self.state.get('equity', 100000),
            'close': bar_data.get('close'),
            'atr': self.state.get('bb_std_dev', bar_data.get('close', 1) * 0.02)
        }
        
        size = self.position_sizer.calculate_size(signal, context)
        
        # Adjust size based on z-score magnitude
        z_score = abs(self.state.get('z_score', 0))
        size_multiplier = min(z_score / self.z_score_threshold, 2.0)
        size *= size_multiplier
        
        # Create position ID
        position_id = f"{self.name}_{len(self.position_entry_data)}"
        
        # Store entry data for mean reversion exit
        self.position_entry_data[position_id] = {
            'entry_price': bar_data.get('close'),
            'entry_mean': self.state.get('sma'),
            'entry_z_score': self.state.get('z_score'),
            'entry_time': bar_data.get('timestamp'),
            'bars_held': 0
        }
        
        # Create enhanced signal
        enhanced_signal = {
            **signal,
            'strategy': self.name,
            'position_id': position_id,
            'size': size,
            'z_score': self.state.get('z_score'),
            'confidence': min(abs(self.state.get('z_score', 0)) / 3, 1.0),
            'timestamp': bar_data.get('timestamp')
        }
        
        # Emit signal
        if self._events and self._events.event_bus:
            signal_event = Event('SIGNAL', enhanced_signal)
            self._events.event_bus.publish(signal_event)
        
        logger.info(f"Generated mean reversion {signal['type']} signal at z-score {self.state.get('z_score'):.2f}")
    
    def _check_mean_exits(self, bar_data: Dict[str, Any]) -> None:
        """Check for mean reversion specific exits."""
        positions = self.state.get('positions', [])
        current_price = bar_data.get('close')
        current_mean = self.state.get('sma')
        
        if current_mean is None:
            return
        
        for position in positions:
            position_id = position.get('id')
            
            if position_id not in self.position_entry_data:
                continue
            
            entry_data = self.position_entry_data[position_id]
            entry_data['bars_held'] += 1
            
            # Check if price has reverted to mean
            distance_to_mean = abs(current_price - current_mean) / current_mean
            
            should_exit = False
            exit_reason = None
            
            # Exit when close to mean
            if distance_to_mean < self.mean_exit_threshold:
                should_exit = True
                exit_reason = "mean_reversion_complete"
            
            # Exit if held too long
            elif entry_data['bars_held'] >= self.max_holding_period:
                should_exit = True
                exit_reason = "max_holding_period"
            
            # Exit if z-score flipped significantly
            current_z_score = self.state.get('z_score', 0)
            entry_z_score = entry_data['entry_z_score']
            
            if entry_z_score > 0 and current_z_score < -1:
                should_exit = True
                exit_reason = "z_score_flipped"
            elif entry_z_score < 0 and current_z_score > 1:
                should_exit = True
                exit_reason = "z_score_flipped"
            
            if should_exit:
                # Calculate profit
                entry_price = entry_data['entry_price']
                if position.get('side') == 'long':
                    profit = (current_price - entry_price) / entry_price
                else:
                    profit = (entry_price - current_price) / entry_price
                
                # Create exit signal
                exit_signal = {
                    'type': 'EXIT',
                    'position_id': position_id,
                    'strategy': self.name,
                    'reason': exit_reason,
                    'exit_price': current_price,
                    'profit': profit,
                    'bars_held': entry_data['bars_held']
                }
                
                # Emit exit signal
                if self._events and self._events.event_bus:
                    exit_event = Event('SIGNAL', exit_signal)
                    self._events.event_bus.publish(exit_event)
                
                # Clean up entry data
                del self.position_entry_data[position_id]
                
                logger.info(f"Mean reversion exit: {exit_reason}, profit: {profit:.2%}")
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get optimization parameter space."""
        return {
            'bb_period': {'type': 'int', 'min': 10, 'max': 30},
            'bb_std': {'type': 'float', 'min': 1.5, 'max': 3.0},
            'z_score_threshold': {'type': 'float', 'min': 1.5, 'max': 3.0},
            'rsi_period': {'type': 'int', 'min': 10, 'max': 20},
            'rsi_threshold': {'type': 'float', 'min': 20, 'max': 40},
            'mean_exit_threshold': {'type': 'float', 'min': 0.05, 'max': 0.2},
            'max_holding_period': {'type': 'int', 'min': 5, 'max': 20}
        }
    
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        
        # Clear position tracking
        self.position_entry_data.clear()
        
        # Reset all components
        for indicator in self.indicators.values():
            if hasattr(indicator, 'reset'):
                indicator.reset()
        
        for extractor in self.feature_extractors:
            if hasattr(extractor, 'reset'):
                extractor.reset()