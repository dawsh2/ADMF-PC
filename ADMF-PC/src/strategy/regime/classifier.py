"""
Market regime classifiers that use indicators to classify market conditions.

These classifiers subscribe to indicator updates from the IndicatorHub
and classify the market into different regimes.
"""

from typing import Dict, Any, Optional, List, Deque
from datetime import datetime, timedelta
from collections import deque
import logging

from ...core.events import Event
from ..protocols import RegimeClassifier


logger = logging.getLogger(__name__)


class TrendVolatilityClassifier:
    """
    Classifies market into regimes based on trend and volatility.
    
    Regimes:
    - TRENDING_UP: Strong upward trend with normal volatility
    - TRENDING_DOWN: Strong downward trend with normal volatility
    - RANGE_BOUND: No clear trend, normal volatility
    - HIGH_VOLATILITY: Any trend with high volatility
    """
    
    def __init__(self, 
                 trend_ma_fast: str = "MA_20",
                 trend_ma_slow: str = "MA_50", 
                 volatility_indicator: str = "ATR_14",
                 volatility_threshold: float = 1.5,
                 min_regime_duration: int = 5):
        """
        Initialize classifier.
        
        Args:
            trend_ma_fast: Name of fast MA indicator for trend
            trend_ma_slow: Name of slow MA indicator for trend
            volatility_indicator: Name of volatility indicator (e.g., ATR)
            volatility_threshold: Multiplier for high volatility detection
            min_regime_duration: Minimum bars before regime can change
        """
        self.trend_ma_fast = trend_ma_fast
        self.trend_ma_slow = trend_ma_slow
        self.volatility_indicator = volatility_indicator
        self.volatility_threshold = volatility_threshold
        self.min_regime_duration = min_regime_duration
        
        # State
        self._current_regime: Optional[str] = None
        self._confidence: float = 0.0
        self._regime_start_time: Optional[datetime] = None
        self._regime_duration: int = 0
        
        # History for stability
        self._volatility_history: Deque[float] = deque(maxlen=20)
        self._avg_volatility: Optional[float] = None
        
        # Capabilities
        self._events = None
    
    def setup_subscriptions(self) -> None:
        """Subscribe to indicator updates."""
        if self._events:
            self._events.subscribe('INDICATOR_UPDATE', self.on_indicator_update)
    
    def on_indicator_update(self, event: Event) -> None:
        """Process indicator update and classify regime."""
        data = event.payload
        indicators = data.get('indicators', {})
        
        # Extract required indicators
        fast_ma = indicators.get(self.trend_ma_fast)
        slow_ma = indicators.get(self.trend_ma_slow)
        volatility = indicators.get(self.volatility_indicator)
        
        # Need all indicators to classify
        if fast_ma is None or slow_ma is None or volatility is None:
            return
        
        # Update volatility history
        self._volatility_history.append(volatility)
        if len(self._volatility_history) >= 10:
            self._avg_volatility = sum(self._volatility_history) / len(self._volatility_history)
        
        # Classify regime
        new_regime = self._classify_regime(fast_ma, slow_ma, volatility)
        
        # Update regime if changed and duration met
        if new_regime != self._current_regime:
            if self._regime_duration >= self.min_regime_duration:
                self._change_regime(new_regime, data.get('timestamp'))
            else:
                # Reduce confidence if we want to change but can't yet
                self._confidence *= 0.9
        else:
            # Increase confidence if regime is stable
            self._confidence = min(1.0, self._confidence + 0.1)
            self._regime_duration += 1
    
    def _classify_regime(self, fast_ma: float, slow_ma: float, volatility: float) -> str:
        """Classify regime based on indicators."""
        # Determine if high volatility
        is_high_vol = False
        if self._avg_volatility is not None:
            is_high_vol = volatility > self._avg_volatility * self.volatility_threshold
        
        # If high volatility, that overrides trend
        if is_high_vol:
            return "HIGH_VOLATILITY"
        
        # Determine trend
        trend_strength = (fast_ma - slow_ma) / slow_ma
        
        if trend_strength > 0.02:  # 2% above
            return "TRENDING_UP"
        elif trend_strength < -0.02:  # 2% below
            return "TRENDING_DOWN"
        else:
            return "RANGE_BOUND"
    
    def _change_regime(self, new_regime: str, timestamp: Optional[datetime]) -> None:
        """Change to new regime."""
        old_regime = self._current_regime
        self._current_regime = new_regime
        self._regime_start_time = timestamp
        self._regime_duration = 0
        self._confidence = 0.5  # Start with moderate confidence
        
        logger.info(f"Regime change: {old_regime} -> {new_regime}")
        
        # Emit regime change event
        if self._events and self._events.event_bus:
            regime_data = {
                'classifier': 'TrendVolatilityClassifier',
                'old_regime': old_regime,
                'new_regime': new_regime,
                'timestamp': timestamp,
                'confidence': self._confidence
            }
            event = Event('REGIME_CHANGE', regime_data)
            self._events.event_bus.publish(event)
    
    def classify(self, data: Dict[str, Any]) -> str:
        """Classify current market regime."""
        # This method is called directly with indicator data
        fast_ma = data.get(self.trend_ma_fast)
        slow_ma = data.get(self.trend_ma_slow)
        volatility = data.get(self.volatility_indicator)
        
        if fast_ma is None or slow_ma is None or volatility is None:
            return "UNKNOWN"
        
        return self._classify_regime(fast_ma, slow_ma, volatility)
    
    @property
    def current_regime(self) -> Optional[str]:
        """Current classified regime."""
        return self._current_regime
    
    @property
    def confidence(self) -> float:
        """Confidence in current classification (0-1)."""
        return self._confidence
    
    def reset(self) -> None:
        """Reset classifier state."""
        self._current_regime = None
        self._confidence = 0.0
        self._regime_start_time = None
        self._regime_duration = 0
        self._volatility_history.clear()
        self._avg_volatility = None


class MultiIndicatorClassifier:
    """
    Advanced classifier using multiple indicators for regime detection.
    
    Uses a scoring system based on multiple indicators to classify
    market conditions more robustly.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Config should contain:
        - indicators: Dict mapping indicator names to their regime rules
        - weights: Dict of indicator weights for scoring
        - thresholds: Dict of score thresholds for each regime
        """
        self.config = config
        self.indicators_config = config.get('indicators', {})
        self.weights = config.get('weights', {})
        self.thresholds = config.get('thresholds', {})
        
        # State
        self._current_regime: Optional[str] = None
        self._confidence: float = 0.0
        self._regime_scores: Dict[str, float] = {}
        
        # Capabilities
        self._events = None
    
    def setup_subscriptions(self) -> None:
        """Subscribe to indicator updates."""
        if self._events:
            self._events.subscribe('INDICATOR_UPDATE', self.on_indicator_update)
    
    def on_indicator_update(self, event: Event) -> None:
        """Process indicator update and classify regime."""
        data = event.payload
        indicators = data.get('all_indicators', {})
        
        # Calculate scores for each regime
        regime_scores = self._calculate_regime_scores(indicators)
        self._regime_scores = regime_scores
        
        # Find regime with highest score
        if regime_scores:
            best_regime = max(regime_scores.items(), key=lambda x: x[1])
            new_regime = best_regime[0]
            score = best_regime[1]
            
            # Update confidence based on score strength
            self._confidence = min(1.0, score / 100.0)
            
            # Change regime if different
            if new_regime != self._current_regime:
                self._change_regime(new_regime, data.get('timestamp'))
    
    def _calculate_regime_scores(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """Calculate scores for each regime based on indicators."""
        scores = {}
        
        for regime, regime_rules in self.thresholds.items():
            score = 0.0
            
            for indicator_name, rules in self.indicators_config.items():
                if indicator_name not in indicators:
                    continue
                
                value = indicators[indicator_name]
                weight = self.weights.get(indicator_name, 1.0)
                
                # Check if indicator supports this regime
                if regime in rules:
                    rule = rules[regime]
                    if self._evaluate_rule(value, rule):
                        score += weight
            
            scores[regime] = score
        
        return scores
    
    def _evaluate_rule(self, value: float, rule: Dict[str, Any]) -> bool:
        """Evaluate if a value satisfies a rule."""
        if 'min' in rule and value < rule['min']:
            return False
        if 'max' in rule and value > rule['max']:
            return False
        if 'above' in rule and value <= rule['above']:
            return False
        if 'below' in rule and value >= rule['below']:
            return False
        
        return True
    
    def _change_regime(self, new_regime: str, timestamp: Optional[datetime]) -> None:
        """Change to new regime."""
        old_regime = self._current_regime
        self._current_regime = new_regime
        
        logger.info(f"Regime change: {old_regime} -> {new_regime}")
        
        # Emit regime change event
        if self._events and self._events.event_bus:
            regime_data = {
                'classifier': 'MultiIndicatorClassifier',
                'old_regime': old_regime,
                'new_regime': new_regime,
                'timestamp': timestamp,
                'confidence': self._confidence,
                'scores': self._regime_scores
            }
            event = Event('REGIME_CHANGE', regime_data)
            self._events.event_bus.publish(event)
    
    def classify(self, data: Dict[str, Any]) -> str:
        """Classify current market regime."""
        scores = self._calculate_regime_scores(data)
        
        if scores:
            best_regime = max(scores.items(), key=lambda x: x[1])
            return best_regime[0]
        
        return "UNKNOWN"
    
    @property
    def current_regime(self) -> Optional[str]:
        """Current classified regime."""
        return self._current_regime
    
    @property
    def confidence(self) -> float:
        """Confidence in current classification (0-1)."""
        return self._confidence
    
    def reset(self) -> None:
        """Reset classifier state."""
        self._current_regime = None
        self._confidence = 0.0
        self._regime_scores.clear()