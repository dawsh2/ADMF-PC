"""
Market condition classifiers that use indicators to classify market states.

These classifiers subscribe to indicator updates from the IndicatorHub
and classify the market into different states or conditions.
"""

from typing import Dict, Any, Optional, List, Deque
from datetime import datetime, timedelta
from collections import deque
import logging

from ...core.events import Event
from ..protocols import Classifier


logger = logging.getLogger(__name__)


class TrendVolatilityClassifier:
    """
    Classifies market conditions based on trend and volatility.
    
    Classifications:
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
                 min_class_duration: int = 5):
        """
        Initialize classifier.
        
        Args:
            trend_ma_fast: Name of fast MA indicator for trend
            trend_ma_slow: Name of slow MA indicator for trend
            volatility_indicator: Name of volatility indicator (e.g., ATR)
            volatility_threshold: Multiplier for high volatility detection
            min_class_duration: Minimum bars before class can change
        """
        self.trend_ma_fast = trend_ma_fast
        self.trend_ma_slow = trend_ma_slow
        self.volatility_indicator = volatility_indicator
        self.volatility_threshold = volatility_threshold
        self.min_class_duration = min_class_duration
        
        # State
        self._current_class: Optional[str] = None
        self._confidence: float = 0.0
        self._class_start_time: Optional[datetime] = None
        self._class_duration: int = 0
        
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
        """Process indicator update and classify class."""
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
        
        # Classify class
        new_class = self._classify_conditions(fast_ma, slow_ma, volatility)
        
        # Update class if changed and duration met
        if new_class != self._current_class:
            if self._class_duration >= self.min_class_duration:
                self._change_class(new_class, data.get('timestamp'))
            else:
                # Reduce confidence if we want to change but can't yet
                self._confidence *= 0.9
        else:
            # Increase confidence if class is stable
            self._confidence = min(1.0, self._confidence + 0.1)
            self._class_duration += 1
    
    def _classify_conditions(self, fast_ma: float, slow_ma: float, volatility: float) -> str:
        """Classify class based on indicators."""
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
    
    def _change_class(self, new_class: str, timestamp: Optional[datetime]) -> None:
        """Change to new class."""
        old_class = self._current_class
        self._current_class = new_class
        self._class_start_time = timestamp
        self._class_duration = 0
        self._confidence = 0.5  # Start with moderate confidence
        
        logger.info(f"Classification change: {old_class} -> {new_class}")
        
        # Emit classification change event
        if self._events and self._events.event_bus:
            class_data = {
                'classifier': 'TrendVolatilityClassifier',
                'old_class': old_class,
                'new_class': new_class,
                'timestamp': timestamp,
                'confidence': self._confidence
            }
            event = Event('CLASSIFICATION_CHANGE', class_data)
            self._events.event_bus.publish(event)
    
    def classify(self, data: Dict[str, Any]) -> str:
        """Classify current market class."""
        # This method is called directly with indicator data
        fast_ma = data.get(self.trend_ma_fast)
        slow_ma = data.get(self.trend_ma_slow)
        volatility = data.get(self.volatility_indicator)
        
        if fast_ma is None or slow_ma is None or volatility is None:
            return "UNKNOWN"
        
        return self._classify_conditions(fast_ma, slow_ma, volatility)
    
    @property
    def current_class(self) -> Optional[str]:
        """Current classified class."""
        return self._current_class
    
    @property
    def confidence(self) -> float:
        """Confidence in current classification (0-1)."""
        return self._confidence
    
    def reset(self) -> None:
        """Reset classifier state."""
        self._current_class = None
        self._confidence = 0.0
        self._class_start_time = None
        self._class_duration = 0
        self._volatility_history.clear()
        self._avg_volatility = None


class MultiIndicatorClassifier:
    """
    Advanced classifier using multiple indicators for class detection.
    
    Uses a scoring system based on multiple indicators to classify
    market conditions more robustly.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Config should contain:
        - indicators: Dict mapping indicator names to their class rules
        - weights: Dict of indicator weights for scoring
        - thresholds: Dict of score thresholds for each class
        """
        self.config = config
        self.indicators_config = config.get('indicators', {})
        self.weights = config.get('weights', {})
        self.thresholds = config.get('thresholds', {})
        
        # State
        self._current_class: Optional[str] = None
        self._confidence: float = 0.0
        self._class_scores: Dict[str, float] = {}
        
        # Capabilities
        self._events = None
    
    def setup_subscriptions(self) -> None:
        """Subscribe to indicator updates."""
        if self._events:
            self._events.subscribe('INDICATOR_UPDATE', self.on_indicator_update)
    
    def on_indicator_update(self, event: Event) -> None:
        """Process indicator update and classify class."""
        data = event.payload
        indicators = data.get('all_indicators', {})
        
        # Calculate scores for each class
        class_scores = self._calculate_class_scores(indicators)
        self._class_scores = class_scores
        
        # Find class with highest score
        if class_scores:
            best_class = max(class_scores.items(), key=lambda x: x[1])
            new_class = best_class[0]
            score = best_class[1]
            
            # Update confidence based on score strength
            self._confidence = min(1.0, score / 100.0)
            
            # Change class if different
            if new_class != self._current_class:
                self._change_class(new_class, data.get('timestamp'))
    
    def _calculate_class_scores(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """Calculate scores for each class based on indicators."""
        scores = {}
        
        for class_name, class_rules in self.thresholds.items():
            score = 0.0
            
            for indicator_name, rules in self.indicators_config.items():
                if indicator_name not in indicators:
                    continue
                
                value = indicators[indicator_name]
                weight = self.weights.get(indicator_name, 1.0)
                
                # Check if indicator supports this class
                if class_name in rules:
                    rule = rules[class_name]
                    if self._evaluate_rule(value, rule):
                        score += weight
            
            scores[class_name] = score
        
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
    
    def _change_class(self, new_class: str, timestamp: Optional[datetime]) -> None:
        """Change to new class."""
        old_class = self._current_class
        self._current_class = new_class
        
        logger.info(f"Classification change: {old_class} -> {new_class}")
        
        # Emit classification change event
        if self._events and self._events.event_bus:
            class_data = {
                'classifier': 'MultiIndicatorClassifier',
                'old_class': old_class,
                'new_class': new_class,
                'timestamp': timestamp,
                'confidence': self._confidence,
                'scores': self._class_scores
            }
            event = Event('CLASSIFICATION_CHANGE', class_data)
            self._events.event_bus.publish(event)
    
    def classify(self, data: Dict[str, Any]) -> str:
        """Classify current market class."""
        scores = self._calculate_class_scores(data)
        
        if scores:
            best_class = max(scores.items(), key=lambda x: x[1])
            return best_class[0]
        
        return "UNKNOWN"
    
    @property
    def current_class(self) -> Optional[str]:
        """Current classified class."""
        return self._current_class
    
    @property
    def confidence(self) -> float:
        """Confidence in current classification (0-1)."""
        return self._confidence
    
    def reset(self) -> None:
        """Reset classifier state."""
        self._current_class = None
        self._confidence = 0.0
        self._class_scores.clear()


# Add missing classes for compatibility
from enum import Enum
from dataclasses import dataclass


class RegimeState(Enum):
    """Market regime states."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeContext:
    """Context information for a regime."""
    state: RegimeState
    confidence: float
    start_time: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RegimeClassifier:
    """
    Classifier specifically for market regimes.
    
    This is a simplified version that wraps MultiIndicatorClassifier
    for regime detection.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize regime classifier."""
        self.config = config or {}
        
        # Map regime states to classes
        self.regime_mapping = {
            'TRENDING_UP': RegimeState.TRENDING_UP,
            'TRENDING_DOWN': RegimeState.TRENDING_DOWN,
            'RANGING': RegimeState.RANGING,
            'HIGH_VOLATILITY': RegimeState.HIGH_VOLATILITY,
            'LOW_VOLATILITY': RegimeState.LOW_VOLATILITY
        }
        
        # Create underlying classifier
        classifier_config = {
            'indicators': self.config.get('indicators', {
                'trend': {'weight': 0.4},
                'volatility': {'weight': 0.3},
                'momentum': {'weight': 0.3}
            }),
            'thresholds': self.config.get('thresholds', {
                'TRENDING_UP': {'trend': {'min': 0.5}},
                'TRENDING_DOWN': {'trend': {'max': -0.5}},
                'RANGING': {'trend': {'min': -0.2, 'max': 0.2}},
                'HIGH_VOLATILITY': {'volatility': {'min': 0.7}},
                'LOW_VOLATILITY': {'volatility': {'max': 0.3}}
            })
        }
        
        self.classifier = MultiIndicatorClassifier(classifier_config)
        self._current_regime = RegimeState.UNKNOWN
        self._regime_history = deque(maxlen=100)
    
    def classify(self, data: Dict[str, Any]) -> RegimeState:
        """Classify current market regime."""
        # Use underlying classifier
        market_class = self.classifier.classify(data)
        
        # Map to regime state
        if market_class and market_class in self.regime_mapping:
            regime = self.regime_mapping[market_class]
        else:
            regime = RegimeState.UNKNOWN
            
        self._current_regime = regime
        self._regime_history.append({
            'timestamp': data.get('timestamp', datetime.now()),
            'regime': regime,
            'confidence': self.classifier.confidence
        })
        
        return regime
    
    def get_current_regime(self) -> RegimeState:
        """Get current regime state."""
        return self._current_regime
    
    def get_regime_context(self) -> RegimeContext:
        """Get full regime context."""
        # Find when current regime started
        start_time = datetime.now()
        for i in range(len(self._regime_history) - 1, -1, -1):
            if self._regime_history[i]['regime'] != self._current_regime:
                if i < len(self._regime_history) - 1:
                    start_time = self._regime_history[i + 1]['timestamp']
                break
        
        return RegimeContext(
            state=self._current_regime,
            confidence=self.classifier.confidence,
            start_time=start_time,
            metadata={
                'class_scores': self.classifier._class_scores,
                'history_length': len(self._regime_history)
            }
        )
    
    @property
    def confidence(self) -> float:
        """Current classification confidence."""
        return self.classifier.confidence
    
    def reset(self) -> None:
        """Reset classifier state."""
        self.classifier.reset()
        self._current_regime = RegimeState.UNKNOWN
        self._regime_history.clear()