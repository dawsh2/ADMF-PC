"""
Market classifiers for categorizing market conditions.

Classifiers analyze market data and assign categorical labels.
"Regime" classification is just one type - classifiers can identify
any categorical market state.
"""

from typing import Dict, Any, Optional, List, Deque
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

from ..protocols import Classifier


class MarketRegime(Enum):
    """Common market regime classifications."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


class TrendClassifier:
    """
    Classifies market trend direction.
    
    A simple classifier that identifies trending vs ranging markets
    based on moving average relationships.
    """
    
    def __init__(self, 
                 fast_period: int = 20,
                 slow_period: int = 50,
                 trend_threshold: float = 0.02):
        """
        Initialize trend classifier.
        
        Args:
            fast_period: Fast MA period
            slow_period: Slow MA period  
            trend_threshold: Minimum MA difference for trend
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trend_threshold = trend_threshold
        
        # State
        self._prices: Deque[float] = deque(maxlen=slow_period)
        self._current_class: Optional[str] = None
        self._confidence: float = 0.0
        self._last_update: Optional[datetime] = None
        
    def classify(self, data: Dict[str, Any]) -> str:
        """Classify current market conditions."""
        price = data.get('close', data.get('price'))
        timestamp = data.get('timestamp', datetime.now())
        
        if price is None:
            return MarketRegime.RANGING.value
        
        # Update price history
        self._prices.append(price)
        self._last_update = timestamp
        
        # Need enough data
        if len(self._prices) < self.slow_period:
            self._current_class = MarketRegime.RANGING.value
            self._confidence = 0.0
            return self._current_class
        
        # Calculate moving averages
        fast_ma = sum(list(self._prices)[-self.fast_period:]) / self.fast_period
        slow_ma = sum(self._prices) / len(self._prices)
        
        # Calculate trend strength
        ma_diff = (fast_ma - slow_ma) / slow_ma
        
        # Classify based on MA relationship
        if ma_diff > self.trend_threshold:
            self._current_class = MarketRegime.TRENDING_UP.value
            self._confidence = min(ma_diff / (self.trend_threshold * 2), 1.0)
        elif ma_diff < -self.trend_threshold:
            self._current_class = MarketRegime.TRENDING_DOWN.value
            self._confidence = min(abs(ma_diff) / (self.trend_threshold * 2), 1.0)
        else:
            self._current_class = MarketRegime.RANGING.value
            self._confidence = 1.0 - (abs(ma_diff) / self.trend_threshold)
        
        return self._current_class
    
    @property
    def current_class(self) -> Optional[str]:
        """Current classification."""
        return self._current_class
    
    @property
    def confidence(self) -> float:
        """Confidence in current classification (0-1)."""
        return self._confidence
    
    def reset(self) -> None:
        """Reset classifier state."""
        self._prices.clear()
        self._current_class = None
        self._confidence = 0.0
        self._last_update = None


class VolatilityClassifier:
    """
    Classifies market volatility levels.
    
    Uses ATR or standard deviation to categorize volatility states.
    """
    
    def __init__(self,
                 lookback_period: int = 20,
                 high_vol_threshold: float = 1.5,
                 low_vol_threshold: float = 0.5):
        """
        Initialize volatility classifier.
        
        Args:
            lookback_period: Period for volatility calculation
            high_vol_threshold: Multiplier for high volatility (vs average)
            low_vol_threshold: Multiplier for low volatility
        """
        self.lookback_period = lookback_period
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        
        # State
        self._returns: Deque[float] = deque(maxlen=lookback_period)
        self._vol_history: Deque[float] = deque(maxlen=lookback_period * 3)
        self._last_price: Optional[float] = None
        self._current_class: Optional[str] = None
        self._confidence: float = 0.0
        
    def classify(self, data: Dict[str, Any]) -> str:
        """Classify current volatility state."""
        price = data.get('close', data.get('price'))
        
        if price is None:
            return MarketRegime.RANGING.value
        
        # Calculate return if we have previous price
        if self._last_price is not None and self._last_price > 0:
            ret = (price - self._last_price) / self._last_price
            self._returns.append(ret)
        
        self._last_price = price
        
        # Need enough data
        if len(self._returns) < self.lookback_period:
            self._current_class = MarketRegime.RANGING.value
            self._confidence = 0.0
            return self._current_class
        
        # Calculate current volatility (standard deviation of returns)
        mean_return = sum(self._returns) / len(self._returns)
        variance = sum((r - mean_return) ** 2 for r in self._returns) / len(self._returns)
        current_vol = variance ** 0.5
        
        # Track volatility history
        self._vol_history.append(current_vol)
        
        # Calculate average historical volatility
        if len(self._vol_history) >= self.lookback_period:
            avg_vol = sum(self._vol_history) / len(self._vol_history)
            
            # Classify based on current vs average volatility
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            if vol_ratio > self.high_vol_threshold:
                self._current_class = MarketRegime.HIGH_VOLATILITY.value
                self._confidence = min((vol_ratio - 1.0) / (self.high_vol_threshold - 1.0), 1.0)
            elif vol_ratio < self.low_vol_threshold:
                self._current_class = MarketRegime.LOW_VOLATILITY.value
                self._confidence = min((1.0 - vol_ratio) / (1.0 - self.low_vol_threshold), 1.0)
            else:
                self._current_class = MarketRegime.RANGING.value
                self._confidence = 0.5
        else:
            self._current_class = MarketRegime.RANGING.value
            self._confidence = 0.0
        
        return self._current_class
    
    @property
    def current_class(self) -> Optional[str]:
        """Current classification."""
        return self._current_class
    
    @property
    def confidence(self) -> float:
        """Confidence in current classification (0-1)."""
        return self._confidence
    
    def reset(self) -> None:
        """Reset classifier state."""
        self._returns.clear()
        self._vol_history.clear()
        self._last_price = None
        self._current_class = None
        self._confidence = 0.0


class CompositeClassifier:
    """
    Combines multiple classifiers for comprehensive market classification.
    
    This demonstrates how different classifiers can work together.
    """
    
    def __init__(self, classifiers: Optional[List[Classifier]] = None):
        """
        Initialize composite classifier.
        
        Args:
            classifiers: List of classifiers to combine
        """
        self.classifiers = classifiers or []
        self._current_class: Optional[str] = None
        self._confidence: float = 0.0
        self._classifications: Dict[str, str] = {}
        
    def add_classifier(self, name: str, classifier: Classifier) -> None:
        """Add a classifier to the composite."""
        self.classifiers.append((name, classifier))
        
    def classify(self, data: Dict[str, Any]) -> str:
        """
        Classify using all sub-classifiers.
        
        Returns a combined classification string.
        """
        if not self.classifiers:
            self._current_class = "UNCLASSIFIED"
            return self._current_class
        
        # Get classifications from all classifiers
        self._classifications.clear()
        confidences = []
        
        for name, classifier in self.classifiers:
            classification = classifier.classify(data)
            self._classifications[name] = classification
            confidences.append(classifier.confidence)
        
        # Combine classifications (simple concatenation for now)
        # In practice, this could use more sophisticated logic
        parts = []
        
        # Check for trend
        if 'trend' in self._classifications:
            trend = self._classifications['trend']
            if trend in [MarketRegime.TRENDING_UP.value, MarketRegime.TRENDING_DOWN.value]:
                parts.append(trend)
        
        # Check for volatility
        if 'volatility' in self._classifications:
            vol = self._classifications['volatility']
            if vol in [MarketRegime.HIGH_VOLATILITY.value, MarketRegime.LOW_VOLATILITY.value]:
                parts.append(vol)
        
        # Default if no strong classification
        if not parts:
            parts.append(MarketRegime.RANGING.value)
        
        self._current_class = "_".join(parts)
        self._confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return self._current_class
    
    @property
    def current_class(self) -> Optional[str]:
        """Current classification."""
        return self._current_class
    
    @property
    def confidence(self) -> float:
        """Confidence in current classification (0-1)."""
        return self._confidence
    
    def reset(self) -> None:
        """Reset classifier state."""
        for _, classifier in self.classifiers:
            classifier.reset()
        self._current_class = None
        self._confidence = 0.0
        self._classifications.clear()
    
    def get_all_classifications(self) -> Dict[str, str]:
        """Get classifications from all sub-classifiers."""
        return self._classifications.copy()


# Factory function
def create_market_regime_classifier() -> CompositeClassifier:
    """
    Create a comprehensive market regime classifier.
    
    This combines trend and volatility classification.
    """
    classifier = CompositeClassifier()
    classifier.add_classifier('trend', TrendClassifier())
    classifier.add_classifier('volatility', VolatilityClassifier())
    return classifier