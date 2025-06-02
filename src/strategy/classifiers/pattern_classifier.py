"""
File: src/strategy/classifiers/pattern_classifier.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#pattern-classifier
Step: 3 - Classifier Container
Dependencies: collections

Pattern-based market regime classifier.
Uses technical indicators and price patterns for regime detection.
"""

from __future__ import annotations
from typing import Dict, Any, List
from collections import deque

from .classifier import BaseClassifier
from .regime_types import MarketRegime, ClassifierConfig
from ...data.models import Bar


class PatternClassifier(BaseClassifier):
    """
    Rule-based pattern classifier for market regimes.
    
    Uses technical indicators and price patterns to classify
    market conditions into trending, ranging, or volatile regimes.
    
    Classification Logic:
    - High volatility → VOLATILE regime
    - Strong trend + low volatility → TRENDING regime  
    - Weak trend + low volatility → RANGING regime
    """
    
    def __init__(self, config: ClassifierConfig):
        """
        Initialize pattern classifier.
        
        Args:
            config: Classifier configuration
        """
        super().__init__(config)
        
        # Configuration
        self.atr_period = config.atr_period
        self.trend_period = config.trend_period
        self.volatility_threshold = config.volatility_threshold
        self.trend_threshold = config.trend_threshold
        
        # Technical indicators
        self.price_history = deque(maxlen=self.trend_period)
        self.high_history = deque(maxlen=self.atr_period)
        self.low_history = deque(maxlen=self.atr_period)
        self.close_history = deque(maxlen=self.atr_period)
        self.volume_history = deque(maxlen=20)
        
        # ATR calculation
        self.true_ranges = deque(maxlen=self.atr_period)
        self.atr_value = 0.0
        
        # SMA calculation
        self.sma_value = 0.0
        
        # Trend calculation
        self.trend_strength = 0.0
        self.trend_direction = 0.0
        
        # Volatility metrics
        self.normalized_volatility = 0.0
        
    def classify(self) -> MarketRegime:
        """
        Classify market regime based on patterns.
        
        Returns:
            Classified market regime
        """
        if not self.is_ready:
            return MarketRegime.UNKNOWN
        
        # Update all indicators
        self._update_indicators()
        
        # Calculate key metrics
        volatility = self._calculate_normalized_volatility()
        trend_strength = self._calculate_trend_strength()
        trend_consistency = self._calculate_trend_consistency()
        
        # Classification logic
        if volatility > self.volatility_threshold:
            return MarketRegime.VOLATILE
        elif abs(trend_strength) > self.trend_threshold and trend_consistency > 0.6:
            return MarketRegime.TRENDING
        else:
            return MarketRegime.RANGING
    
    def _calculate_confidence(self) -> float:
        """
        Calculate classification confidence.
        
        Returns:
            Confidence level (0.0 to 1.0)
        """
        if not self.is_ready:
            return 0.0
        
        # Calculate key metrics
        volatility = self._calculate_normalized_volatility()
        trend_strength = abs(self._calculate_trend_strength())
        trend_consistency = self._calculate_trend_consistency()
        
        # Confidence based on clarity of classification
        if volatility > self.volatility_threshold:
            # Volatile regime - confidence based on how much above threshold
            excess_volatility = (volatility - self.volatility_threshold) / self.volatility_threshold
            confidence = min(0.6 + excess_volatility * 0.3, 0.95)
        elif trend_strength > self.trend_threshold:
            # Trending regime - confidence based on trend strength and consistency
            trend_clarity = (trend_strength - self.trend_threshold) / (1.0 - self.trend_threshold)
            confidence = 0.5 + trend_clarity * 0.3 + trend_consistency * 0.2
        else:
            # Ranging regime - confidence based on low trend strength and volatility
            range_clarity = 1.0 - (trend_strength / self.trend_threshold)
            vol_clarity = 1.0 - (volatility / self.volatility_threshold)
            confidence = 0.4 + (range_clarity + vol_clarity) * 0.25
        
        return min(max(confidence, 0.1), 0.95)
    
    def _update_indicators(self) -> None:
        """Update all technical indicators."""
        if len(self.bar_history) < 2:
            return
        
        current_bar = self.bar_history[-1]
        
        # Update price histories
        self.price_history.append(current_bar.close)
        self.high_history.append(current_bar.high)
        self.low_history.append(current_bar.low)
        self.close_history.append(current_bar.close)
        self.volume_history.append(current_bar.volume)
        
        # Update ATR
        self._update_atr()
        
        # Update SMA
        self._update_sma()
    
    def _update_atr(self) -> None:
        """Update Average True Range."""
        if len(self.bar_history) < 2:
            return
        
        current_bar = self.bar_history[-1]
        prev_bar = self.bar_history[-2]
        
        # Calculate True Range
        high_low = current_bar.high - current_bar.low
        high_close = abs(current_bar.high - prev_bar.close)
        low_close = abs(current_bar.low - prev_bar.close)
        
        true_range = max(high_low, high_close, low_close)
        self.true_ranges.append(true_range)
        
        # Calculate ATR (simple moving average of true ranges)
        if len(self.true_ranges) >= self.atr_period:
            self.atr_value = sum(self.true_ranges) / len(self.true_ranges)
    
    def _update_sma(self) -> None:
        """Update Simple Moving Average."""
        if len(self.price_history) >= self.trend_period:
            self.sma_value = sum(self.price_history) / len(self.price_history)
    
    def _calculate_normalized_volatility(self) -> float:
        """
        Calculate normalized volatility.
        
        Returns:
            Normalized volatility (ATR / Price)
        """
        if not self.price_history or self.atr_value == 0:
            return 0.0
        
        current_price = self.price_history[-1]
        if current_price == 0:
            return 0.0
        
        self.normalized_volatility = self.atr_value / current_price
        return self.normalized_volatility
    
    def _calculate_trend_strength(self) -> float:
        """
        Calculate trend strength using linear regression.
        
        Returns:
            Trend strength (-1.0 to 1.0, negative = downtrend)
        """
        if len(self.price_history) < self.trend_period:
            return 0.0
        
        prices = list(self.price_history)
        n = len(prices)
        
        # Calculate linear regression slope
        x_sum = sum(range(n))
        y_sum = sum(prices)
        xy_sum = sum(i * prices[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        # Avoid division by zero
        denominator = n * x2_sum - x_sum * x_sum
        if denominator == 0:
            return 0.0
        
        slope = (n * xy_sum - x_sum * y_sum) / denominator
        
        # Normalize slope by average price
        avg_price = y_sum / n
        if avg_price == 0:
            return 0.0
        
        normalized_slope = slope / avg_price
        
        # Scale to reasonable range
        self.trend_strength = max(min(normalized_slope * 100, 1.0), -1.0)
        return self.trend_strength
    
    def _calculate_trend_consistency(self) -> float:
        """
        Calculate trend consistency using price direction changes.
        
        Returns:
            Trend consistency (0.0 to 1.0)
        """
        if len(self.price_history) < 5:
            return 0.0
        
        prices = list(self.price_history)[-10:]  # Use last 10 prices
        direction_changes = 0
        total_moves = 0
        
        for i in range(1, len(prices) - 1):
            prev_direction = 1 if prices[i] > prices[i-1] else -1
            curr_direction = 1 if prices[i+1] > prices[i] else -1
            
            if prev_direction != curr_direction:
                direction_changes += 1
            total_moves += 1
        
        if total_moves == 0:
            return 0.0
        
        # Consistency = 1 - (direction_changes / total_moves)
        consistency = 1.0 - (direction_changes / total_moves)
        return max(min(consistency, 1.0), 0.0)
    
    def get_classification_details(self) -> Dict[str, Any]:
        """
        Get detailed classification information.
        
        Returns:
            Dictionary with classification details
        """
        if not self.is_ready:
            return {
                'regime': MarketRegime.UNKNOWN.value,
                'confidence': 0.0,
                'ready': False
            }
        
        return {
            'regime': self.current_regime.value,
            'confidence': self.confidence,
            'ready': True,
            'metrics': {
                'volatility': self._calculate_normalized_volatility(),
                'trend_strength': self._calculate_trend_strength(),
                'trend_consistency': self._calculate_trend_consistency(),
                'atr': self.atr_value,
                'sma': self.sma_value
            },
            'thresholds': {
                'volatility_threshold': self.volatility_threshold,
                'trend_threshold': self.trend_threshold
            },
            'data_points': {
                'bars_processed': len(self.bar_history),
                'price_history_length': len(self.price_history),
                'atr_period': self.atr_period,
                'trend_period': self.trend_period
            }
        }
    
    def reset(self) -> None:
        """Reset classifier to initial state."""
        super().reset()
        
        # Reset technical indicators
        self.price_history.clear()
        self.high_history.clear()
        self.low_history.clear()
        self.close_history.clear()
        self.volume_history.clear()
        self.true_ranges.clear()
        
        # Reset calculated values
        self.atr_value = 0.0
        self.sma_value = 0.0
        self.trend_strength = 0.0
        self.trend_direction = 0.0
        self.normalized_volatility = 0.0