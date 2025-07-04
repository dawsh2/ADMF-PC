"""
Volatility Percentile Feature

Calculates the percentile rank of current volatility within a rolling window.
This helps identify whether we're in low, normal, or high volatility conditions.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from collections import deque

from ..protocols import FeatureState
from .volatility import ATR


class VolatilityPercentile:
    """
    Calculates the percentile rank of current volatility.
    
    Uses ATR (Average True Range) as the volatility measure and ranks it
    against a rolling window of historical ATR values.
    
    Returns a value between 0 and 100:
    - 0 = lowest volatility in the window
    - 50 = median volatility
    - 100 = highest volatility in the window
    """
    
    def __init__(self, period: int = 50, atr_period: int = 14):
        """
        Initialize volatility percentile calculator.
        
        Args:
            period: Number of bars to use for percentile calculation (default: 50)
            atr_period: Period for ATR calculation (default: 14)
        """
        self.period = period
        self.atr_period = atr_period
        self.atr_values = deque(maxlen=period)
        
        # For ATR calculation
        self.true_ranges = deque(maxlen=atr_period)
        self.current_atr = None
        self.prev_close = None
        
    def update(self, bar: Dict[str, Any]) -> None:
        """Update with new bar data."""
        high = bar.get('high', bar.get('High', 0))
        low = bar.get('low', bar.get('Low', 0))
        close = bar.get('close', bar.get('Close', 0))
        
        if self.prev_close is not None:
            # Calculate true range
            true_range = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close)
            )
            self.true_ranges.append(true_range)
            
            # Calculate ATR
            if len(self.true_ranges) >= self.atr_period:
                self.current_atr = np.mean(self.true_ranges)
                self.atr_values.append(self.current_atr)
        
        self.prev_close = close
    
    def value(self) -> float:
        """
        Get current volatility percentile.
        
        Returns:
            Percentile rank (0-100) of current volatility
        """
        if not self.atr_values or len(self.atr_values) < 2 or self.current_atr is None:
            return 50.0  # Default to median if insufficient data
        
        # Calculate percentile rank
        # Count how many historical values are below current
        below_count = sum(1 for atr in self.atr_values if atr < self.current_atr)
        percentile = (below_count / len(self.atr_values)) * 100
        
        return percentile
    
    def values(self) -> Dict[str, Any]:
        """Get all calculated values."""
        percentile = self.value()
        return {
            'value': percentile,
            'percentile': percentile,
            'current_atr': self.current_atr if self.current_atr is not None else 0,
            'is_high_vol': percentile > 70,
            'is_low_vol': percentile < 30
        }
    
    def reset(self) -> None:
        """Reset calculator state."""
        self.atr_values.clear()
        self.true_ranges.clear()
        self.current_atr = None
        self.prev_close = None
    
    @property
    def min_periods(self) -> int:
        """Minimum periods before first valid calculation."""
        return self.atr_period + 1  # Need at least ATR period + 1 for percentile
    
    @property
    def feature_names(self) -> List[str]:
        """Names of features this calculator produces."""
        return ['volatility_percentile', 'current_atr', 'is_high_vol', 'is_low_vol']


# Component registration removed - handled by feature hub