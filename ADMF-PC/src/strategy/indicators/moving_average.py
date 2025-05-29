"""
Moving average indicators.
"""

from typing import List, Optional, Deque
from datetime import datetime
from collections import deque

from ..protocols import Indicator


class SimpleMovingAverage:
    """Simple Moving Average (SMA) indicator."""
    
    def __init__(self, period: int):
        self.period = period
        self.values: Deque[float] = deque(maxlen=period)
        self._current_value: Optional[float] = None
        
    def calculate(self, value: float, timestamp: datetime) -> Optional[float]:
        """Calculate SMA value."""
        self.values.append(value)
        
        if self.ready:
            self._current_value = sum(self.values) / len(self.values)
            return self._current_value
        
        return None
    
    @property
    def value(self) -> Optional[float]:
        """Current SMA value."""
        return self._current_value
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data."""
        return len(self.values) >= self.period
    
    def reset(self) -> None:
        """Reset indicator state."""
        self.values.clear()
        self._current_value = None


class ExponentialMovingAverage:
    """Exponential Moving Average (EMA) indicator."""
    
    def __init__(self, period: int):
        self.period = period
        self.multiplier = 2.0 / (period + 1)
        self._current_value: Optional[float] = None
        self._count = 0
        
    def calculate(self, value: float, timestamp: datetime) -> Optional[float]:
        """Calculate EMA value."""
        if self._current_value is None:
            # First value
            self._current_value = value
        else:
            # EMA = (Close - Previous EMA) * multiplier + Previous EMA
            self._current_value = (value - self._current_value) * self.multiplier + self._current_value
        
        self._count += 1
        
        if self.ready:
            return self._current_value
        
        return None
    
    @property
    def value(self) -> Optional[float]:
        """Current EMA value."""
        return self._current_value if self.ready else None
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data."""
        return self._count >= self.period
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._current_value = None
        self._count = 0


class WeightedMovingAverage:
    """Weighted Moving Average (WMA) indicator."""
    
    def __init__(self, period: int):
        self.period = period
        self.values: Deque[float] = deque(maxlen=period)
        self._current_value: Optional[float] = None
        
        # Pre-calculate weights
        self.weights = list(range(1, period + 1))
        self.weight_sum = sum(self.weights)
        
    def calculate(self, value: float, timestamp: datetime) -> Optional[float]:
        """Calculate WMA value."""
        self.values.append(value)
        
        if self.ready:
            weighted_sum = sum(val * weight 
                             for val, weight in zip(self.values, self.weights))
            self._current_value = weighted_sum / self.weight_sum
            return self._current_value
        
        return None
    
    @property
    def value(self) -> Optional[float]:
        """Current WMA value."""
        return self._current_value
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data."""
        return len(self.values) >= self.period
    
    def reset(self) -> None:
        """Reset indicator state."""
        self.values.clear()
        self._current_value = None