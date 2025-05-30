"""
Technical indicator implementations.

Simple indicators that implement the Indicator protocol without inheritance.
These can be enhanced with capabilities as needed.
"""

from typing import Optional, List, Deque
from datetime import datetime
from collections import deque

from ..protocols import Indicator


class SimpleMovingAverage:
    """
    Simple moving average indicator.
    
    No inheritance - just implements the Indicator protocol.
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize SMA indicator.
        
        Args:
            period: Number of periods for average
        """
        self.period = period
        self._values: Deque[float] = deque(maxlen=period)
        self._current_value: Optional[float] = None
        
    def calculate(self, value: float, timestamp: Optional[datetime] = None) -> Optional[float]:
        """Calculate indicator value for new data point."""
        self._values.append(value)
        
        if len(self._values) >= self.period:
            self._current_value = sum(self._values) / len(self._values)
            return self._current_value
        
        return None
    
    def update(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Update indicator state without returning value."""
        self.calculate(value, timestamp)
    
    @property
    def value(self) -> Optional[float]:
        """Current indicator value."""
        return self._current_value
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data to produce values."""
        return len(self._values) >= self.period
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._values.clear()
        self._current_value = None


class ExponentialMovingAverage:
    """
    Exponential moving average indicator.
    
    Gives more weight to recent values.
    """
    
    def __init__(self, period: int = 20, smoothing: float = 2.0):
        """
        Initialize EMA indicator.
        
        Args:
            period: Number of periods for average
            smoothing: Smoothing factor (typically 2)
        """
        self.period = period
        self.smoothing = smoothing
        self.multiplier = smoothing / (period + 1)
        
        self._ema: Optional[float] = None
        self._count: int = 0
        self._sma_sum: float = 0.0
        
    def calculate(self, value: float, timestamp: Optional[datetime] = None) -> Optional[float]:
        """Calculate indicator value for new data point."""
        self._count += 1
        
        if self._ema is None:
            # Use SMA for first 'period' values
            self._sma_sum += value
            if self._count >= self.period:
                self._ema = self._sma_sum / self.period
                return self._ema
        else:
            # EMA = (Value - EMA_prev) * multiplier + EMA_prev
            self._ema = (value - self._ema) * self.multiplier + self._ema
            return self._ema
        
        return None
    
    def update(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Update indicator state without returning value."""
        self.calculate(value, timestamp)
    
    @property
    def value(self) -> Optional[float]:
        """Current indicator value."""
        return self._ema
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data to produce values."""
        return self._ema is not None
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._ema = None
        self._count = 0
        self._sma_sum = 0.0


class RSI:
    """
    Relative Strength Index indicator.
    
    Measures momentum - overbought/oversold conditions.
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize RSI indicator.
        
        Args:
            period: Number of periods for RSI calculation
        """
        self.period = period
        self._gains: Deque[float] = deque(maxlen=period)
        self._losses: Deque[float] = deque(maxlen=period)
        self._last_price: Optional[float] = None
        self._current_rsi: Optional[float] = None
        
    def calculate(self, value: float, timestamp: Optional[datetime] = None) -> Optional[float]:
        """Calculate indicator value for new data point."""
        if self._last_price is not None:
            change = value - self._last_price
            gain = max(0, change)
            loss = max(0, -change)
            
            self._gains.append(gain)
            self._losses.append(loss)
            
            if len(self._gains) >= self.period:
                avg_gain = sum(self._gains) / self.period
                avg_loss = sum(self._losses) / self.period
                
                if avg_loss == 0:
                    self._current_rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    self._current_rsi = 100 - (100 / (1 + rs))
                
                return self._current_rsi
        
        self._last_price = value
        return None
    
    def update(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Update indicator state without returning value."""
        self.calculate(value, timestamp)
    
    @property
    def value(self) -> Optional[float]:
        """Current indicator value."""
        return self._current_rsi
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data to produce values."""
        return self._current_rsi is not None
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._gains.clear()
        self._losses.clear()
        self._last_price = None
        self._current_rsi = None


class ATR:
    """
    Average True Range indicator.
    
    Measures volatility using the true range.
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize ATR indicator.
        
        Args:
            period: Number of periods for ATR calculation
        """
        self.period = period
        self._true_ranges: Deque[float] = deque(maxlen=period)
        self._last_close: Optional[float] = None
        self._current_atr: Optional[float] = None
        
    def calculate(self, value: float, timestamp: Optional[datetime] = None, 
                  high: Optional[float] = None, low: Optional[float] = None) -> Optional[float]:
        """
        Calculate indicator value for new data point.
        
        For ATR, we need high/low/close values. If only close is provided,
        we approximate using close price.
        """
        if high is None:
            high = value
        if low is None:
            low = value
            
        if self._last_close is not None:
            # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr1 = high - low
            tr2 = abs(high - self._last_close)
            tr3 = abs(low - self._last_close)
            true_range = max(tr1, tr2, tr3)
        else:
            # First value - just use high-low
            true_range = high - low
        
        self._true_ranges.append(true_range)
        
        if len(self._true_ranges) >= self.period:
            self._current_atr = sum(self._true_ranges) / len(self._true_ranges)
            
        self._last_close = value
        return self._current_atr
    
    def update(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Update indicator state without returning value."""
        self.calculate(value, timestamp)
    
    @property
    def value(self) -> Optional[float]:
        """Current indicator value."""
        return self._current_atr
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data to produce values."""
        return self._current_atr is not None
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._true_ranges.clear()
        self._last_close = None
        self._current_atr = None


# Factory functions for creating indicators with capabilities
def create_sma(period: int = 20) -> SimpleMovingAverage:
    """Create SMA indicator."""
    return SimpleMovingAverage(period)


def create_ema(period: int = 20, smoothing: float = 2.0) -> ExponentialMovingAverage:
    """Create EMA indicator."""
    return ExponentialMovingAverage(period, smoothing)


def create_rsi(period: int = 14) -> RSI:
    """Create RSI indicator."""
    return RSI(period)


def create_atr(period: int = 14) -> ATR:
    """Create ATR indicator."""
    return ATR(period)