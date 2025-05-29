"""
Momentum-based indicators.
"""

from typing import Optional, Deque
from datetime import datetime
from collections import deque
import math

from ..protocols import Indicator


class RSI:
    """Relative Strength Index indicator."""
    
    def __init__(self, period: int = 14):
        self.period = period
        self._current_value: Optional[float] = None
        self._prev_price: Optional[float] = None
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._count = 0
        
        # For initial SMA calculation
        self._gains: Deque[float] = deque(maxlen=period)
        self._losses: Deque[float] = deque(maxlen=period)
        
    def calculate(self, value: float, timestamp: datetime) -> Optional[float]:
        """Calculate RSI value."""
        if self._prev_price is None:
            self._prev_price = value
            return None
        
        # Calculate price change
        change = value - self._prev_price
        gain = max(0, change)
        loss = max(0, -change)
        
        self._prev_price = value
        self._count += 1
        
        if self._count <= self.period:
            # Initial period - collect gains and losses
            self._gains.append(gain)
            self._losses.append(loss)
            
            if self._count == self.period:
                # Calculate initial averages
                self._avg_gain = sum(self._gains) / self.period
                self._avg_loss = sum(self._losses) / self.period
        else:
            # Use smoothed averages
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period
        
        if self.ready:
            if self._avg_loss == 0:
                self._current_value = 100.0
            else:
                rs = self._avg_gain / self._avg_loss
                self._current_value = 100.0 - (100.0 / (1.0 + rs))
            
            return self._current_value
        
        return None
    
    @property
    def value(self) -> Optional[float]:
        """Current RSI value."""
        return self._current_value
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data."""
        return self._count >= self.period
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._current_value = None
        self._prev_price = None
        self._avg_gain = None
        self._avg_loss = None
        self._count = 0
        self._gains.clear()
        self._losses.clear()


class MACD:
    """Moving Average Convergence Divergence indicator."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # EMA components
        self._fast_ema: Optional[float] = None
        self._slow_ema: Optional[float] = None
        self._signal_ema: Optional[float] = None
        
        # Multipliers
        self._fast_mult = 2.0 / (fast_period + 1)
        self._slow_mult = 2.0 / (slow_period + 1)
        self._signal_mult = 2.0 / (signal_period + 1)
        
        # Values
        self._macd_line: Optional[float] = None
        self._signal_line: Optional[float] = None
        self._histogram: Optional[float] = None
        
        self._count = 0
        
    def calculate(self, value: float, timestamp: datetime) -> Optional[float]:
        """Calculate MACD values."""
        # Update EMAs
        if self._fast_ema is None:
            self._fast_ema = value
            self._slow_ema = value
        else:
            self._fast_ema = (value - self._fast_ema) * self._fast_mult + self._fast_ema
            self._slow_ema = (value - self._slow_ema) * self._slow_mult + self._slow_ema
        
        self._count += 1
        
        # Calculate MACD line
        if self._count >= self.slow_period:
            self._macd_line = self._fast_ema - self._slow_ema
            
            # Update signal line
            if self._signal_ema is None:
                self._signal_ema = self._macd_line
            else:
                self._signal_ema = (self._macd_line - self._signal_ema) * self._signal_mult + self._signal_ema
            
            # Calculate histogram
            if self._count >= self.slow_period + self.signal_period:
                self._signal_line = self._signal_ema
                self._histogram = self._macd_line - self._signal_line
                return self._histogram
        
        return None
    
    @property
    def value(self) -> Optional[float]:
        """Current MACD histogram value."""
        return self._histogram
    
    @property
    def macd_line(self) -> Optional[float]:
        """Current MACD line value."""
        return self._macd_line
    
    @property
    def signal_line(self) -> Optional[float]:
        """Current signal line value."""
        return self._signal_line
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data."""
        return self._count >= self.slow_period + self.signal_period
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._fast_ema = None
        self._slow_ema = None
        self._signal_ema = None
        self._macd_line = None
        self._signal_line = None
        self._histogram = None
        self._count = 0


class Momentum:
    """Simple momentum indicator."""
    
    def __init__(self, period: int = 10):
        self.period = period
        self.prices: Deque[float] = deque(maxlen=period + 1)
        self._current_value: Optional[float] = None
        
    def calculate(self, value: float, timestamp: datetime) -> Optional[float]:
        """Calculate momentum value."""
        self.prices.append(value)
        
        if len(self.prices) > self.period:
            # Momentum = Current Price - Price N periods ago
            self._current_value = value - self.prices[0]
            return self._current_value
        
        return None
    
    @property
    def value(self) -> Optional[float]:
        """Current momentum value."""
        return self._current_value
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data."""
        return len(self.prices) > self.period
    
    def reset(self) -> None:
        """Reset indicator state."""
        self.prices.clear()
        self._current_value = None


class RateOfChange:
    """Rate of Change (ROC) indicator."""
    
    def __init__(self, period: int = 10):
        self.period = period
        self.prices: Deque[float] = deque(maxlen=period + 1)
        self._current_value: Optional[float] = None
        
    def calculate(self, value: float, timestamp: datetime) -> Optional[float]:
        """Calculate ROC value."""
        self.prices.append(value)
        
        if len(self.prices) > self.period:
            # ROC = ((Current Price - Price N periods ago) / Price N periods ago) * 100
            old_price = self.prices[0]
            if old_price != 0:
                self._current_value = ((value - old_price) / old_price) * 100
                return self._current_value
        
        return None
    
    @property
    def value(self) -> Optional[float]:
        """Current ROC value."""
        return self._current_value
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data."""
        return len(self.prices) > self.period
    
    def reset(self) -> None:
        """Reset indicator state."""
        self.prices.clear()
        self._current_value = None