"""
Volatility indicators.
"""

from typing import Optional, Deque, Dict, Any
from datetime import datetime
from collections import deque
import math

from ..protocols import Indicator


class ATR:
    """Average True Range indicator."""
    
    def __init__(self, period: int = 14):
        self.period = period
        self._current_value: Optional[float] = None
        self._tr_values: Deque[float] = deque(maxlen=period)
        self._atr: Optional[float] = None
        self._prev_close: Optional[float] = None
        self._count = 0
    
    def calculate(self, bar_data: Dict[str, Any], timestamp: datetime) -> Optional[float]:
        """Calculate ATR from bar data."""
        high = bar_data.get('high', bar_data.get('close', 0))
        low = bar_data.get('low', bar_data.get('close', 0))
        close = bar_data.get('close', 0)
        
        if self._prev_close is not None:
            # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close)
            )
            
            self._tr_values.append(tr)
            self._count += 1
            
            if self._count >= self.period:
                if self._atr is None:
                    # First ATR is simple average
                    self._atr = sum(self._tr_values) / len(self._tr_values)
                else:
                    # Subsequent ATRs use smoothing
                    self._atr = ((self._atr * (self.period - 1)) + tr) / self.period
                
                self._current_value = self._atr
        
        self._prev_close = close
        return self._current_value
    
    @property
    def value(self) -> Optional[float]:
        """Current ATR value."""
        return self._current_value
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data."""
        return self._count >= self.period
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._current_value = None
        self._tr_values.clear()
        self._atr = None
        self._prev_close = None
        self._count = 0


class BollingerBands:
    """Bollinger Bands indicator."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev_multiplier = std_dev
        self.values: Deque[float] = deque(maxlen=period)
        self.upper_band: Optional[float] = None
        self.middle_band: Optional[float] = None
        self.lower_band: Optional[float] = None
        self.bandwidth: Optional[float] = None
        self.percent_b: Optional[float] = None
        self.std_dev: Optional[float] = None
    
    def calculate(self, value: float, timestamp: datetime) -> Optional[float]:
        """Calculate Bollinger Bands."""
        self.values.append(value)
        
        if self.ready:
            # Calculate middle band (SMA)
            self.middle_band = sum(self.values) / len(self.values)
            
            # Calculate standard deviation
            variance = sum((x - self.middle_band) ** 2 for x in self.values) / len(self.values)
            self.std_dev = math.sqrt(variance)
            
            # Calculate bands
            band_width = self.std_dev_multiplier * self.std_dev
            self.upper_band = self.middle_band + band_width
            self.lower_band = self.middle_band - band_width
            
            # Calculate bandwidth
            if self.middle_band > 0:
                self.bandwidth = (self.upper_band - self.lower_band) / self.middle_band
            
            # Calculate %B
            if self.upper_band != self.lower_band:
                self.percent_b = (value - self.lower_band) / (self.upper_band - self.lower_band)
            
            return self.bandwidth
        
        return None
    
    @property
    def value(self) -> Optional[float]:
        """Current bandwidth value."""
        return self.bandwidth
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data."""
        return len(self.values) >= self.period
    
    def reset(self) -> None:
        """Reset indicator state."""
        self.values.clear()
        self.upper_band = None
        self.middle_band = None
        self.lower_band = None
        self.bandwidth = None
        self.percent_b = None
        self.std_dev = None


class ADX:
    """Average Directional Index indicator."""
    
    def __init__(self, period: int = 14):
        self.period = period
        self._current_value: Optional[float] = None
        self._count = 0
        
        # Directional movement
        self._prev_high: Optional[float] = None
        self._prev_low: Optional[float] = None
        self._tr_ema: Optional[float] = None
        self._plus_dm_ema: Optional[float] = None
        self._minus_dm_ema: Optional[float] = None
        self._dx_values: Deque[float] = deque(maxlen=period)
    
    def calculate(self, bar_data: Dict[str, Any], timestamp: datetime) -> Optional[float]:
        """Calculate ADX from bar data."""
        high = bar_data.get('high', bar_data.get('close', 0))
        low = bar_data.get('low', bar_data.get('close', 0))
        
        if self._prev_high is not None and self._prev_low is not None:
            # Calculate directional movement
            up_move = high - self._prev_high
            down_move = self._prev_low - low
            
            plus_dm = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm = down_move if down_move > up_move and down_move > 0 else 0
            
            # True range
            tr = max(
                high - low,
                abs(high - self._prev_close) if hasattr(self, '_prev_close') else 0,
                abs(low - self._prev_close) if hasattr(self, '_prev_close') else 0
            )
            
            # Update EMAs
            multiplier = 1.0 / self.period
            
            if self._tr_ema is None:
                self._tr_ema = tr
                self._plus_dm_ema = plus_dm
                self._minus_dm_ema = minus_dm
            else:
                self._tr_ema = self._tr_ema + multiplier * (tr - self._tr_ema)
                self._plus_dm_ema = self._plus_dm_ema + multiplier * (plus_dm - self._plus_dm_ema)
                self._minus_dm_ema = self._minus_dm_ema + multiplier * (minus_dm - self._minus_dm_ema)
            
            self._count += 1
            
            if self._count >= self.period and self._tr_ema > 0:
                # Calculate directional indicators
                plus_di = 100 * self._plus_dm_ema / self._tr_ema
                minus_di = 100 * self._minus_dm_ema / self._tr_ema
                
                # Calculate DX
                di_sum = plus_di + minus_di
                if di_sum > 0:
                    dx = 100 * abs(plus_di - minus_di) / di_sum
                    self._dx_values.append(dx)
                    
                    # Calculate ADX
                    if len(self._dx_values) >= self.period:
                        self._current_value = sum(self._dx_values) / len(self._dx_values)
        
        self._prev_high = high
        self._prev_low = low
        if hasattr(bar_data, 'get'):
            self._prev_close = bar_data.get('close', 0)
        
        return self._current_value
    
    @property
    def value(self) -> Optional[float]:
        """Current ADX value."""
        return self._current_value
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data."""
        return self._count >= self.period * 2
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._current_value = None
        self._count = 0
        self._prev_high = None
        self._prev_low = None
        self._tr_ema = None
        self._plus_dm_ema = None
        self._minus_dm_ema = None
        self._dx_values.clear()