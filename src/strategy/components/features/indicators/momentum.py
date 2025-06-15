"""
Momentum-based incremental features.

Features that analyze price momentum and directional strength.
All features maintain state and update in O(1) time complexity.

Uses protocol + composition architecture - no inheritance.
"""

from typing import Optional, Dict, Any
from collections import deque
from ..protocols import Feature, FeatureState
from .trend import EMA


class MACD:
    """MACD with O(1) updates."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9, name: str = "macd"):
        self._state = FeatureState(name)
        self.fast_ema = EMA(fast_period, name=f"{name}_fast")
        self.slow_ema = EMA(slow_period, name=f"{name}_slow")
        self.signal_ema = EMA(signal_period, name=f"{name}_signal")
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, float]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, **kwargs) -> Optional[Dict[str, float]]:
        fast_val = self.fast_ema.update(price)
        slow_val = self.slow_ema.update(price)
        
        if self.fast_ema.is_ready and self.slow_ema.is_ready:
            macd_line = fast_val - slow_val
            signal_line = self.signal_ema.update(macd_line)
            
            if self.signal_ema.is_ready:
                self._state.set_value({
                    "macd": macd_line,
                    "signal": signal_line,
                    "histogram": macd_line - signal_line
                })
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self.fast_ema.reset()
        self.slow_ema.reset()
        self.signal_ema.reset()


class Momentum:
    """Price Momentum with O(1) updates."""
    
    def __init__(self, period: int = 10, name: str = "momentum"):
        self._state = FeatureState(name)
        self.period = period
        self._price_buffer = deque(maxlen=period + 1)
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[float]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, **kwargs) -> Optional[float]:
        self._price_buffer.append(price)
        
        if len(self._price_buffer) == self.period + 1:
            old_price = self._price_buffer[0]
            momentum = price - old_price
            self._state.set_value(momentum)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._price_buffer.clear()


class ROC:
    """Rate of Change with O(1) updates."""
    
    def __init__(self, period: int = 10, name: str = "roc"):
        self._state = FeatureState(name)
        self.period = period
        self._price_buffer = deque(maxlen=period + 1)
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[float]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, **kwargs) -> Optional[float]:
        self._price_buffer.append(price)
        
        if len(self._price_buffer) == self.period + 1:
            old_price = self._price_buffer[0]
            if old_price > 0:
                roc = ((price - old_price) / old_price) * 100
                self._state.set_value(roc)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._price_buffer.clear()


class ADX:
    """Average Directional Index with O(1) updates."""
    
    def __init__(self, period: int = 14, name: str = "adx"):
        self._state = FeatureState(name)
        self.period = period
        self._prev_high: Optional[float] = None
        self._prev_low: Optional[float] = None
        self._prev_close: Optional[float] = None
        
        # Rolling buffers for calculations
        self._tr_buffer = deque(maxlen=period)
        self._dm_plus_buffer = deque(maxlen=period)
        self._dm_minus_buffer = deque(maxlen=period)
        self._dx_buffer = deque(maxlen=period)
        
        # Sums for efficiency
        self._tr_sum = 0.0
        self._dm_plus_sum = 0.0
        self._dm_minus_sum = 0.0
        self._dx_sum = 0.0
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, float]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, high: Optional[float] = None, 
               low: Optional[float] = None, **kwargs) -> Optional[Dict[str, float]]:
        if high is None or low is None:
            raise ValueError("ADX requires high and low prices")
        
        if self._prev_high is not None and self._prev_low is not None and self._prev_close is not None:
            # Calculate True Range
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close)
            )
            
            # Calculate Directional Movement
            dm_plus = max(high - self._prev_high, 0) if (high - self._prev_high) > (self._prev_low - low) else 0
            dm_minus = max(self._prev_low - low, 0) if (self._prev_low - low) > (high - self._prev_high) else 0
            
            # Update buffers
            if len(self._tr_buffer) == self.period:
                self._tr_sum -= self._tr_buffer[0]
                self._dm_plus_sum -= self._dm_plus_buffer[0]
                self._dm_minus_sum -= self._dm_minus_buffer[0]
            
            self._tr_buffer.append(tr)
            self._dm_plus_buffer.append(dm_plus)
            self._dm_minus_buffer.append(dm_minus)
            self._tr_sum += tr
            self._dm_plus_sum += dm_plus
            self._dm_minus_sum += dm_minus
            
            if len(self._tr_buffer) == self.period and self._tr_sum > 0:
                # Calculate Directional Indicators
                di_plus = (self._dm_plus_sum / self._tr_sum) * 100
                di_minus = (self._dm_minus_sum / self._tr_sum) * 100
                
                # Calculate DX
                di_sum = di_plus + di_minus
                if di_sum > 0:
                    dx = abs(di_plus - di_minus) / di_sum * 100
                    
                    # Update DX buffer for ADX calculation
                    if len(self._dx_buffer) == self.period:
                        self._dx_sum -= self._dx_buffer[0]
                    
                    self._dx_buffer.append(dx)
                    self._dx_sum += dx
                    
                    if len(self._dx_buffer) == self.period:
                        adx = self._dx_sum / self.period
                        self._state.set_value({
                            "adx": adx,
                            "di_plus": di_plus,
                            "di_minus": di_minus,
                            "dx": dx
                        })
        
        self._prev_high = high
        self._prev_low = low
        self._prev_close = price
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._prev_high = None
        self._prev_low = None
        self._prev_close = None
        self._tr_buffer.clear()
        self._dm_plus_buffer.clear()
        self._dm_minus_buffer.clear()
        self._dx_buffer.clear()
        self._tr_sum = 0.0
        self._dm_plus_sum = 0.0
        self._dm_minus_sum = 0.0
        self._dx_sum = 0.0


class Aroon:
    """Aroon Oscillator with O(1) updates."""
    
    def __init__(self, period: int = 25, name: str = "aroon"):
        self._state = FeatureState(name)
        self.period = period
        self._high_buffer = deque(maxlen=period)
        self._low_buffer = deque(maxlen=period)
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, float]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, high: Optional[float] = None, 
               low: Optional[float] = None, **kwargs) -> Optional[Dict[str, float]]:
        if high is None or low is None:
            raise ValueError("Aroon requires high and low prices")
        
        self._high_buffer.append(high)
        self._low_buffer.append(low)
        
        if len(self._high_buffer) == self.period:
            # Find the index of highest high and lowest low
            highest_high = max(self._high_buffer)
            lowest_low = min(self._low_buffer)
            
            # Find the most recent occurrence
            high_index = len(self._high_buffer) - 1 - list(reversed(self._high_buffer)).index(highest_high)
            low_index = len(self._low_buffer) - 1 - list(reversed(self._low_buffer)).index(lowest_low)
            
            # Calculate Aroon indicators
            aroon_up = ((self.period - (self.period - 1 - high_index)) / self.period) * 100
            aroon_down = ((self.period - (self.period - 1 - low_index)) / self.period) * 100
            aroon_oscillator = aroon_up - aroon_down
            
            self._state.set_value({
                "up": aroon_up,
                "down": aroon_down,
                "oscillator": aroon_oscillator
            })
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._high_buffer.clear()
        self._low_buffer.clear()


class Vortex:
    """Vortex Indicator with O(1) updates."""
    
    def __init__(self, period: int = 14, name: str = "vortex"):
        self._state = FeatureState(name)
        self.period = period
        self._prev_high: Optional[float] = None
        self._prev_low: Optional[float] = None
        self._prev_close: Optional[float] = None
        
        self._tr_buffer = deque(maxlen=period)
        self._vm_plus_buffer = deque(maxlen=period)
        self._vm_minus_buffer = deque(maxlen=period)
        
        self._tr_sum = 0.0
        self._vm_plus_sum = 0.0
        self._vm_minus_sum = 0.0
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, float]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, high: Optional[float] = None, 
               low: Optional[float] = None, **kwargs) -> Optional[Dict[str, float]]:
        if high is None or low is None:
            raise ValueError("Vortex requires high and low prices")
        
        if self._prev_high is not None and self._prev_low is not None and self._prev_close is not None:
            # Calculate True Range
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close)
            )
            
            # Calculate Vortex Movement
            vm_plus = abs(high - self._prev_low)
            vm_minus = abs(low - self._prev_high)
            
            # Update buffers
            if len(self._tr_buffer) == self.period:
                self._tr_sum -= self._tr_buffer[0]
                self._vm_plus_sum -= self._vm_plus_buffer[0]
                self._vm_minus_sum -= self._vm_minus_buffer[0]
            
            self._tr_buffer.append(tr)
            self._vm_plus_buffer.append(vm_plus)
            self._vm_minus_buffer.append(vm_minus)
            self._tr_sum += tr
            self._vm_plus_sum += vm_plus
            self._vm_minus_sum += vm_minus
            
            if len(self._tr_buffer) == self.period and self._tr_sum > 0:
                vi_plus = self._vm_plus_sum / self._tr_sum
                vi_minus = self._vm_minus_sum / self._tr_sum
                
                self._state.set_value({
                    "vi_plus": vi_plus,
                    "vi_minus": vi_minus
                })
        
        self._prev_high = high
        self._prev_low = low
        self._prev_close = price
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._prev_high = None
        self._prev_low = None
        self._prev_close = None
        self._tr_buffer.clear()
        self._vm_plus_buffer.clear()
        self._vm_minus_buffer.clear()
        self._tr_sum = 0.0
        self._vm_plus_sum = 0.0
        self._vm_minus_sum = 0.0


# Momentum feature registry for the FeatureHub factory
MOMENTUM_FEATURES = {
    "macd": MACD,
    "momentum": Momentum,
    "roc": ROC,
    "rate_of_change": ROC,  # Alias
    "adx": ADX,
    "aroon": Aroon,
    "vortex": Vortex,
}