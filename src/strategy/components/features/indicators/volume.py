"""
Volume-based incremental features.

Features that analyze volume and price-volume relationships.
All features maintain state and update in O(1) time complexity.

Uses protocol + composition architecture - no inheritance.
"""

from typing import Optional, Dict, Any
from collections import deque
from ..protocols import Feature, FeatureState
from .trend import SMA


class Volume:
    """Raw volume feature - returns the volume from the bar data."""
    
    def __init__(self, name: str = "volume"):
        self._state = FeatureState(name)
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[float]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, volume: Optional[float] = None, **kwargs) -> Optional[float]:
        if volume is not None:
            self._state.set_value(volume)
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()


class VolumeSMA:
    """Volume Simple Moving Average with O(1) updates."""
    
    def __init__(self, period: int = 20, name: str = "volume_sma"):
        self._state = FeatureState(name)
        self.period = period
        self._buffer = deque(maxlen=period)
        self._sum = 0.0
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[float]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, volume: Optional[float] = None, **kwargs) -> Optional[float]:
        if volume is None:
            raise ValueError("Volume SMA requires volume data")
        
        if len(self._buffer) == self.period:
            self._sum -= self._buffer[0]
        
        self._buffer.append(volume)
        self._sum += volume
        
        if len(self._buffer) == self.period:
            self._state.set_value(self._sum / self.period)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._buffer.clear()
        self._sum = 0.0


class VolumeRatio:
    """Volume Ratio (current volume / average volume) with O(1) updates."""
    
    def __init__(self, period: int = 20, name: str = "volume_ratio"):
        self._state = FeatureState(name)
        self.period = period
        self.volume_sma = VolumeSMA(period, name=f"{name}_sma")
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[float]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, volume: Optional[float] = None, **kwargs) -> Optional[float]:
        if volume is None:
            raise ValueError("Volume Ratio requires volume data")
        
        volume_avg = self.volume_sma.update(price, volume=volume)
        
        if volume_avg is not None and volume_avg > 0:
            ratio = volume / volume_avg
            self._state.set_value(ratio)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self.volume_sma.reset()


class OBV:
    """On Balance Volume with O(1) updates."""
    
    def __init__(self, name: str = "obv"):
        self._state = FeatureState(name)
        self._prev_price: Optional[float] = None
        self._obv_value = 0.0
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[float]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, volume: Optional[float] = None, **kwargs) -> Optional[float]:
        if volume is None:
            raise ValueError("OBV requires volume data")
        
        if self._prev_price is not None:
            if price > self._prev_price:
                self._obv_value += volume
            elif price < self._prev_price:
                self._obv_value -= volume
            # If price == prev_price, OBV stays the same
            
            self._state.set_value(self._obv_value)
        else:
            # First data point
            self._obv_value = volume
            self._state.set_value(self._obv_value)
        
        self._prev_price = price
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._prev_price = None
        self._obv_value = 0.0


class VPT:
    """Volume Price Trend with O(1) updates."""
    
    def __init__(self, name: str = "vpt"):
        self._state = FeatureState(name)
        self._prev_price: Optional[float] = None
        self._vpt_value = 0.0
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[float]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, volume: Optional[float] = None, **kwargs) -> Optional[float]:
        if volume is None:
            raise ValueError("VPT requires volume data")
        
        if self._prev_price is not None and self._prev_price > 0:
            price_change_pct = (price - self._prev_price) / self._prev_price
            self._vpt_value += volume * price_change_pct
            self._state.set_value(self._vpt_value)
        
        self._prev_price = price
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._prev_price = None
        self._vpt_value = 0.0


class ChaikinMoneyFlow:
    """Chaikin Money Flow with O(1) updates."""
    
    def __init__(self, period: int = 20, name: str = "cmf"):
        self._state = FeatureState(name)
        self.period = period
        self._mfv_buffer = deque(maxlen=period)
        self._volume_buffer = deque(maxlen=period)
        self._mfv_sum = 0.0
        self._volume_sum = 0.0
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[float]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, high: Optional[float] = None, 
               low: Optional[float] = None, volume: Optional[float] = None, **kwargs) -> Optional[float]:
        if high is None or low is None or volume is None:
            raise ValueError("Chaikin Money Flow requires high, low, and volume data")
        
        # Calculate Money Flow Multiplier
        if high != low:
            mf_multiplier = ((price - low) - (high - price)) / (high - low)
        else:
            mf_multiplier = 0.0
        
        # Calculate Money Flow Volume
        mf_volume = mf_multiplier * volume
        
        # Update buffers
        if len(self._mfv_buffer) == self.period:
            self._mfv_sum -= self._mfv_buffer[0]
            self._volume_sum -= self._volume_buffer[0]
        
        self._mfv_buffer.append(mf_volume)
        self._volume_buffer.append(volume)
        self._mfv_sum += mf_volume
        self._volume_sum += volume
        
        if len(self._mfv_buffer) == self.period and self._volume_sum > 0:
            cmf = self._mfv_sum / self._volume_sum
            self._state.set_value(cmf)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._mfv_buffer.clear()
        self._volume_buffer.clear()
        self._mfv_sum = 0.0
        self._volume_sum = 0.0


class AccDistLine:
    """Accumulation/Distribution Line with O(1) updates."""
    
    def __init__(self, name: str = "ad_line"):
        self._state = FeatureState(name)
        self._ad_value = 0.0
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[float]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, high: Optional[float] = None, 
               low: Optional[float] = None, volume: Optional[float] = None, **kwargs) -> Optional[float]:
        if high is None or low is None or volume is None:
            raise ValueError("A/D Line requires high, low, and volume data")
        
        # Calculate Money Flow Multiplier
        if high != low:
            mf_multiplier = ((price - low) - (high - price)) / (high - low)
        else:
            mf_multiplier = 0.0
        
        # Calculate Money Flow Volume
        mf_volume = mf_multiplier * volume
        
        # Add to accumulation/distribution line
        self._ad_value += mf_volume
        self._state.set_value(self._ad_value)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._ad_value = 0.0


class VROC:
    """Volume Rate of Change with O(1) updates."""
    
    def __init__(self, period: int = 12, name: str = "vroc"):
        self._state = FeatureState(name)
        self.period = period
        self._volume_buffer = deque(maxlen=period + 1)
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[float]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, volume: Optional[float] = None, **kwargs) -> Optional[float]:
        if volume is None:
            raise ValueError("VROC requires volume data")
        
        self._volume_buffer.append(volume)
        
        if len(self._volume_buffer) == self.period + 1:
            old_volume = self._volume_buffer[0]
            if old_volume > 0:
                vroc = ((volume - old_volume) / old_volume) * 100
                self._state.set_value(vroc)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._volume_buffer.clear()


class VolumeMomentum:
    """Volume Momentum with O(1) updates."""
    
    def __init__(self, period: int = 14, name: str = "volume_momentum"):
        self._state = FeatureState(name)
        self.period = period
        self._volume_buffer = deque(maxlen=period + 1)
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[float]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, volume: Optional[float] = None, **kwargs) -> Optional[float]:
        if volume is None:
            raise ValueError("Volume Momentum requires volume data")
        
        self._volume_buffer.append(volume)
        
        if len(self._volume_buffer) == self.period + 1:
            old_volume = self._volume_buffer[0]
            momentum = volume - old_volume
            self._state.set_value(momentum)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._volume_buffer.clear()


# Volume feature registry for the FeatureHub factory
VOLUME_FEATURES = {
    "volume": Volume,  # Raw volume from bar data
    "volume_sma": VolumeSMA,
    "volume_ratio": VolumeRatio,
    "obv": OBV,
    "vpt": VPT,
    "cmf": ChaikinMoneyFlow,
    "chaikin_money_flow": ChaikinMoneyFlow,  # Alias
    "ad": AccDistLine,  # Short alias for accumulation_distribution strategy
    "ad_line": AccDistLine,
    "accumulation_distribution": AccDistLine,  # Alias
    "vroc": VROC,
    "volume_roc": VROC,  # Alias
    "volume_momentum": VolumeMomentum,
}