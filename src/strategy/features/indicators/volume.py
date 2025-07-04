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


class VWAP:
    """Volume Weighted Average Price (VWAP) with O(1) updates.
    
    Standard VWAP that accumulates throughout the entire data series.
    For session-based VWAP, use SessionVWAP instead.
    """
    
    def __init__(self, name: str = "vwap"):
        self._state = FeatureState(name)
        self._cumulative_pv = 0.0  # Cumulative price * volume
        self._cumulative_volume = 0.0  # Cumulative volume
    
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
            raise ValueError("VWAP requires volume data")
        
        # Use typical price if high/low available
        high = kwargs.get('high', price)
        low = kwargs.get('low', price)
        typical_price = (high + low + price) / 3
        
        # Accumulate price * volume and volume
        self._cumulative_pv += typical_price * volume
        self._cumulative_volume += volume
        
        if self._cumulative_volume > 0:
            vwap = self._cumulative_pv / self._cumulative_volume
            self._state.set_value(vwap)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._cumulative_pv = 0.0
        self._cumulative_volume = 0.0


class SessionVWAP:
    """Session-based Volume Weighted Average Price with O(1) updates.
    
    Resets at the beginning of each trading session (market open).
    Detects session boundaries based on timestamp gaps or explicit session times.
    """
    
    def __init__(self, session_start_hour: int = 9, session_start_minute: int = 30,
                 reset_on_gap_minutes: int = 60, name: str = "session_vwap"):
        """
        Initialize SessionVWAP.
        
        Args:
            session_start_hour: Hour when trading session starts (default: 9 for 9:30 AM)
            session_start_minute: Minute when trading session starts (default: 30)
            reset_on_gap_minutes: Reset if time gap exceeds this many minutes (default: 60)
            name: Feature name
        """
        self._state = FeatureState(name)
        self.session_start_hour = session_start_hour
        self.session_start_minute = session_start_minute
        self.reset_on_gap_minutes = reset_on_gap_minutes
        
        self._cumulative_pv = 0.0  # Cumulative price * volume for current session
        self._cumulative_volume = 0.0  # Cumulative volume for current session
        self._last_timestamp = None
        self._current_session_date = None
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[float]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, volume: Optional[float] = None, 
               timestamp: Optional[Any] = None, **kwargs) -> Optional[float]:
        if volume is None:
            raise ValueError("SessionVWAP requires volume data")
        
        # Check if we need to reset for a new session
        if timestamp is not None:
            if self._should_reset_session(timestamp):
                self._reset_session()
        
        # Use typical price if high/low available
        high = kwargs.get('high', price)
        low = kwargs.get('low', price)
        typical_price = (high + low + price) / 3
        
        # Accumulate price * volume and volume for current session
        self._cumulative_pv += typical_price * volume
        self._cumulative_volume += volume
        
        if self._cumulative_volume > 0:
            vwap = self._cumulative_pv / self._cumulative_volume
            self._state.set_value(vwap)
        
        self._last_timestamp = timestamp
        return self._state.value
    
    def _should_reset_session(self, timestamp) -> bool:
        """Determine if we should reset for a new trading session."""
        if self._last_timestamp is None:
            return False
        
        # Handle different timestamp formats
        import datetime
        import pandas as pd
        
        # Convert to datetime if needed
        if isinstance(timestamp, str):
            current_dt = pd.to_datetime(timestamp)
        elif isinstance(timestamp, (int, float)):
            current_dt = pd.to_datetime(timestamp, unit='s')
        else:
            current_dt = timestamp
        
        if isinstance(self._last_timestamp, str):
            last_dt = pd.to_datetime(self._last_timestamp)
        elif isinstance(self._last_timestamp, (int, float)):
            last_dt = pd.to_datetime(self._last_timestamp, unit='s')
        else:
            last_dt = self._last_timestamp
        
        # Check for new trading day
        if current_dt.date() != last_dt.date():
            return True
        
        # Check if we're at session start time
        if (current_dt.hour == self.session_start_hour and 
            current_dt.minute == self.session_start_minute and
            last_dt.hour < self.session_start_hour):
            return True
        
        # Check for large time gaps (e.g., between sessions)
        time_diff = (current_dt - last_dt).total_seconds() / 60  # Minutes
        if time_diff > self.reset_on_gap_minutes:
            return True
        
        return False
    
    def _reset_session(self) -> None:
        """Reset accumulations for a new session."""
        self._cumulative_pv = 0.0
        self._cumulative_volume = 0.0
        # Don't reset the state value - let it update with new session data
    
    def reset(self) -> None:
        """Full reset of the indicator."""
        self._state.reset()
        self._cumulative_pv = 0.0
        self._cumulative_volume = 0.0
        self._last_timestamp = None
        self._current_session_date = None


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
    "vwap": VWAP,
    "session_vwap": SessionVWAP,
}