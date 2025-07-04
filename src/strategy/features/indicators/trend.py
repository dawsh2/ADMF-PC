"""
Trend-based incremental features.

Features that analyze price trends, moving averages, and directional movement.
All features maintain state and update in O(1) time complexity.

Uses protocol + composition architecture - no inheritance.
"""

from typing import Optional, Dict, Any
from collections import deque
from ..protocols import Feature, FeatureState


class SMA:
    """Simple Moving Average with O(1) updates."""
    
    def __init__(self, period: int = 20, name: str = "sma"):
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
    
    def update(self, price: float, **kwargs) -> Optional[float]:
        if len(self._buffer) == self.period:
            self._sum -= self._buffer[0]
        
        self._buffer.append(price)
        self._sum += price
        
        if len(self._buffer) == self.period:
            self._state.set_value(self._sum / self.period)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._buffer.clear()
        self._sum = 0.0


class EMA:
    """Exponential Moving Average with O(1) updates."""
    
    def __init__(self, period: int = 20, name: str = "ema", smoothing: float = 2.0):
        self._state = FeatureState(name)
        self.period = period
        self.alpha = smoothing / (period + 1)
        self._count = 0
    
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
        self._count += 1
        
        if self._state.value is None:
            self._state.set_value(price)
        else:
            new_value = self.alpha * price + (1 - self.alpha) * self._state.value
            self._state.set_value(new_value)
        
        if self._count >= self.period:
            self._state._is_ready = True
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._count = 0


class DEMA:
    """Double Exponential Moving Average with O(1) updates."""
    
    def __init__(self, period: int = 20, name: str = "dema"):
        self._state = FeatureState(name)
        self.period = period
        self.ema1 = EMA(period, name=f"{name}_ema1")
        self.ema2 = EMA(period, name=f"{name}_ema2")
    
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
        ema1_val = self.ema1.update(price)
        
        if ema1_val is not None:
            ema2_val = self.ema2.update(ema1_val)
            
            if ema2_val is not None and self.ema1.is_ready and self.ema2.is_ready:
                dema_val = 2 * ema1_val - ema2_val
                self._state.set_value(dema_val)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self.ema1.reset()
        self.ema2.reset()


class TEMA:
    """Triple Exponential Moving Average with O(1) updates."""
    
    def __init__(self, period: int = 20, name: str = "tema"):
        self._state = FeatureState(name)
        self.period = period
        self.ema1 = EMA(period, name=f"{name}_ema1")
        self.ema2 = EMA(period, name=f"{name}_ema2")
        self.ema3 = EMA(period, name=f"{name}_ema3")
    
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
        ema1_val = self.ema1.update(price)
        
        if ema1_val is not None:
            ema2_val = self.ema2.update(ema1_val)
            
            if ema2_val is not None:
                ema3_val = self.ema3.update(ema2_val)
                
                if (ema3_val is not None and 
                    self.ema1.is_ready and self.ema2.is_ready and self.ema3.is_ready):
                    tema_val = 3 * ema1_val - 3 * ema2_val + ema3_val
                    self._state.set_value(tema_val)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self.ema1.reset()
        self.ema2.reset()
        self.ema3.reset()


class WMA:
    """Weighted Moving Average with O(1) updates."""
    
    def __init__(self, period: int = 20, name: str = "wma"):
        self._state = FeatureState(name)
        self.period = period
        self._buffer = deque(maxlen=period)
        self._weights = list(range(1, period + 1))
        self._weight_sum = sum(self._weights)
    
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
        self._buffer.append(price)
        
        if len(self._buffer) == self.period:
            weighted_sum = sum(p * w for p, w in zip(self._buffer, self._weights))
            self._state.set_value(weighted_sum / self._weight_sum)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._buffer.clear()


class HMA:
    """Hull Moving Average with O(1) updates."""
    
    def __init__(self, period: int = 20, name: str = "hma"):
        self._state = FeatureState(name)
        self.period = period
        self.half_period = period // 2
        self.sqrt_period = int(period ** 0.5)
        
        self.wma_half = WMA(self.half_period, name=f"{name}_wma_half")
        self.wma_full = WMA(period, name=f"{name}_wma_full")
        self.wma_sqrt = WMA(self.sqrt_period, name=f"{name}_wma_sqrt")
    
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
        wma_half_val = self.wma_half.update(price)
        wma_full_val = self.wma_full.update(price)
        
        if wma_half_val is not None and wma_full_val is not None:
            hull_raw = 2 * wma_half_val - wma_full_val
            hma_val = self.wma_sqrt.update(hull_raw)
            
            if hma_val is not None and self.wma_sqrt.is_ready:
                self._state.set_value(hma_val)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self.wma_half.reset()
        self.wma_full.reset()
        self.wma_sqrt.reset()


class VWMA:
    """Volume Weighted Moving Average with O(1) updates."""
    
    def __init__(self, period: int = 20, name: str = "vwma"):
        self._state = FeatureState(name)
        self.period = period
        self._price_buffer = deque(maxlen=period)
        self._volume_buffer = deque(maxlen=period)
        self._pv_sum = 0.0
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
    
    def update(self, price: float, volume: Optional[float] = None, **kwargs) -> Optional[float]:
        if volume is None:
            raise ValueError("VWMA requires volume data")
        
        if len(self._price_buffer) == self.period:
            old_price = self._price_buffer[0]
            old_volume = self._volume_buffer[0]
            self._pv_sum -= old_price * old_volume
            self._volume_sum -= old_volume
        
        self._price_buffer.append(price)
        self._volume_buffer.append(volume)
        self._pv_sum += price * volume
        self._volume_sum += volume
        
        if len(self._price_buffer) == self.period and self._volume_sum > 0:
            self._state.set_value(self._pv_sum / self._volume_sum)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._price_buffer.clear()
        self._volume_buffer.clear()
        self._pv_sum = 0.0
        self._volume_sum = 0.0


class IchimokuCloud:
    """Ichimoku Cloud with O(1) updates."""
    
    def __init__(self, conversion_period: int = 9, base_period: int = 26, 
                 lagging_span_period: int = 52, displacement: int = 26, name: str = "ichimoku"):
        self._state = FeatureState(name)
        self.conversion_period = conversion_period
        self.base_period = base_period
        self.lagging_span_period = lagging_span_period
        self.displacement = displacement
        
        # Buffers for high/low tracking
        self._high_buffer_conversion = deque(maxlen=conversion_period)
        self._low_buffer_conversion = deque(maxlen=conversion_period)
        self._high_buffer_base = deque(maxlen=base_period)
        self._low_buffer_base = deque(maxlen=base_period)
        self._high_buffer_lagging = deque(maxlen=lagging_span_period)
        self._low_buffer_lagging = deque(maxlen=lagging_span_period)
        
        # Close buffer for lagging span
        self._close_buffer = deque(maxlen=lagging_span_period)
        
        # Displacement buffers for future values
        self._senkou_span_a_buffer = deque(maxlen=displacement)
        self._senkou_span_b_buffer = deque(maxlen=displacement)
    
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
            raise ValueError("Ichimoku requires high and low prices")
        
        # Update all buffers
        self._high_buffer_conversion.append(high)
        self._low_buffer_conversion.append(low)
        self._high_buffer_base.append(high)
        self._low_buffer_base.append(low)
        self._high_buffer_lagging.append(high)
        self._low_buffer_lagging.append(low)
        self._close_buffer.append(price)
        
        result = {}
        
        # Conversion Line (Tenkan-sen)
        if len(self._high_buffer_conversion) == self.conversion_period:
            conversion_high = max(self._high_buffer_conversion)
            conversion_low = min(self._low_buffer_conversion)
            conversion_line = (conversion_high + conversion_low) / 2
            result["tenkan_sen"] = conversion_line
        
        # Base Line (Kijun-sen)
        if len(self._high_buffer_base) == self.base_period:
            base_high = max(self._high_buffer_base)
            base_low = min(self._low_buffer_base)
            base_line = (base_high + base_low) / 2
            result["kijun_sen"] = base_line
            
            # Senkou Span A (Leading Span A) - displaced into future
            if "tenkan_sen" in result:
                senkou_span_a = (result["tenkan_sen"] + base_line) / 2
                self._senkou_span_a_buffer.append(senkou_span_a)
                
                # Return displaced value if available
                if len(self._senkou_span_a_buffer) == self.displacement:
                    result["senkou_span_a"] = self._senkou_span_a_buffer[0]
        
        # Senkou Span B (Leading Span B)
        if len(self._high_buffer_lagging) == self.lagging_span_period:
            lagging_high = max(self._high_buffer_lagging)
            lagging_low = min(self._low_buffer_lagging)
            senkou_span_b = (lagging_high + lagging_low) / 2
            self._senkou_span_b_buffer.append(senkou_span_b)
            
            # Return displaced value if available
            if len(self._senkou_span_b_buffer) == self.displacement:
                result["senkou_span_b"] = self._senkou_span_b_buffer[0]
        
        # Chikou Span (Lagging Span) - current close displaced back
        if len(self._close_buffer) >= self.displacement:
            result["chikou_span"] = self._close_buffer[-self.displacement]
        
        if result:
            self._state.set_value(result)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._high_buffer_conversion.clear()
        self._low_buffer_conversion.clear()
        self._high_buffer_base.clear()
        self._low_buffer_base.clear()
        self._high_buffer_lagging.clear()
        self._low_buffer_lagging.clear()
        self._close_buffer.clear()
        self._senkou_span_a_buffer.clear()
        self._senkou_span_b_buffer.clear()


class ParabolicSAR:
    """
    Parabolic SAR (Stop and Reverse) with O(1) updates.
    
    The Parabolic SAR is a trend-following indicator that provides
    potential reversal points. It uses an acceleration factor that
    increases as the trend extends.
    """
    
    def __init__(self, af_start: float = 0.02, af_max: float = 0.2, name: str = "psar"):
        """
        Initialize Parabolic SAR.
        
        Args:
            af_start: Initial acceleration factor (default 0.02)
            af_max: Maximum acceleration factor (default 0.2)
            name: Feature name
        """
        self._state = FeatureState(name)
        self.af_start = af_start
        self.af_max = af_max
        self.af_increment = af_start
        
        # PSAR calculation state
        self.psar = None
        self.ep = None  # Extreme point
        self.af = af_start
        self.trend = None  # 1 for uptrend, -1 for downtrend
        self.high = None
        self.low = None
        
        # Previous bar values for trend detection
        self.prev_psar = None
        self.prev_high = None
        self.prev_low = None
        
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[float]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, high: float, low: float, close: float, **kwargs) -> Optional[float]:
        """
        Update PSAR with new price data.
        
        Args:
            high: High price
            low: Low price
            close: Close price (used for initialization)
            
        Returns:
            Current PSAR value
        """
        # First bar initialization
        if self.psar is None:
            self.psar = low
            self.ep = high
            self.trend = 1  # Start with uptrend
            self.high = high
            self.low = low
            self._state.set_value(self.psar)
            return self.psar
        
        # Store previous values
        self.prev_psar = self.psar
        self.prev_high = self.high
        self.prev_low = self.low
        self.high = high
        self.low = low
        
        # Calculate new PSAR
        if self.trend == 1:  # Uptrend
            # Update PSAR
            self.psar = self.psar + self.af * (self.ep - self.psar)
            
            # Make sure PSAR is not above the prior two lows
            if self.prev_low is not None:
                self.psar = min(self.psar, self.prev_low, self.low)
            
            # Check for new extreme point
            if high > self.ep:
                self.ep = high
                self.af = min(self.af + self.af_increment, self.af_max)
            
            # Check for trend reversal
            if low <= self.psar:
                self.trend = -1
                self.psar = self.ep
                self.ep = low
                self.af = self.af_start
                
        else:  # Downtrend
            # Update PSAR
            self.psar = self.psar + self.af * (self.ep - self.psar)
            
            # Make sure PSAR is not below the prior two highs
            if self.prev_high is not None:
                self.psar = max(self.psar, self.prev_high, self.high)
            
            # Check for new extreme point
            if low < self.ep:
                self.ep = low
                self.af = min(self.af + self.af_increment, self.af_max)
            
            # Check for trend reversal
            if high >= self.psar:
                self.trend = 1
                self.psar = self.ep
                self.ep = high
                self.af = self.af_start
        
        self._state.set_value(self.psar)
        return self.psar
    
    def reset(self) -> None:
        """Reset the indicator state."""
        self._state.reset()
        self.psar = None
        self.ep = None
        self.af = self.af_start
        self.trend = None
        self.high = None
        self.low = None
        self.prev_psar = None
        self.prev_high = None
        self.prev_low = None


# Trend feature registry for the FeatureHub factory
TREND_FEATURES = {
    "sma": SMA,
    "ema": EMA,
    "dema": DEMA,
    "tema": TEMA,
    "wma": WMA,
    "hma": HMA,
    "vwma": VWMA,
    "ichimoku": IchimokuCloud,
    "psar": ParabolicSAR,
    "parabolic_sar": ParabolicSAR,  # Alias
}