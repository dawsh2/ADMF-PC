"""
Volatility-based incremental features.

Features that measure price volatility, channels, and range breakouts.
All features maintain state and update in O(1) time complexity.

Uses protocol + composition architecture - no inheritance.
"""

import math
from typing import Optional, Dict, Any
from collections import deque
from ..protocols import Feature, FeatureState
from .trend import EMA


class ATR:
    """Average True Range with O(1) updates."""
    
    def __init__(self, period: int = 14, name: str = "atr"):
        self._state = FeatureState(name)
        self.period = period
        self._prev_close: Optional[float] = None
        self._tr_ema = EMA(period, name=f"{name}_tr_ema")
    
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
               low: Optional[float] = None, **kwargs) -> Optional[float]:
        if high is None or low is None:
            raise ValueError("ATR requires high and low prices")
        
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close)
            )
        
        self._tr_ema.update(tr)
        
        if self._tr_ema.is_ready:
            self._state.set_value(self._tr_ema.value)
        
        self._prev_close = price
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._prev_close = None
        self._tr_ema.reset()


class BollingerBands:
    """Bollinger Bands with O(1) updates."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, name: str = "bollinger"):
        self._state = FeatureState(name)
        self.period = period
        self.std_dev = std_dev
        self._buffer = deque(maxlen=period)
        self._sum = 0.0
        self._sum_sq = 0.0
    
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
        if len(self._buffer) == self.period:
            old_val = self._buffer[0]
            self._sum -= old_val
            self._sum_sq -= old_val * old_val
        
        self._buffer.append(price)
        self._sum += price
        self._sum_sq += price * price
        
        if len(self._buffer) == self.period:
            mean = self._sum / self.period
            variance = (self._sum_sq / self.period) - (mean * mean)
            std = math.sqrt(max(0, variance))
            
            self._state.set_value({
                "middle": mean,
                "upper": mean + (self.std_dev * std),
                "lower": mean - (self.std_dev * std)
            })
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._buffer.clear()
        self._sum = 0.0
        self._sum_sq = 0.0


class KeltnerChannel:
    """Keltner Channel with O(1) updates."""
    
    def __init__(self, period: int = 20, multiplier: float = 2.0, name: str = "keltner"):
        self._state = FeatureState(name)
        self.period = period
        self.multiplier = multiplier
        self.ema = EMA(period, name=f"{name}_ema")
        self.atr = ATR(period, name=f"{name}_atr")
    
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
            raise ValueError("Keltner Channel requires high and low prices")
        
        ema_val = self.ema.update(price)
        atr_val = self.atr.update(price, high=high, low=low)
        
        if ema_val is not None and atr_val is not None and self.ema.is_ready and self.atr.is_ready:
            self._state.set_value({
                "middle": ema_val,
                "upper": ema_val + (self.multiplier * atr_val),
                "lower": ema_val - (self.multiplier * atr_val)
            })
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self.ema.reset()
        self.atr.reset()


class DonchianChannel:
    """Donchian Channel with O(1) updates."""
    
    def __init__(self, period: int = 20, name: str = "donchian"):
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
            raise ValueError("Donchian Channel requires high and low prices")
        
        self._high_buffer.append(high)
        self._low_buffer.append(low)
        
        if len(self._high_buffer) == self.period:
            upper = max(self._high_buffer)
            lower = min(self._low_buffer)
            middle = (upper + lower) / 2
            
            self._state.set_value({
                "upper": upper,
                "lower": lower,
                "middle": middle
            })
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._high_buffer.clear()
        self._low_buffer.clear()


class Volatility:
    """Price Volatility (Standard Deviation of Returns) with O(1) updates."""
    
    def __init__(self, period: int = 20, name: str = "volatility"):
        self._state = FeatureState(name)
        self.period = period
        self._prev_price: Optional[float] = None
        self._returns_buffer = deque(maxlen=period)
        self._returns_sum = 0.0
        self._returns_sum_sq = 0.0
    
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
        if self._prev_price is not None:
            returns = (price - self._prev_price) / self._prev_price
            
            if len(self._returns_buffer) == self.period:
                old_return = self._returns_buffer[0]
                self._returns_sum -= old_return
                self._returns_sum_sq -= old_return * old_return
            
            self._returns_buffer.append(returns)
            self._returns_sum += returns
            self._returns_sum_sq += returns * returns
            
            if len(self._returns_buffer) == self.period:
                mean_return = self._returns_sum / self.period
                variance = (self._returns_sum_sq / self.period) - (mean_return * mean_return)
                volatility = math.sqrt(max(0, variance))
                self._state.set_value(volatility)
        
        self._prev_price = price
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._prev_price = None
        self._returns_buffer.clear()
        self._returns_sum = 0.0
        self._returns_sum_sq = 0.0


class SuperTrend:
    """SuperTrend with O(1) updates."""
    
    def __init__(self, period: int = 10, multiplier: float = 3.0, name: str = "supertrend"):
        self._state = FeatureState(name)
        self.period = period
        self.multiplier = multiplier
        self.atr = ATR(period, name=f"{name}_atr")
        self._prev_supertrend: Optional[float] = None
        self._prev_trend: int = 1  # 1 for uptrend, -1 for downtrend
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, Any]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, high: Optional[float] = None, 
               low: Optional[float] = None, **kwargs) -> Optional[Dict[str, Any]]:
        if high is None or low is None:
            raise ValueError("SuperTrend requires high and low prices")
        
        atr_val = self.atr.update(price, high=high, low=low)
        
        if atr_val is not None and self.atr.is_ready:
            hl2 = (high + low) / 2
            
            # Calculate basic upper and lower bands
            basic_upper = hl2 + (self.multiplier * atr_val)
            basic_lower = hl2 - (self.multiplier * atr_val)
            
            # Calculate final bands
            if self._prev_supertrend is not None:
                # Upper band
                if basic_upper < self._state.value.get("upper", basic_upper) or price > self._state.value.get("upper", basic_upper):
                    final_upper = basic_upper
                else:
                    final_upper = self._state.value.get("upper", basic_upper)
                
                # Lower band
                if basic_lower > self._state.value.get("lower", basic_lower) or price < self._state.value.get("lower", basic_lower):
                    final_lower = basic_lower
                else:
                    final_lower = self._state.value.get("lower", basic_lower)
            else:
                final_upper = basic_upper
                final_lower = basic_lower
            
            # Determine trend
            if price <= final_lower:
                trend = -1
                supertrend = final_upper
            elif price >= final_upper:
                trend = 1
                supertrend = final_lower
            else:
                trend = self._prev_trend
                supertrend = final_upper if trend == -1 else final_lower
            
            self._state.set_value({
                "supertrend": supertrend,
                "trend": trend,
                "upper": final_upper,
                "lower": final_lower
            })
            
            self._prev_supertrend = supertrend
            self._prev_trend = trend
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self.atr.reset()
        self._prev_supertrend = None
        self._prev_trend = 1


class VWAP:
    """Volume Weighted Average Price with O(1) updates."""
    
    def __init__(self, name: str = "vwap"):
        self._state = FeatureState(name)
        self._cumulative_pv = 0.0
        self._cumulative_volume = 0.0
    
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
        
        if volume > 0:
            typical_price = price  # Can be extended to use HLC/3 if high/low available
            self._cumulative_pv += typical_price * volume
            self._cumulative_volume += volume
            
            if self._cumulative_volume > 0:
                vwap_value = self._cumulative_pv / self._cumulative_volume
                self._state.set_value(vwap_value)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._cumulative_pv = 0.0
        self._cumulative_volume = 0.0


# Volatility feature registry for the FeatureHub factory
VOLATILITY_FEATURES = {
    "atr": ATR,
    "bollinger": BollingerBands,
    "bollinger_bands": BollingerBands,  # Alias
    "bb": BollingerBands,  # Alias
    "keltner": KeltnerChannel,
    "keltner_channel": KeltnerChannel,  # Alias
    "donchian": DonchianChannel,
    "donchian_channel": DonchianChannel,  # Alias
    "volatility": Volatility,
    "volatility_percentile": "volatility_percentile",  # Lazy import via component registry
    "supertrend": SuperTrend,
    "vwap": VWAP,
}