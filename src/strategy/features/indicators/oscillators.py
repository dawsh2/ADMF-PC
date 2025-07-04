"""
Oscillator-based incremental features.

Features that analyze momentum and bounded indicators that oscillate between fixed ranges.
All features maintain state and update in O(1) time complexity.

Uses protocol + composition architecture - no inheritance.
"""

import math
from typing import Optional, Dict, Any
from collections import deque
from ..protocols import Feature, FeatureState


class RSI:
    """Relative Strength Index with O(1) updates."""
    
    def __init__(self, period: int = 14, name: str = "rsi"):
        self._state = FeatureState(name)
        self.period = period
        self._prev_price: Optional[float] = None
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._count = 0
        self._gain_buffer = deque(maxlen=period)
        self._loss_buffer = deque(maxlen=period)
    
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
            change = price - self._prev_price
            gain = max(0, change)
            loss = max(0, -change)
            
            self._gain_buffer.append(gain)
            self._loss_buffer.append(loss)
            
            if len(self._gain_buffer) == self.period:
                if self._count == self.period:
                    self._avg_gain = sum(self._gain_buffer) / self.period
                    self._avg_loss = sum(self._loss_buffer) / self.period
                else:
                    self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
                    self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period
                
                if self._avg_loss == 0:
                    rsi_value = 100.0
                else:
                    rs = self._avg_gain / self._avg_loss
                    rsi_value = 100.0 - (100.0 / (1.0 + rs))
                
                self._state.set_value(rsi_value)
        
        self._prev_price = price
        self._count += 1
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._prev_price = None
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._count = 0
        self._gain_buffer.clear()
        self._loss_buffer.clear()


class StochasticOscillator:
    """Stochastic Oscillator (%K and %D) with O(1) updates."""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, name: str = "stochastic"):
        self._state = FeatureState(name)
        self.k_period = k_period
        self.d_period = d_period
        self._high_buffer = deque(maxlen=k_period)
        self._low_buffer = deque(maxlen=k_period)
        self._k_buffer = deque(maxlen=d_period)
        self._k_sum = 0.0
    
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
            raise ValueError("Stochastic requires high and low prices")
        
        self._high_buffer.append(high)
        self._low_buffer.append(low)
        
        if len(self._high_buffer) == self.k_period:
            highest_high = max(self._high_buffer)
            lowest_low = min(self._low_buffer)
            
            if highest_high != lowest_low:
                k_percent = 100 * (price - lowest_low) / (highest_high - lowest_low)
            else:
                k_percent = 50.0  # Neutral when no range
            
            # Calculate %D (SMA of %K)
            if len(self._k_buffer) == self.d_period:
                self._k_sum -= self._k_buffer[0]
            
            self._k_buffer.append(k_percent)
            self._k_sum += k_percent
            
            if len(self._k_buffer) == self.d_period:
                d_percent = self._k_sum / self.d_period
                self._state.set_value({
                    "k": k_percent,
                    "d": d_percent
                })
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._k_buffer.clear()
        self._k_sum = 0.0


class WilliamsR:
    """Williams %R with O(1) updates."""
    
    def __init__(self, period: int = 14, name: str = "williams_r"):
        self._state = FeatureState(name)
        self.period = period
        self._high_buffer = deque(maxlen=period)
        self._low_buffer = deque(maxlen=period)
    
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
            raise ValueError("Williams %R requires high and low prices")
        
        self._high_buffer.append(high)
        self._low_buffer.append(low)
        
        if len(self._high_buffer) == self.period:
            highest_high = max(self._high_buffer)
            lowest_low = min(self._low_buffer)
            
            if highest_high != lowest_low:
                williams_r = -100 * (highest_high - price) / (highest_high - lowest_low)
            else:
                williams_r = -50.0  # Neutral when no range
            
            self._state.set_value(williams_r)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._high_buffer.clear()
        self._low_buffer.clear()


class CCI:
    """Commodity Channel Index with O(1) updates."""
    
    def __init__(self, period: int = 20, name: str = "cci"):
        self._state = FeatureState(name)
        self.period = period
        self._tp_buffer = deque(maxlen=period)
        self._tp_sum = 0.0
    
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
            raise ValueError("CCI requires high and low prices")
        
        typical_price = (high + low + price) / 3
        
        if len(self._tp_buffer) == self.period:
            self._tp_sum -= self._tp_buffer[0]
        
        self._tp_buffer.append(typical_price)
        self._tp_sum += typical_price
        
        if len(self._tp_buffer) == self.period:
            sma_tp = self._tp_sum / self.period
            
            # Calculate mean deviation
            mean_deviation = sum(abs(tp - sma_tp) for tp in self._tp_buffer) / self.period
            
            if mean_deviation > 0:
                cci_value = (typical_price - sma_tp) / (0.015 * mean_deviation)
                self._state.set_value(cci_value)
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._tp_buffer.clear()
        self._tp_sum = 0.0


class StochasticRSI:
    """Stochastic RSI with O(1) updates."""
    
    def __init__(self, rsi_period: int = 14, stoch_period: int = 14, 
                 d_period: int = 3, name: str = "stochastic_rsi"):
        self._state = FeatureState(name)
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.d_period = d_period
        
        self.rsi = RSI(rsi_period, name=f"{name}_rsi")
        self._rsi_buffer = deque(maxlen=stoch_period)
        self._k_buffer = deque(maxlen=d_period)
        self._k_sum = 0.0
    
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
        rsi_value = self.rsi.update(price)
        
        if rsi_value is not None:
            self._rsi_buffer.append(rsi_value)
            
            if len(self._rsi_buffer) == self.stoch_period:
                highest_rsi = max(self._rsi_buffer)
                lowest_rsi = min(self._rsi_buffer)
                
                if highest_rsi != lowest_rsi:
                    stoch_rsi_k = 100 * (rsi_value - lowest_rsi) / (highest_rsi - lowest_rsi)
                else:
                    stoch_rsi_k = 50.0  # Neutral when no range
                
                # Calculate %D (SMA of %K)
                if len(self._k_buffer) == self.d_period:
                    self._k_sum -= self._k_buffer[0]
                
                self._k_buffer.append(stoch_rsi_k)
                self._k_sum += stoch_rsi_k
                
                if len(self._k_buffer) == self.d_period:
                    stoch_rsi_d = self._k_sum / self.d_period
                    self._state.set_value({
                        "k": stoch_rsi_k,
                        "d": stoch_rsi_d
                    })
        
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self.rsi.reset()
        self._rsi_buffer.clear()
        self._k_buffer.clear()
        self._k_sum = 0.0


class MFI:
    """Money Flow Index with O(1) updates."""
    
    def __init__(self, period: int = 14, name: str = "mfi"):
        self._state = FeatureState(name)
        self.period = period
        self._prev_typical_price: Optional[float] = None
        self._positive_mf_buffer = deque(maxlen=period)
        self._negative_mf_buffer = deque(maxlen=period)
        self._positive_mf_sum = 0.0
        self._negative_mf_sum = 0.0
    
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
            raise ValueError("MFI requires high, low, and volume prices")
        
        typical_price = (high + low + price) / 3
        money_flow = typical_price * volume
        
        if self._prev_typical_price is not None:
            if typical_price > self._prev_typical_price:
                positive_mf = money_flow
                negative_mf = 0.0
            elif typical_price < self._prev_typical_price:
                positive_mf = 0.0
                negative_mf = money_flow
            else:
                positive_mf = 0.0
                negative_mf = 0.0
            
            # Update buffers
            if len(self._positive_mf_buffer) == self.period:
                self._positive_mf_sum -= self._positive_mf_buffer[0]
                self._negative_mf_sum -= self._negative_mf_buffer[0]
            
            self._positive_mf_buffer.append(positive_mf)
            self._negative_mf_buffer.append(negative_mf)
            self._positive_mf_sum += positive_mf
            self._negative_mf_sum += negative_mf
            
            if len(self._positive_mf_buffer) == self.period:
                if self._negative_mf_sum == 0:
                    mfi_value = 100.0
                else:
                    money_ratio = self._positive_mf_sum / self._negative_mf_sum
                    mfi_value = 100 - (100 / (1 + money_ratio))
                
                self._state.set_value(mfi_value)
        
        self._prev_typical_price = typical_price
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._prev_typical_price = None
        self._positive_mf_buffer.clear()
        self._negative_mf_buffer.clear()
        self._positive_mf_sum = 0.0
        self._negative_mf_sum = 0.0


class UltimateOscillator:
    """Ultimate Oscillator with O(1) updates."""
    
    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28, name: str = "ultimate_oscillator"):
        self._state = FeatureState(name)
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        
        # Buffers for each period
        self._tr_buffer1 = deque(maxlen=period1)
        self._tr_buffer2 = deque(maxlen=period2)
        self._tr_buffer3 = deque(maxlen=period3)
        
        self._bp_buffer1 = deque(maxlen=period1)
        self._bp_buffer2 = deque(maxlen=period2)
        self._bp_buffer3 = deque(maxlen=period3)
        
        # Running sums for O(1) updates
        self._tr_sum1 = 0.0
        self._tr_sum2 = 0.0
        self._tr_sum3 = 0.0
        
        self._bp_sum1 = 0.0
        self._bp_sum2 = 0.0
        self._bp_sum3 = 0.0
        
        self._prev_close: Optional[float] = None
    
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
            raise ValueError("Ultimate Oscillator requires high and low prices")
        
        if self._prev_close is not None:
            # Calculate True Range and Buying Pressure
            true_range = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close)
            )
            
            buying_pressure = price - min(low, self._prev_close)
            
            # Update period 1
            if len(self._tr_buffer1) == self.period1:
                self._tr_sum1 -= self._tr_buffer1[0]
                self._bp_sum1 -= self._bp_buffer1[0]
            
            self._tr_buffer1.append(true_range)
            self._bp_buffer1.append(buying_pressure)
            self._tr_sum1 += true_range
            self._bp_sum1 += buying_pressure
            
            # Update period 2
            if len(self._tr_buffer2) == self.period2:
                self._tr_sum2 -= self._tr_buffer2[0]
                self._bp_sum2 -= self._bp_buffer2[0]
            
            self._tr_buffer2.append(true_range)
            self._bp_buffer2.append(buying_pressure)
            self._tr_sum2 += true_range
            self._bp_sum2 += buying_pressure
            
            # Update period 3
            if len(self._tr_buffer3) == self.period3:
                self._tr_sum3 -= self._tr_buffer3[0]
                self._bp_sum3 -= self._bp_buffer3[0]
            
            self._tr_buffer3.append(true_range)
            self._bp_buffer3.append(buying_pressure)
            self._tr_sum3 += true_range
            self._bp_sum3 += buying_pressure
            
            # Calculate Ultimate Oscillator when all periods are ready
            if (len(self._tr_buffer1) == self.period1 and 
                len(self._tr_buffer2) == self.period2 and 
                len(self._tr_buffer3) == self.period3):
                
                avg1 = (self._bp_sum1 / self._tr_sum1) if self._tr_sum1 > 0 else 0
                avg2 = (self._bp_sum2 / self._tr_sum2) if self._tr_sum2 > 0 else 0
                avg3 = (self._bp_sum3 / self._tr_sum3) if self._tr_sum3 > 0 else 0
                
                # Ultimate Oscillator formula with weighted averages
                uo_value = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
                self._state.set_value(uo_value)
        
        self._prev_close = price
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._tr_buffer1.clear()
        self._tr_buffer2.clear()
        self._tr_buffer3.clear()
        self._bp_buffer1.clear()
        self._bp_buffer2.clear()
        self._bp_buffer3.clear()
        
        self._tr_sum1 = 0.0
        self._tr_sum2 = 0.0
        self._tr_sum3 = 0.0
        self._bp_sum1 = 0.0
        self._bp_sum2 = 0.0
        self._bp_sum3 = 0.0
        
        self._prev_close = None


# Oscillator feature registry for the FeatureHub factory
OSCILLATOR_FEATURES = {
    "rsi": RSI,
    "stochastic": StochasticOscillator,
    "williams_r": WilliamsR,
    "cci": CCI,
    "stochastic_rsi": StochasticRSI,
    "mfi": MFI,
    "ultimate_oscillator": UltimateOscillator,
}