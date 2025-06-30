"""
Proper Bollinger Band + RSI Divergence Feature.

This feature computes BB and RSI internally and tracks divergences,
maintaining all necessary state.
"""

from typing import Optional, Dict, Any, Tuple, List, Deque
from collections import deque
import math
from ..protocols import Feature, FeatureState


class BollingerRSIDivergenceProper:
    """
    Tracks RSI divergences at Bollinger Band extremes - PROPER implementation.
    
    This computes BB and RSI internally from price data and tracks the
    multi-bar divergence pattern from the profitable backtest.
    """
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14,
                 lookback: int = 20, rsi_divergence_threshold: float = 5.0,
                 confirmation_bars: int = 10, name: str = "bb_rsi_divergence_proper"):
        self._state = FeatureState(name)
        
        # Parameters
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.lookback = lookback
        self.rsi_divergence_threshold = rsi_divergence_threshold
        self.confirmation_bars = confirmation_bars
        
        # Price buffers for indicator calculation
        self._price_buffer = deque(maxlen=max(bb_period, rsi_period + 1))
        
        # RSI calculation state
        self._gains = deque(maxlen=rsi_period)
        self._losses = deque(maxlen=rsi_period)
        self._prev_price = None
        self._rsi_initialized = False
        
        # Track extremes when price is outside bands
        self.potential_longs: Dict[int, Tuple[float, float]] = {}  # bar_idx: (low_price, rsi_value)
        self.potential_shorts: Dict[int, Tuple[float, float]] = {}  # bar_idx: (high_price, rsi_value)
        
        # Current bar index
        self._bar_index = 0
        
        # Track if we're currently outside bands
        self._was_below_lower = False
        self._was_above_upper = False
        
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, Any]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def _calculate_rsi(self, price: float) -> Optional[float]:
        """Calculate RSI using incremental updates."""
        if self._prev_price is None:
            self._prev_price = price
            return None
            
        change = price - self._prev_price
        gain = max(change, 0)
        loss = max(-change, 0)
        
        self._gains.append(gain)
        self._losses.append(loss)
        
        if len(self._gains) < self.rsi_period:
            self._prev_price = price
            return None
            
        avg_gain = sum(self._gains) / self.rsi_period
        avg_loss = sum(self._losses) / self.rsi_period
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
        self._prev_price = price
        return rsi
    
    def _calculate_bollinger_bands(self) -> Optional[Tuple[float, float, float]]:
        """Calculate Bollinger Bands from price buffer."""
        if len(self._price_buffer) < self.bb_period:
            return None
            
        # Calculate SMA (middle band)
        sma = sum(self._price_buffer) / self.bb_period
        
        # Calculate standard deviation
        variance = sum((p - sma) ** 2 for p in self._price_buffer) / self.bb_period
        std_dev = math.sqrt(variance)
        
        # Calculate bands
        upper_band = sma + (self.bb_std * std_dev)
        lower_band = sma - (self.bb_std * std_dev)
        
        return upper_band, sma, lower_band
    
    def update(self, price: float, high: Optional[float] = None, 
               low: Optional[float] = None, volume: Optional[float] = None,
               **kwargs) -> Optional[Dict[str, Any]]:
        """
        Update divergence tracking with new bar data.
        """
        if high is None or low is None:
            return None
            
        self._bar_index += 1
        self._price_buffer.append(price)
        
        # Calculate indicators
        rsi = self._calculate_rsi(price)
        bb_result = self._calculate_bollinger_bands()
        
        if rsi is None or bb_result is None:
            return None
            
        upper_band, middle_band, lower_band = bb_result
        
        # Clean old potential signals beyond lookback
        self.potential_longs = {idx: val for idx, val in self.potential_longs.items() 
                               if self._bar_index - idx <= self.lookback}
        self.potential_shorts = {idx: val for idx, val in self.potential_shorts.items() 
                                if self._bar_index - idx <= self.lookback}
        
        # Track new extremes when price is outside bands
        if price < lower_band:
            self.potential_longs[self._bar_index] = (low, rsi)
            self._was_below_lower = True
        elif price > upper_band:
            self.potential_shorts[self._bar_index] = (high, rsi)
            self._was_above_upper = True
        
        # Initialize result
        result = {
            'confirmed_long': False,
            'confirmed_short': False,
            'has_bullish_divergence': False,
            'has_bearish_divergence': False,
            'divergence_strength': 0.0,
            'bars_since_divergence': None,
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band,
            'rsi': rsi
        }
        
        # Look for divergence and confirmation (Long)
        if price > lower_band and self._was_below_lower:  # Price back inside bands
            # Look through all potential long setups
            for prev_idx, (prev_low, prev_rsi) in self.potential_longs.items():
                if prev_idx < self._bar_index - 1:  # Not same or adjacent bar
                    # Check all bars between prev and now for lower low + higher RSI
                    for recent_idx in range(max(self._bar_index - self.confirmation_bars, prev_idx + 1), self._bar_index):
                        if recent_idx in self.potential_longs:
                            recent_low, recent_rsi = self.potential_longs[recent_idx]
                            
                            # Bullish divergence: lower low in price, higher RSI
                            if (recent_low < prev_low and 
                                recent_rsi > prev_rsi + self.rsi_divergence_threshold):
                                
                                result['has_bullish_divergence'] = True
                                result['confirmed_long'] = True
                                result['divergence_strength'] = recent_rsi - prev_rsi
                                result['bars_since_divergence'] = self._bar_index - recent_idx
                                break
                
                if result['confirmed_long']:
                    break
            
            self._was_below_lower = False
        
        # Look for divergence and confirmation (Short)
        if price < upper_band and self._was_above_upper:  # Price back inside bands
            # Look through all potential short setups
            for prev_idx, (prev_high, prev_rsi) in self.potential_shorts.items():
                if prev_idx < self._bar_index - 1:
                    for recent_idx in range(max(self._bar_index - self.confirmation_bars, prev_idx + 1), self._bar_index):
                        if recent_idx in self.potential_shorts:
                            recent_high, recent_rsi = self.potential_shorts[recent_idx]
                            
                            # Bearish divergence: higher high in price, lower RSI
                            if (recent_high > prev_high and 
                                recent_rsi < prev_rsi - self.rsi_divergence_threshold):
                                
                                result['has_bearish_divergence'] = True
                                result['confirmed_short'] = True
                                result['divergence_strength'] = prev_rsi - recent_rsi
                                result['bars_since_divergence'] = self._bar_index - recent_idx
                                break
                
                if result['confirmed_short']:
                    break
            
            self._was_above_upper = False
        
        self._state.set_value(result)
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._price_buffer.clear()
        self._gains.clear()
        self._losses.clear()
        self._prev_price = None
        self._rsi_initialized = False
        self.potential_longs.clear()
        self.potential_shorts.clear()
        self._bar_index = 0
        self._was_below_lower = False
        self._was_above_upper = False