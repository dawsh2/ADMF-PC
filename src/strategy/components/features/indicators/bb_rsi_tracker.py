"""
Bollinger Band + RSI Divergence Tracker.

This feature tracks price extremes at Bollinger Bands and their corresponding
RSI values to detect divergences across multiple bars - exactly as specified
in the profitable backtest.
"""

from typing import Optional, Dict, Any, Tuple, List
from collections import deque
from ..protocols import Feature, FeatureState


class BollingerRSITracker:
    """
    Tracks price extremes outside Bollinger Bands with RSI values for divergence detection.
    
    This implements the EXACT logic from the profitable backtest:
    - Stores price lows/highs when outside bands with RSI values
    - Detects divergence: price makes new extreme, RSI doesn't
    - Confirms when price closes back inside bands
    """
    
    def __init__(self, lookback_bars: int = 20, rsi_divergence_threshold: float = 5.0,
                 confirmation_bars: int = 10, name: str = "bb_rsi_tracker"):
        self._state = FeatureState(name)
        self.lookback_bars = lookback_bars
        self.rsi_divergence_threshold = rsi_divergence_threshold
        self.confirmation_bars = confirmation_bars
        
        # Track extremes when price is outside bands
        # Format: (bar_index, price_extreme, rsi_value)
        self.potential_longs: Dict[int, Tuple[float, float]] = {}
        self.potential_shorts: Dict[int, Tuple[float, float]] = {}
        
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
    
    def update(self, price: float, high: Optional[float] = None, 
               low: Optional[float] = None, volume: Optional[float] = None,
               **kwargs) -> Optional[Dict[str, Any]]:
        """
        Update divergence tracking. Expects additional data in kwargs:
        - bb_upper: Upper Bollinger Band
        - bb_lower: Lower Bollinger Band  
        - bb_middle: Middle Band (SMA)
        - rsi: Current RSI value
        """
        # Get required values from kwargs
        bb_upper = kwargs.get('bb_upper')
        bb_lower = kwargs.get('bb_lower')
        bb_middle = kwargs.get('bb_middle')
        rsi = kwargs.get('rsi')
        
        if any(v is None for v in [high, low, bb_upper, bb_lower, bb_middle, rsi]):
            return None
            
        self._bar_index += 1
        
        # Clean old potential signals beyond lookback
        self.potential_longs = {idx: val for idx, val in self.potential_longs.items() 
                               if self._bar_index - idx <= self.lookback_bars}
        self.potential_shorts = {idx: val for idx, val in self.potential_shorts.items() 
                                if self._bar_index - idx <= self.lookback_bars}
        
        # Track new extremes when price is outside bands
        if price < bb_lower:
            self.potential_longs[self._bar_index] = (low, rsi)
            self._was_below_lower = True
        elif price > bb_upper:
            self.potential_shorts[self._bar_index] = (high, rsi)
            self._was_above_upper = True
        
        # Initialize result
        result = {
            'confirmed_long': False,
            'confirmed_short': False,
            'has_bullish_divergence': False,
            'has_bearish_divergence': False,
            'divergence_strength': 0.0,
            'bars_since_divergence': None
        }
        
        # Look for divergence and confirmation (Long)
        if price > bb_lower and self._was_below_lower:  # Price back inside bands
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
        if price < bb_upper and self._was_above_upper:  # Price back inside bands
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
        self.potential_longs.clear()
        self.potential_shorts.clear()
        self._bar_index = 0
        self._was_below_lower = False
        self._was_above_upper = False