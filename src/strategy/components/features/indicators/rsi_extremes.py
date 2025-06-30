"""
RSI extreme tracking for divergence detection.

This feature tracks RSI values when price makes new highs/lows,
enabling divergence detection in strategies.
"""

from typing import Optional, Dict, Any, Tuple, List
from collections import deque
from ..protocols import Feature, FeatureState


class RSIExtremeTracker:
    """
    Tracks RSI values at price extremes for divergence detection.
    
    This is a simpler approach that just tracks when price makes new extremes
    and what the RSI value was at that time. Strategies can use this data
    to detect divergences.
    """
    
    def __init__(self, lookback: int = 20, extreme_threshold: float = 0.001, name: str = "rsi_extremes"):
        self._state = FeatureState(name)
        self.lookback = lookback
        self.extreme_threshold = extreme_threshold
        
        # Price and RSI tracking
        self._price_buffer = deque(maxlen=lookback)
        self._rsi_buffer = deque(maxlen=lookback)
        self._high_buffer = deque(maxlen=lookback)
        self._low_buffer = deque(maxlen=lookback)
        
        # Track recent extremes
        self._recent_high_price = None
        self._recent_high_rsi = None
        self._recent_high_bar = None
        
        self._recent_low_price = None
        self._recent_low_rsi = None
        self._recent_low_bar = None
        
        self._bar_count = 0
        
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
               rsi_14: Optional[float] = None, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Track RSI at price extremes.
        """
        if high is None or low is None:
            return None
            
        # For now, we'll need the strategy to pass RSI via metadata
        # This is a limitation of the current feature hub design
        self._bar_count += 1
        
        self._price_buffer.append(price)
        self._high_buffer.append(high)
        self._low_buffer.append(low)
        
        # If we have enough data
        if len(self._high_buffer) >= 5:
            # Check if current bar is a recent high
            if high >= max(self._high_buffer):
                self._recent_high_price = high
                self._recent_high_bar = self._bar_count
                # Note: RSI would need to be passed from strategy
                
            # Check if current bar is a recent low  
            if low <= min(self._low_buffer):
                self._recent_low_price = low
                self._recent_low_bar = self._bar_count
                # Note: RSI would need to be passed from strategy
        
        result = {
            'recent_high_price': self._recent_high_price,
            'recent_high_bar': self._recent_high_bar,
            'recent_low_price': self._recent_low_price,
            'recent_low_bar': self._recent_low_bar,
            'bars_since_high': self._bar_count - self._recent_high_bar if self._recent_high_bar else None,
            'bars_since_low': self._bar_count - self._recent_low_bar if self._recent_low_bar else None,
        }
        
        self._state.set_value(result)
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._price_buffer.clear()
        self._rsi_buffer.clear()
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._recent_high_price = None
        self._recent_high_rsi = None
        self._recent_high_bar = None
        self._recent_low_price = None
        self._recent_low_rsi = None
        self._recent_low_bar = None
        self._bar_count = 0


# Feature registry
RSI_EXTREME_FEATURES = {
    'rsi_extremes': RSIExtremeTracker,
}