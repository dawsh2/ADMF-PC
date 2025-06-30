"""
Divergence detection features.

Features that track price/indicator divergences for mean reversion signals.
"""

from typing import Optional, Dict, Any, Tuple, List
from collections import deque
from ..protocols import Feature, FeatureState


class BollingerRSIDivergence:
    """
    Tracks RSI divergences at Bollinger Band extremes.
    
    This implements the exact logic from the profitable backtest:
    - Price makes new low below lower band
    - RSI makes higher low (divergence)
    - Confirms when price closes back inside bands
    - Tracks for both long and short setups
    """
    
    def __init__(self, lookback: int = 20, rsi_divergence_threshold: float = 5.0,
                 confirmation_bars: int = 10, name: str = "bb_rsi_divergence"):
        self._state = FeatureState(name)
        self.lookback = lookback
        self.rsi_divergence_threshold = rsi_divergence_threshold
        self.confirmation_bars = confirmation_bars
        
        # Track extremes when price is outside bands
        self._potential_longs: List[Tuple[int, float, float]] = []  # (bar_idx, low_price, rsi_value)
        self._potential_shorts: List[Tuple[int, float, float]] = []  # (bar_idx, high_price, rsi_value)
        
        # Current bar index
        self._bar_index = 0
        
        # Last confirmed divergence
        self._last_divergence: Optional[Dict[str, Any]] = None
        
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
        Update divergence tracking with new bar data.
        
        Args:
            price: Close price
            high: High price
            low: Low price
            rsi: RSI value
            upper_band: Bollinger upper band
            lower_band: Bollinger lower band
        """
        # Get band values from kwargs
        upper_band = kwargs.get('bb_upper')
        lower_band = kwargs.get('bb_lower')
        rsi = kwargs.get('rsi')
        
        if any(v is None for v in [high, low, rsi, upper_band, lower_band]):
            return None
            
        self._bar_index += 1
        
        # Clean old potential signals beyond lookback
        self._potential_longs = [(idx, low_p, rsi_v) for idx, low_p, rsi_v in self._potential_longs 
                                if self._bar_index - idx <= self.lookback]
        self._potential_shorts = [(idx, high_p, rsi_v) for idx, high_p, rsi_v in self._potential_shorts 
                                 if self._bar_index - idx <= self.lookback]
        
        # Track new extremes when price is outside bands
        if price < lower_band:
            self._potential_longs.append((self._bar_index, low, rsi))
        elif price > upper_band:
            self._potential_shorts.append((self._bar_index, high, rsi))
        
        # Initialize result
        result = {
            'has_bullish_divergence': False,
            'has_bearish_divergence': False,
            'confirmed_long': False,
            'confirmed_short': False,
            'bars_since_divergence': None,
            'divergence_strength': 0.0
        }
        
        # Look for bullish divergence (for longs)
        if price > lower_band:  # Price back inside bands - potential confirmation
            for i in range(len(self._potential_longs)):
                for j in range(i + 1, len(self._potential_longs)):
                    prev_idx, prev_low, prev_rsi = self._potential_longs[i]
                    recent_idx, recent_low, recent_rsi = self._potential_longs[j]
                    
                    # Check for divergence: lower low in price, higher RSI
                    if (recent_low < prev_low and 
                        recent_rsi > prev_rsi + self.rsi_divergence_threshold and
                        self._bar_index - recent_idx <= self.confirmation_bars):
                        
                        result['has_bullish_divergence'] = True
                        result['confirmed_long'] = True
                        result['bars_since_divergence'] = self._bar_index - recent_idx
                        result['divergence_strength'] = recent_rsi - prev_rsi
                        self._last_divergence = {
                            'type': 'bullish',
                            'bar': self._bar_index,
                            'extreme_price': recent_low,
                            'extreme_rsi': recent_rsi
                        }
                        break
                if result['confirmed_long']:
                    break
        
        # Look for bearish divergence (for shorts)
        if price < upper_band:  # Price back inside bands - potential confirmation
            for i in range(len(self._potential_shorts)):
                for j in range(i + 1, len(self._potential_shorts)):
                    prev_idx, prev_high, prev_rsi = self._potential_shorts[i]
                    recent_idx, recent_high, recent_rsi = self._potential_shorts[j]
                    
                    # Check for divergence: higher high in price, lower RSI
                    if (recent_high > prev_high and 
                        recent_rsi < prev_rsi - self.rsi_divergence_threshold and
                        self._bar_index - recent_idx <= self.confirmation_bars):
                        
                        result['has_bearish_divergence'] = True
                        result['confirmed_short'] = True
                        result['bars_since_divergence'] = self._bar_index - recent_idx
                        result['divergence_strength'] = prev_rsi - recent_rsi
                        self._last_divergence = {
                            'type': 'bearish',
                            'bar': self._bar_index,
                            'extreme_price': recent_high,
                            'extreme_rsi': recent_rsi
                        }
                        break
                if result['confirmed_short']:
                    break
        
        # Add last divergence info if exists
        if self._last_divergence:
            result['last_divergence'] = self._last_divergence
            
        self._state.set_value(result)
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        self._potential_longs.clear()
        self._potential_shorts.clear()
        self._bar_index = 0
        self._last_divergence = None


# Feature registry for divergence indicators
DIVERGENCE_FEATURES = {
    'bb_rsi_divergence': BollingerRSIDivergence,
}