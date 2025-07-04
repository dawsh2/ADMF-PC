"""
Bollinger Band + RSI Divergence - Dependent Feature Version.

This version properly depends on other features (BB and RSI) rather than
computing them internally.
"""

from typing import Optional, Dict, Any, Tuple, List
from collections import deque
from ..protocols import Feature, FeatureState


class BollingerRSIDependentFeature:
    """
    Tracks RSI divergences at Bollinger Band extremes.
    
    This version depends on bollinger_bands and rsi features being computed first.
    """
    
    def __init__(self, lookback: int = 20, rsi_divergence_threshold: float = 5.0,
                 confirmation_bars: int = 10, bb_period: int = 20, bb_std: float = 2.0,
                 rsi_period: int = 14, name: str = "bb_rsi_dependent"):
        self._state = FeatureState(name)
        
        # Parameters
        self.lookback = lookback
        self.rsi_divergence_threshold = rsi_divergence_threshold
        self.confirmation_bars = confirmation_bars
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        
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
        
    @property
    def dependencies(self) -> List[str]:
        """List of features this depends on."""
        # Return the base feature names that will be expanded by FeatureHub
        return ['bollinger_bands', 'rsi']
    
    def update_with_features(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update with bar data and computed features.
        
        This method is called by EnhancedFeatureHub with all available data.
        """
        # Extract bar data
        price = data.get('close', 0)
        high = data.get('high')
        low = data.get('low')
        
        # Extract computed features we depend on
        # Based on actual feature naming from FeatureHub
        upper_band = data.get('bollinger_bands_upper')
        middle_band = data.get('bollinger_bands_middle')
        lower_band = data.get('bollinger_bands_lower')
        rsi = data.get('rsi')
        
        if any(v is None for v in [high, low, upper_band, lower_band, middle_band, rsi]):
            return None
            
        self._bar_index += 1
        
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
            'bars_since_divergence': None
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
    
    def update(self, price: float, high: Optional[float] = None, 
               low: Optional[float] = None, volume: Optional[float] = None,
               **kwargs) -> Optional[Dict[str, Any]]:
        """Standard update method - redirects to update_with_features."""
        # This method is here for compatibility but shouldn't be called
        # when used with EnhancedFeatureHub
        data = {
            'close': price,
            'high': high,
            'low': low,
            'volume': volume,
            **kwargs
        }
        return self.update_with_features(data)
    
    def reset(self) -> None:
        self._state.reset()
        self.potential_longs.clear()
        self.potential_shorts.clear()
        self._bar_index = 0
        self._was_below_lower = False
        self._was_above_upper = False