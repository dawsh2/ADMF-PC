"""
Debug version of BB RSI Dependent feature with comprehensive logging.
"""

from typing import Optional, Dict, Any, Tuple, List
from collections import deque
from ..protocols import Feature, FeatureState


class BollingerRSIDependentFeatureDebug:
    """
    Debug version with logging at every step.
    """
    
    def __init__(self, lookback: int = 20, rsi_divergence_threshold: float = 5.0,
                 confirmation_bars: int = 10, bb_period: int = 20, bb_std: float = 2.0,
                 rsi_period: int = 14, name: str = "bb_rsi_dependent"):
        print(f"[INIT] BollingerRSIDependentFeatureDebug created with threshold={rsi_divergence_threshold}")
        self._state = FeatureState(name)
        
        # Parameters
        self.lookback = lookback
        self.rsi_divergence_threshold = rsi_divergence_threshold
        self.confirmation_bars = confirmation_bars
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        
        # Track extremes when price is outside bands
        self.potential_longs: Dict[int, Tuple[float, float]] = {}
        self.potential_shorts: Dict[int, Tuple[float, float]] = {}
        
        # Current bar index
        self._bar_index = 0
        
        # Track if we're currently outside bands
        self._was_below_lower = False
        self._was_above_upper = False
        
        # Debug counters
        self.debug_outside_band_count = 0
        self.debug_divergence_checks = 0
        self.debug_divergences_found = 0
        self.debug_confirmations = 0
        
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
        return ['bollinger_bands', 'rsi']
    
    def update_with_features(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update with bar data and computed features.
        """
        # Extract bar data
        price = data.get('close', 0)
        high = data.get('high')
        low = data.get('low')
        
        # Extract computed features - need to match the exact key names with parameters
        # Try to find the BB and RSI keys
        upper_band = None
        middle_band = None
        lower_band = None
        rsi_value = None
        
        for key, value in data.items():
            if key.startswith('bollinger_bands_') and key.endswith('_upper'):
                upper_band = value
            elif key.startswith('bollinger_bands_') and key.endswith('_middle'):
                middle_band = value
            elif key.startswith('bollinger_bands_') and key.endswith('_lower'):
                lower_band = value
            elif key.startswith('rsi_'):
                rsi_value = value
        
        rsi = rsi_value
        
        if any(v is None for v in [high, low, upper_band, lower_band, middle_band, rsi]):
            if self._bar_index < 3:  # Log early failures
                print(f"[UPDATE] Bar {self._bar_index}: Missing data - high={high}, low={low}, "
                      f"upper={upper_band}, lower={lower_band}, middle={middle_band}, rsi={rsi}")
                print(f"  All available keys: {list(data.keys())}")
            return None
            
        self._bar_index += 1
        
        # Log when we start getting data
        if self._bar_index == 1:
            print(f"[SUCCESS] First bar with complete data! Bar {self._bar_index}")
            print(f"  Price={price:.2f}, RSI={rsi:.2f}, Bands=[{lower_band:.2f}, {middle_band:.2f}, {upper_band:.2f}]")
        
        # Log every 1000 bars
        if self._bar_index % 1000 == 0:
            print(f"[DEBUG] Bar {self._bar_index}: Price={price:.2f}, RSI={rsi:.2f}, "
                  f"Bands=[{lower_band:.2f}, {middle_band:.2f}, {upper_band:.2f}]")
            print(f"  Stats: Outside bands {self.debug_outside_band_count} times, "
                  f"Checked {self.debug_divergence_checks} divergences, "
                  f"Found {self.debug_divergences_found}, Confirmed {self.debug_confirmations}")
        
        # Clean old potential signals
        self.potential_longs = {idx: val for idx, val in self.potential_longs.items() 
                               if self._bar_index - idx <= self.lookback}
        self.potential_shorts = {idx: val for idx, val in self.potential_shorts.items() 
                                if self._bar_index - idx <= self.lookback}
        
        # Track new extremes when price is outside bands
        if price < lower_band:
            self.potential_longs[self._bar_index] = (low, rsi)
            self._was_below_lower = True
            self.debug_outside_band_count += 1
            
            # Log significant events
            if self._bar_index % 100 == 0:
                print(f"  [OUTSIDE] Bar {self._bar_index}: Price {price:.2f} < Lower {lower_band:.2f}, RSI={rsi:.2f}")
                
        elif price > upper_band:
            self.potential_shorts[self._bar_index] = (high, rsi)
            self._was_above_upper = True
            self.debug_outside_band_count += 1
        
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
            print(f"  [CONFIRM CHECK] Bar {self._bar_index}: Price back inside bands, checking {len(self.potential_longs)} potential longs")
            
            # Look through all potential long setups
            for prev_idx, (prev_low, prev_rsi) in self.potential_longs.items():
                if prev_idx < self._bar_index - 1:  # Not same or adjacent bar
                    # Check all bars between prev and now for lower low + higher RSI
                    for recent_idx in range(max(self._bar_index - self.confirmation_bars, prev_idx + 1), self._bar_index):
                        if recent_idx in self.potential_longs:
                            recent_low, recent_rsi = self.potential_longs[recent_idx]
                            self.debug_divergence_checks += 1
                            
                            # Bullish divergence: lower low in price, higher RSI
                            if recent_low < prev_low and recent_rsi > prev_rsi + self.rsi_divergence_threshold:
                                self.debug_divergences_found += 1
                                self.debug_confirmations += 1
                                
                                print(f"  [DIVERGENCE!] Bullish divergence found!")
                                print(f"    Old: Bar {prev_idx}, Low={prev_low:.2f}, RSI={prev_rsi:.2f}")
                                print(f"    New: Bar {recent_idx}, Low={recent_low:.2f}, RSI={recent_rsi:.2f}")
                                print(f"    RSI diff: {recent_rsi - prev_rsi:.2f} > {self.rsi_divergence_threshold}")
                                
                                result['has_bullish_divergence'] = True
                                result['confirmed_long'] = True
                                result['divergence_strength'] = recent_rsi - prev_rsi
                                result['bars_since_divergence'] = self._bar_index - recent_idx
                                break
                
                if result['confirmed_long']:
                    break
            
            self._was_below_lower = False
        
        # Look for divergence and confirmation (Short) - similar logic
        if price < upper_band and self._was_above_upper:
            for prev_idx, (prev_high, prev_rsi) in self.potential_shorts.items():
                if prev_idx < self._bar_index - 1:
                    for recent_idx in range(max(self._bar_index - self.confirmation_bars, prev_idx + 1), self._bar_index):
                        if recent_idx in self.potential_shorts:
                            recent_high, recent_rsi = self.potential_shorts[recent_idx]
                            self.debug_divergence_checks += 1
                            
                            # Bearish divergence: higher high in price, lower RSI
                            if recent_high > prev_high and recent_rsi < prev_rsi - self.rsi_divergence_threshold:
                                self.debug_divergences_found += 1
                                self.debug_confirmations += 1
                                
                                print(f"  [DIVERGENCE!] Bearish divergence found!")
                                print(f"    Old: Bar {prev_idx}, High={prev_high:.2f}, RSI={prev_rsi:.2f}")
                                print(f"    New: Bar {recent_idx}, High={recent_high:.2f}, RSI={recent_rsi:.2f}")
                                print(f"    RSI diff: {prev_rsi - recent_rsi:.2f} > {self.rsi_divergence_threshold}")
                                
                                result['has_bearish_divergence'] = True
                                result['confirmed_short'] = True
                                result['divergence_strength'] = prev_rsi - recent_rsi
                                result['bars_since_divergence'] = self._bar_index - recent_idx
                                break
                
                if result['confirmed_short']:
                    break
            
            self._was_above_upper = False
        
        self._state.set_value(result)
        
        # Debug: log when we have a signal
        if result['confirmed_long'] or result['confirmed_short']:
            print(f"[SIGNAL] Bar {self._bar_index}: Long={result['confirmed_long']}, Short={result['confirmed_short']}")
        
        return self._state.value
    
    def update(self, price: float, high: Optional[float] = None, 
               low: Optional[float] = None, volume: Optional[float] = None,
               **kwargs) -> Optional[Dict[str, Any]]:
        """Standard update method - redirects to update_with_features."""
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
        self.debug_outside_band_count = 0
        self.debug_divergence_checks = 0
        self.debug_divergences_found = 0
        self.debug_confirmations = 0