"""
Bollinger Band + RSI Immediate Divergence Feature

This feature detects divergences immediately when they occur outside the bands,
without waiting for price to re-enter the bands.
"""

from typing import Optional, Dict, Any, List, Tuple
from ..protocols import Feature, FeatureState
from collections import deque


class BollingerRSIImmediateDivergence(Feature):
    """
    Detects BB+RSI divergences immediately at extremes.
    
    Key differences from bb_rsi_dependent:
    - Signals divergence immediately when detected outside bands
    - No waiting for confirmation (price re-entering bands)
    - Simpler state tracking
    """
    
    def __init__(self, 
                 lookback: int = 20,
                 rsi_divergence_threshold: float = 5.0,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_period: int = 14,
                 name: str = "bb_rsi_immediate_divergence"):
        
        self._state = FeatureState(name)
        self.lookback = lookback
        self.rsi_divergence_threshold = rsi_divergence_threshold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        
        # Track recent extremes
        self.price_lows = deque(maxlen=lookback)  # (bar_idx, price, rsi)
        self.price_highs = deque(maxlen=lookback)  # (bar_idx, price, rsi)
        
        self._bar_index = 0
        self._last_signal_bar = -1000  # Track when we last signaled
        self._min_bars_between_signals = 10
        
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, Any]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._bar_index > self.bb_period
    
    @property
    def dependencies(self) -> List[str]:
        """This feature depends on BB and RSI."""
        return [
            f'bollinger_bands_{self.bb_period}_{self.bb_std}',
            f'rsi_{self.rsi_period}'
        ]
    
    def update_with_features(self, data: Dict[str, Any], features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update with pre-computed features."""
        self._bar_index += 1
        
        # Get current values
        price = data.get('close', 0)
        low = data.get('low', price)
        high = data.get('high', price)
        
        # Get features
        upper_band = features.get(f'bollinger_bands_{self.bb_period}_{self.bb_std}_upper', price)
        lower_band = features.get(f'bollinger_bands_{self.bb_period}_{self.bb_std}_lower', price)
        middle_band = features.get(f'bollinger_bands_{self.bb_period}_{self.bb_std}_middle', price)
        rsi = features.get(f'rsi_{self.rsi_period}', 50)
        
        # Initialize result
        result = {
            'has_immediate_long': False,
            'has_immediate_short': False,
            'price': price,
            'rsi': rsi,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': middle_band,
            'bars_since_signal': self._bar_index - self._last_signal_bar
        }
        
        # Check if enough bars have passed since last signal
        if self._bar_index - self._last_signal_bar < self._min_bars_between_signals:
            self._state.set_value(result)
            return self._state.value
        
        # Track extremes when outside bands
        if price < lower_band:
            self.price_lows.append((self._bar_index, low, rsi))
            
            # Check for immediate bullish divergence
            if len(self.price_lows) >= 2:
                # Compare current extreme with previous ones
                for prev_idx, prev_low, prev_rsi in list(self.price_lows)[:-1]:
                    if self._bar_index - prev_idx > 5:  # Not too close
                        # Bullish divergence: lower low in price, higher RSI
                        if low < prev_low and rsi > prev_rsi + self.rsi_divergence_threshold:
                            result['has_immediate_long'] = True
                            result['divergence_strength'] = rsi - prev_rsi
                            result['bars_between_extremes'] = self._bar_index - prev_idx
                            self._last_signal_bar = self._bar_index
                            break
        
        elif price > upper_band:
            self.price_highs.append((self._bar_index, high, rsi))
            
            # Check for immediate bearish divergence
            if len(self.price_highs) >= 2:
                # Compare current extreme with previous ones
                for prev_idx, prev_high, prev_rsi in list(self.price_highs)[:-1]:
                    if self._bar_index - prev_idx > 5:  # Not too close
                        # Bearish divergence: higher high in price, lower RSI
                        if high > prev_high and rsi < prev_rsi - self.rsi_divergence_threshold:
                            result['has_immediate_short'] = True
                            result['divergence_strength'] = prev_rsi - rsi
                            result['bars_between_extremes'] = self._bar_index - prev_idx
                            self._last_signal_bar = self._bar_index
                            break
        
        self._state.set_value(result)
        return self._state.value
    
    def update(self, price: float, high: Optional[float] = None, 
               low: Optional[float] = None, volume: Optional[float] = None,
               **kwargs) -> Optional[Dict[str, Any]]:
        """Standard update method - requires features to be passed separately."""
        raise NotImplementedError("Use update_with_features for this feature")
    
    def reset(self) -> None:
        self._state.reset()
        self.price_lows.clear()
        self.price_highs.clear()
        self._bar_index = 0
        self._last_signal_bar = -1000