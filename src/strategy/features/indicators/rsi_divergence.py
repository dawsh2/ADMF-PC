"""
True RSI Divergence Feature

This feature detects actual RSI divergences by comparing price extremes
with their corresponding RSI values over time.
"""

from typing import Optional, Dict, Any, List, Tuple
from ..protocols import Feature, FeatureState
from collections import deque
import numpy as np


class RSIDivergence(Feature):
    """
    Detects true RSI divergences.
    
    Bullish divergence: Price makes lower low, RSI makes higher low
    Bearish divergence: Price makes higher high, RSI makes lower high
    """
    
    def __init__(self, 
                 rsi_period: int = 14,
                 lookback_bars: int = 50,
                 min_bars_between: int = 5,
                 rsi_divergence_threshold: float = 5.0,
                 price_threshold_pct: float = 0.001,
                 name: str = "rsi_divergence"):
        
        self._state = FeatureState(name)
        self.rsi_period = rsi_period
        self.lookback_bars = lookback_bars
        self.min_bars_between = min_bars_between
        self.rsi_divergence_threshold = rsi_divergence_threshold
        self.price_threshold_pct = price_threshold_pct
        
        # Track price extremes with RSI values
        self.lows = deque(maxlen=lookback_bars)  # (bar_idx, price_low, rsi_value)
        self.highs = deque(maxlen=lookback_bars)  # (bar_idx, price_high, rsi_value)
        
        self._bar_index = 0
        self._last_divergence = None
        
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[Dict[str, Any]]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._bar_index > self.rsi_period
    
    @property
    def dependencies(self) -> List[str]:
        """This feature depends on RSI."""
        return [f'rsi_{self.rsi_period}']
    
    def update_with_features(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update with pre-computed features."""
        self._bar_index += 1
        
        # Get current values from bar data
        high = data.get('high', data.get('close', 0))
        low = data.get('low', data.get('close', 0))
        close = data.get('close', 0)
        # Get RSI from computed features (already merged into data)
        rsi = data.get(f'rsi_{self.rsi_period}', 50)
        
        # Debug: log RSI values periodically
        import logging
        logger = logging.getLogger(__name__)
        if self._bar_index % 20 == 1:
            logger.info(f"RSIDivergence update {self._bar_index}: RSI={rsi}, close={close}, high={high}, low={low}")
            logger.info(f"  Looking for RSI key: rsi_{self.rsi_period}")
            logger.info(f"  Available keys: {[k for k in data.keys() if 'rsi' in k.lower()]}")
            if len(self.lows) > 5:
                recent_lows = list(self.lows)[-5:]
                logger.info(f"Recent lows: {[(idx, f'{price:.2f}', f'{r:.1f}') for idx, price, r in recent_lows]}")
        
        # Always track all bars with their RSI values
        self.lows.append((self._bar_index, low, rsi))
        self.highs.append((self._bar_index, high, rsi))
        
        # Initialize result
        result = {
            'has_bullish_divergence': False,
            'has_bearish_divergence': False,
            'divergence_type': None,
            'divergence_strength': 0.0,
            'current_rsi': rsi,
            'current_price': close,
            'bars_since_divergence': None
        }
        
        # Need at least some history
        if len(self.lows) < 10:
            self._state.set_value(result)
            return self._state.value
        
        # Find local extremes for divergence detection
        
        # Check for bullish divergence (at price lows)
        # Current bar might be a significant low
        recent_lows = [(idx, price, rsi_val) for idx, price, rsi_val in self.lows 
                       if self._bar_index - idx <= 20]
        
        if len(recent_lows) >= 2:
            # Is current low a local minimum?
            current_low = low
            is_local_low = True
            for idx, price, _ in recent_lows[-5:-1]:  # Check last few bars
                if price < current_low:
                    is_local_low = False
                    break
            
            if is_local_low:
                # Compare with previous significant lows
                for prev_idx, prev_low, prev_rsi in recent_lows[:-5]:
                    if self._bar_index - prev_idx >= self.min_bars_between:
                        # Check for bullish divergence
                        price_change_pct = (current_low - prev_low) / prev_low
                        
                        # Price made lower low (with threshold)
                        if price_change_pct < -self.price_threshold_pct:
                            # RSI made higher low
                            if rsi > prev_rsi + self.rsi_divergence_threshold:
                                result['has_bullish_divergence'] = True
                                result['divergence_type'] = 'bullish'
                                result['divergence_strength'] = rsi - prev_rsi
                                result['bars_between'] = self._bar_index - prev_idx
                                result['price_change_pct'] = price_change_pct * 100
                                self._last_divergence = {
                                    'type': 'bullish',
                                    'bar': self._bar_index,
                                    'price': current_low,
                                    'rsi': rsi
                                }
                                logger.info(f"BULLISH DIVERGENCE DETECTED at bar {self._bar_index}: price {current_low:.2f} < {prev_low:.2f}, RSI {rsi:.1f} > {prev_rsi:.1f}")
                                break
        
        # Check for bearish divergence (at price highs)
        recent_highs = [(idx, price, rsi_val) for idx, price, rsi_val in self.highs 
                        if self._bar_index - idx <= 20]
        
        if len(recent_highs) >= 2 and not result['has_bullish_divergence']:
            # Is current high a local maximum?
            current_high = high
            is_local_high = True
            for idx, price, _ in recent_highs[-5:-1]:
                if price > current_high:
                    is_local_high = False
                    break
            
            if is_local_high:
                # Compare with previous significant highs
                for prev_idx, prev_high, prev_rsi in recent_highs[:-5]:
                    if self._bar_index - prev_idx >= self.min_bars_between:
                        # Check for bearish divergence
                        price_change_pct = (current_high - prev_high) / prev_high
                        
                        # Price made higher high (with threshold)
                        if price_change_pct > self.price_threshold_pct:
                            # RSI made lower high
                            if rsi < prev_rsi - self.rsi_divergence_threshold:
                                result['has_bearish_divergence'] = True
                                result['divergence_type'] = 'bearish'
                                result['divergence_strength'] = prev_rsi - rsi
                                result['bars_between'] = self._bar_index - prev_idx
                                result['price_change_pct'] = price_change_pct * 100
                                self._last_divergence = {
                                    'type': 'bearish',
                                    'bar': self._bar_index,
                                    'price': current_high,
                                    'rsi': rsi
                                }
                                logger.info(f"BEARISH DIVERGENCE DETECTED at bar {self._bar_index}: price {current_high:.2f} > {prev_high:.2f}, RSI {rsi:.1f} < {prev_rsi:.1f}")
                                break
        
        # Add info about last divergence
        if self._last_divergence:
            result['bars_since_divergence'] = self._bar_index - self._last_divergence['bar']
        
        self._state.set_value(result)
        return self._state.value
    
    def update(self, price: float, high: Optional[float] = None, 
               low: Optional[float] = None, volume: Optional[float] = None,
               **kwargs) -> Optional[Dict[str, Any]]:
        """Standard update method - requires features to be passed separately."""
        raise NotImplementedError("Use update_with_features for this feature")
    
    def reset(self) -> None:
        self._state.reset()
        self.lows.clear()
        self.highs.clear()
        self._bar_index = 0
        self._last_divergence = None