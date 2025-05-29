"""
Price-based feature extractors.
"""

from typing import Dict, Any, List, Optional, Deque
from datetime import datetime
from collections import deque
import numpy as np
import logging

from ...core.events import Event
from ..protocols import FeatureExtractor


logger = logging.getLogger(__name__)


class PriceFeatureExtractor:
    """
    Extracts basic price features.
    
    Features:
    - Price levels (open, high, low, close)
    - Price ratios (HL ratio, OC ratio)
    - Price position (position within daily range)
    """
    
    def __init__(self, name: str = "price_features"):
        self.name = name
        self.features: Dict[str, float] = {}
        self.last_bar: Optional[Dict[str, Any]] = None
        
        # Capabilities
        self._events = None
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract price features from bar data."""
        features = {}
        
        # Basic price levels
        open_price = data.get('open', 0)
        high = data.get('high', 0)
        low = data.get('low', 0)
        close = data.get('close', 0)
        
        features['open'] = open_price
        features['high'] = high
        features['low'] = low
        features['close'] = close
        
        # Price ratios
        if high > low:
            features['hl_ratio'] = high / low
            features['price_position'] = (close - low) / (high - low)
        else:
            features['hl_ratio'] = 1.0
            features['price_position'] = 0.5
        
        if open_price > 0:
            features['oc_ratio'] = close / open_price
        else:
            features['oc_ratio'] = 1.0
        
        # Bar characteristics
        features['range'] = high - low
        features['body'] = abs(close - open_price)
        features['upper_shadow'] = high - max(open_price, close)
        features['lower_shadow'] = min(open_price, close) - low
        
        # Gap from previous bar
        if self.last_bar:
            prev_close = self.last_bar.get('close', close)
            features['gap'] = open_price - prev_close
            features['gap_pct'] = (open_price - prev_close) / prev_close if prev_close > 0 else 0
        else:
            features['gap'] = 0
            features['gap_pct'] = 0
        
        self.last_bar = data
        self.features = features
        
        return features
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return list(self.features.keys())
    
    @property
    def ready(self) -> bool:
        """Whether extractor is ready."""
        return len(self.features) > 0
    
    def reset(self) -> None:
        """Reset extractor state."""
        self.features.clear()
        self.last_bar = None


class PriceReturnExtractor:
    """
    Extracts price return features over multiple timeframes.
    """
    
    def __init__(self, periods: List[int] = [1, 5, 10, 20], name: str = "return_features"):
        self.name = name
        self.periods = periods
        self.price_history: Deque[float] = deque(maxlen=max(periods) + 1)
        self.features: Dict[str, float] = {}
        
        # Capabilities
        self._events = None
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract return features."""
        features = {}
        
        close = data.get('close', data.get('price', 0))
        self.price_history.append(close)
        
        # Calculate returns for each period
        for period in self.periods:
            if len(self.price_history) > period:
                old_price = self.price_history[-(period + 1)]
                if old_price > 0:
                    ret = (close - old_price) / old_price
                    features[f'return_{period}'] = ret
                    features[f'log_return_{period}'] = np.log(close / old_price)
                else:
                    features[f'return_{period}'] = 0
                    features[f'log_return_{period}'] = 0
            else:
                features[f'return_{period}'] = 0
                features[f'log_return_{period}'] = 0
        
        self.features = features
        return features
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return list(self.features.keys())
    
    @property
    def ready(self) -> bool:
        """Whether extractor is ready."""
        return len(self.price_history) > self.periods[0]
    
    def reset(self) -> None:
        """Reset extractor state."""
        self.price_history.clear()
        self.features.clear()


class VolatilityExtractor:
    """
    Extracts volatility-related features.
    """
    
    def __init__(self, periods: List[int] = [10, 20, 50], name: str = "volatility_features"):
        self.name = name
        self.periods = periods
        self.return_history: Deque[float] = deque(maxlen=max(periods))
        self.features: Dict[str, float] = {}
        self.last_price: Optional[float] = None
        
        # Capabilities
        self._events = None
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract volatility features."""
        features = {}
        
        close = data.get('close', data.get('price', 0))
        
        # Calculate return
        if self.last_price is not None and self.last_price > 0:
            ret = (close - self.last_price) / self.last_price
            self.return_history.append(ret)
        
        self.last_price = close
        
        # Calculate volatility for each period
        for period in self.periods:
            if len(self.return_history) >= period:
                recent_returns = list(self.return_history)[-period:]
                vol = np.std(recent_returns) * np.sqrt(252)  # Annualized
                features[f'volatility_{period}'] = vol
                
                # Also calculate realized volatility (sum of squared returns)
                realized_vol = np.sqrt(sum(r**2 for r in recent_returns)) * np.sqrt(252/period)
                features[f'realized_vol_{period}'] = realized_vol
            else:
                features[f'volatility_{period}'] = 0
                features[f'realized_vol_{period}'] = 0
        
        # High-low volatility (Parkinson)
        high = data.get('high', close)
        low = data.get('low', close)
        if low > 0:
            hl_vol = np.log(high / low) / (2 * np.sqrt(np.log(2)))
            features['hl_volatility'] = hl_vol * np.sqrt(252)
        else:
            features['hl_volatility'] = 0
        
        self.features = features
        return features
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return list(self.features.keys())
    
    @property
    def ready(self) -> bool:
        """Whether extractor is ready."""
        return len(self.return_history) >= self.periods[0]
    
    def reset(self) -> None:
        """Reset extractor state."""
        self.return_history.clear()
        self.features.clear()
        self.last_price = None


class PricePatternExtractor:
    """
    Extracts price pattern features.
    
    Identifies common candlestick patterns and price formations.
    """
    
    def __init__(self, lookback: int = 5, name: str = "pattern_features"):
        self.name = name
        self.lookback = lookback
        self.bar_history: Deque[Dict[str, Any]] = deque(maxlen=lookback)
        self.features: Dict[str, float] = {}
        
        # Capabilities
        self._events = None
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract pattern features."""
        features = {}
        
        self.bar_history.append(data)
        
        if len(self.bar_history) >= 2:
            # Candlestick patterns
            features.update(self._extract_candlestick_patterns())
        
        if len(self.bar_history) >= self.lookback:
            # Price formations
            features.update(self._extract_price_formations())
        
        self.features = features
        return features
    
    def _extract_candlestick_patterns(self) -> Dict[str, float]:
        """Extract candlestick pattern features."""
        patterns = {}
        
        # Current and previous bars
        curr = self.bar_history[-1]
        prev = self.bar_history[-2]
        
        curr_open = curr.get('open', 0)
        curr_high = curr.get('high', 0)
        curr_low = curr.get('low', 0)
        curr_close = curr.get('close', 0)
        
        prev_open = prev.get('open', 0)
        prev_close = prev.get('close', 0)
        
        # Doji pattern
        body = abs(curr_close - curr_open)
        range_hl = curr_high - curr_low
        if range_hl > 0:
            patterns['doji'] = 1.0 if body / range_hl < 0.1 else 0.0
        else:
            patterns['doji'] = 0.0
        
        # Hammer pattern
        lower_shadow = min(curr_open, curr_close) - curr_low
        upper_shadow = curr_high - max(curr_open, curr_close)
        if body > 0 and lower_shadow > 2 * body and upper_shadow < 0.1 * body:
            patterns['hammer'] = 1.0
        else:
            patterns['hammer'] = 0.0
        
        # Engulfing pattern
        if prev_close > prev_open:  # Previous bullish
            if curr_open > prev_close and curr_close < prev_open:
                patterns['bearish_engulfing'] = 1.0
            else:
                patterns['bearish_engulfing'] = 0.0
            patterns['bullish_engulfing'] = 0.0
        else:  # Previous bearish
            if curr_open < prev_close and curr_close > prev_open:
                patterns['bullish_engulfing'] = 1.0
            else:
                patterns['bullish_engulfing'] = 0.0
            patterns['bearish_engulfing'] = 0.0
        
        return patterns
    
    def _extract_price_formations(self) -> Dict[str, float]:
        """Extract price formation features."""
        formations = {}
        
        # Extract closing prices
        closes = [bar.get('close', 0) for bar in self.bar_history]
        highs = [bar.get('high', 0) for bar in self.bar_history]
        lows = [bar.get('low', 0) for bar in self.bar_history]
        
        # Higher highs and lower lows
        hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        ll_count = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        
        formations['higher_highs'] = hh_count / (len(highs) - 1)
        formations['lower_lows'] = ll_count / (len(lows) - 1)
        
        # Trend strength
        if closes[0] > 0:
            trend = (closes[-1] - closes[0]) / closes[0]
            formations['trend_strength'] = trend
        else:
            formations['trend_strength'] = 0
        
        # Price acceleration
        if len(closes) >= 3:
            recent_change = closes[-1] - closes[-2] if closes[-2] > 0 else 0
            older_change = closes[-2] - closes[-3] if closes[-3] > 0 else 0
            formations['acceleration'] = recent_change - older_change
        else:
            formations['acceleration'] = 0
        
        return formations
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return list(self.features.keys())
    
    @property
    def ready(self) -> bool:
        """Whether extractor is ready."""
        return len(self.bar_history) >= 2
    
    def reset(self) -> None:
        """Reset extractor state."""
        self.bar_history.clear()
        self.features.clear()