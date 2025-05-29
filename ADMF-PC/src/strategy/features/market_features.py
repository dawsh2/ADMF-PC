"""
Market microstructure feature extractors.
"""

from typing import Dict, Any, List, Optional, Deque
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import logging

from ...core.events import Event
from ..protocols import FeatureExtractor


logger = logging.getLogger(__name__)


class MarketMicrostructureExtractor:
    """
    Extracts market microstructure features.
    
    Features:
    - Bid-ask spread
    - Order imbalance
    - Trade intensity
    - Price impact
    """
    
    def __init__(self, window: int = 100, name: str = "microstructure_features"):
        self.name = name
        self.window = window
        self.features: Dict[str, float] = {}
        
        # Market data history
        self.bid_history: Deque[float] = deque(maxlen=window)
        self.ask_history: Deque[float] = deque(maxlen=window)
        self.spread_history: Deque[float] = deque(maxlen=window)
        self.trade_history: Deque[Dict[str, Any]] = deque(maxlen=window)
        
        # Capabilities
        self._events = None
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract microstructure features."""
        features = {}
        
        # Extract bid/ask data
        bid = data.get('bid', data.get('close', 0))
        ask = data.get('ask', data.get('close', 0))
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else data.get('close', 0)
        
        self.bid_history.append(bid)
        self.ask_history.append(ask)
        
        # Spread features
        if bid > 0 and ask > bid:
            spread = ask - bid
            spread_pct = spread / mid
            self.spread_history.append(spread_pct)
            
            features['spread'] = spread
            features['spread_pct'] = spread_pct
            features['avg_spread'] = np.mean(self.spread_history)
            features['spread_volatility'] = np.std(self.spread_history) if len(self.spread_history) > 1 else 0
        else:
            features['spread'] = 0
            features['spread_pct'] = 0
            features['avg_spread'] = 0
            features['spread_volatility'] = 0
        
        # Order imbalance (if volume data available)
        bid_vol = data.get('bid_volume', 0)
        ask_vol = data.get('ask_volume', 0)
        total_vol = bid_vol + ask_vol
        
        if total_vol > 0:
            features['order_imbalance'] = (bid_vol - ask_vol) / total_vol
            features['bid_ratio'] = bid_vol / total_vol
            features['ask_ratio'] = ask_vol / total_vol
        else:
            features['order_imbalance'] = 0
            features['bid_ratio'] = 0.5
            features['ask_ratio'] = 0.5
        
        # Trade intensity
        volume = data.get('volume', 0)
        if volume > 0:
            self.trade_history.append({
                'volume': volume,
                'timestamp': data.get('timestamp', datetime.now())
            })
        
        if len(self.trade_history) > 1:
            recent_volume = sum(t['volume'] for t in self.trade_history)
            features['trade_intensity'] = recent_volume / len(self.trade_history)
            
            # Volume acceleration
            mid_point = len(self.trade_history) // 2
            first_half_vol = sum(t['volume'] for t in list(self.trade_history)[:mid_point])
            second_half_vol = sum(t['volume'] for t in list(self.trade_history)[mid_point:])
            
            if first_half_vol > 0:
                features['volume_acceleration'] = (second_half_vol - first_half_vol) / first_half_vol
            else:
                features['volume_acceleration'] = 0
        else:
            features['trade_intensity'] = volume
            features['volume_acceleration'] = 0
        
        self.features = features
        return features
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return list(self.features.keys())
    
    @property
    def ready(self) -> bool:
        """Whether extractor is ready."""
        return len(self.bid_history) > 0
    
    def reset(self) -> None:
        """Reset extractor state."""
        self.features.clear()
        self.bid_history.clear()
        self.ask_history.clear()
        self.spread_history.clear()
        self.trade_history.clear()


class VolumeProfileExtractor:
    """
    Extracts volume profile features.
    
    Analyzes volume distribution across price levels.
    """
    
    def __init__(self, 
                 bins: int = 20,
                 window: int = 100,
                 name: str = "volume_profile_features"):
        self.name = name
        self.bins = bins
        self.window = window
        self.features: Dict[str, float] = {}
        
        # Price-volume history
        self.pv_history: Deque[Tuple[float, float]] = deque(maxlen=window)
        
        # Capabilities
        self._events = None
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract volume profile features."""
        features = {}
        
        price = data.get('close', data.get('price', 0))
        volume = data.get('volume', 0)
        
        if price > 0 and volume > 0:
            self.pv_history.append((price, volume))
        
        if len(self.pv_history) >= 10:
            # Calculate volume profile
            prices = [pv[0] for pv in self.pv_history]
            volumes = [pv[1] for pv in self.pv_history]
            
            min_price = min(prices)
            max_price = max(prices)
            price_range = max_price - min_price
            
            if price_range > 0:
                # Create volume profile
                profile = self._calculate_volume_profile(prices, volumes, min_price, max_price)
                
                # Find POC (Point of Control)
                poc_idx = np.argmax(profile)
                poc_price = min_price + (poc_idx + 0.5) * price_range / self.bins
                
                features['poc_price'] = poc_price
                features['price_vs_poc'] = (price - poc_price) / poc_price if poc_price > 0 else 0
                
                # Value area (70% of volume)
                value_area = self._calculate_value_area(profile, 0.7)
                va_low = min_price + value_area[0] * price_range / self.bins
                va_high = min_price + value_area[1] * price_range / self.bins
                
                features['value_area_low'] = va_low
                features['value_area_high'] = va_high
                features['in_value_area'] = 1.0 if va_low <= price <= va_high else 0.0
                
                # Volume distribution skewness
                features['volume_skew'] = self._calculate_skewness(profile)
                
                # Current volume vs average at this price level
                current_bin = int((price - min_price) / price_range * self.bins)
                current_bin = max(0, min(self.bins - 1, current_bin))
                avg_vol_at_level = profile[current_bin] / len(self.pv_history)
                
                if avg_vol_at_level > 0:
                    features['volume_vs_avg'] = volume / avg_vol_at_level
                else:
                    features['volume_vs_avg'] = 1.0
        
        self.features = features
        return features
    
    def _calculate_volume_profile(self, prices: List[float], volumes: List[float],
                                min_price: float, max_price: float) -> np.ndarray:
        """Calculate volume profile histogram."""
        profile = np.zeros(self.bins)
        price_range = max_price - min_price
        
        for price, volume in zip(prices, volumes):
            bin_idx = int((price - min_price) / price_range * self.bins)
            bin_idx = max(0, min(self.bins - 1, bin_idx))
            profile[bin_idx] += volume
        
        return profile
    
    def _calculate_value_area(self, profile: np.ndarray, pct: float) -> Tuple[int, int]:
        """Calculate value area containing specified percentage of volume."""
        total_volume = np.sum(profile)
        target_volume = total_volume * pct
        
        # Start from POC and expand
        poc_idx = np.argmax(profile)
        low_idx = poc_idx
        high_idx = poc_idx
        current_volume = profile[poc_idx]
        
        while current_volume < target_volume:
            # Expand in direction with more volume
            left_vol = profile[low_idx - 1] if low_idx > 0 else 0
            right_vol = profile[high_idx + 1] if high_idx < self.bins - 1 else 0
            
            if left_vol > right_vol and low_idx > 0:
                low_idx -= 1
                current_volume += left_vol
            elif high_idx < self.bins - 1:
                high_idx += 1
                current_volume += right_vol
            else:
                break
        
        return (low_idx, high_idx)
    
    def _calculate_skewness(self, profile: np.ndarray) -> float:
        """Calculate skewness of volume distribution."""
        if np.sum(profile) == 0:
            return 0
        
        # Normalize profile
        norm_profile = profile / np.sum(profile)
        
        # Calculate moments
        indices = np.arange(len(profile))
        mean = np.sum(indices * norm_profile)
        variance = np.sum((indices - mean)**2 * norm_profile)
        
        if variance > 0:
            skewness = np.sum((indices - mean)**3 * norm_profile) / (variance**1.5)
            return skewness
        
        return 0
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return list(self.features.keys())
    
    @property
    def ready(self) -> bool:
        """Whether extractor is ready."""
        return len(self.pv_history) >= 10
    
    def reset(self) -> None:
        """Reset extractor state."""
        self.features.clear()
        self.pv_history.clear()


class OrderFlowExtractor:
    """
    Extracts order flow features.
    
    Analyzes buying/selling pressure and order flow dynamics.
    """
    
    def __init__(self, window: int = 50, name: str = "order_flow_features"):
        self.name = name
        self.window = window
        self.features: Dict[str, float] = {}
        
        # Order flow history
        self.flow_history: Deque[Dict[str, Any]] = deque(maxlen=window)
        self.delta_history: Deque[float] = deque(maxlen=window)
        
        # Capabilities
        self._events = None
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract order flow features."""
        features = {}
        
        # Calculate order flow delta
        volume = data.get('volume', 0)
        close = data.get('close', 0)
        open_price = data.get('open', close)
        high = data.get('high', close)
        low = data.get('low', close)
        
        # Estimate buying/selling volume
        if high > low:
            # Use close position to estimate buy/sell ratio
            buy_ratio = (close - low) / (high - low)
            buy_volume = volume * buy_ratio
            sell_volume = volume * (1 - buy_ratio)
        else:
            buy_volume = volume / 2
            sell_volume = volume / 2
        
        delta = buy_volume - sell_volume
        self.delta_history.append(delta)
        
        flow_data = {
            'timestamp': data.get('timestamp', datetime.now()),
            'delta': delta,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'price': close
        }
        self.flow_history.append(flow_data)
        
        # Current delta
        features['delta'] = delta
        features['delta_normalized'] = delta / volume if volume > 0 else 0
        
        # Cumulative delta
        features['cumulative_delta'] = sum(self.delta_history)
        
        # Delta momentum
        if len(self.delta_history) >= 5:
            recent_delta = list(self.delta_history)[-5:]
            features['delta_momentum'] = sum(recent_delta) / 5
            features['delta_acceleration'] = recent_delta[-1] - recent_delta[0]
        else:
            features['delta_momentum'] = delta
            features['delta_acceleration'] = 0
        
        # Buy/sell pressure
        total_buy = sum(f['buy_volume'] for f in self.flow_history)
        total_sell = sum(f['sell_volume'] for f in self.flow_history)
        total_volume = total_buy + total_sell
        
        if total_volume > 0:
            features['buy_pressure'] = total_buy / total_volume
            features['sell_pressure'] = total_sell / total_volume
            features['pressure_ratio'] = total_buy / total_sell if total_sell > 0 else 2.0
        else:
            features['buy_pressure'] = 0.5
            features['sell_pressure'] = 0.5
            features['pressure_ratio'] = 1.0
        
        # Aggressive vs passive flow (based on price movement)
        if len(self.flow_history) >= 2:
            aggressive_buy = 0
            aggressive_sell = 0
            
            for i in range(1, len(self.flow_history)):
                curr = self.flow_history[i]
                prev = self.flow_history[i-1]
                
                if curr['price'] > prev['price']:
                    aggressive_buy += curr['buy_volume']
                elif curr['price'] < prev['price']:
                    aggressive_sell += curr['sell_volume']
            
            total_aggressive = aggressive_buy + aggressive_sell
            if total_aggressive > 0:
                features['aggressive_buy_ratio'] = aggressive_buy / total_aggressive
                features['aggressive_sell_ratio'] = aggressive_sell / total_aggressive
            else:
                features['aggressive_buy_ratio'] = 0.5
                features['aggressive_sell_ratio'] = 0.5
        
        self.features = features
        return features
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return list(self.features.keys())
    
    @property
    def ready(self) -> bool:
        """Whether extractor is ready."""
        return len(self.flow_history) > 0
    
    def reset(self) -> None:
        """Reset extractor state."""
        self.features.clear()
        self.flow_history.clear()
        self.delta_history.clear()