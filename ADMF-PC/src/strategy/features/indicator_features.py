"""
Indicator-based feature extractors.
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import logging

from ...core.events import Event
from ..protocols import FeatureExtractor


logger = logging.getLogger(__name__)


class IndicatorFeatureExtractor:
    """
    Extracts features from indicator values.
    
    Subscribes to INDICATOR_UPDATE events and extracts features
    based on indicator values and their relationships.
    """
    
    def __init__(self, 
                 indicator_names: List[str],
                 name: str = "indicator_features"):
        self.name = name
        self.indicator_names = set(indicator_names)
        self.features: Dict[str, float] = {}
        self.indicator_values: Dict[str, float] = {}
        
        # Capabilities
        self._events = None
    
    def setup_subscriptions(self) -> None:
        """Subscribe to indicator updates."""
        if self._events:
            self._events.subscribe('INDICATOR_UPDATE', self.on_indicator_update)
    
    def on_indicator_update(self, event: Event) -> None:
        """Process indicator update."""
        data = event.payload
        indicators = data.get('indicators', {})
        
        # Update stored values
        for name in self.indicator_names:
            if name in indicators:
                self.indicator_values[name] = indicators[name]
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from current indicator values."""
        features = {}
        
        # Direct indicator values
        for name, value in self.indicator_values.items():
            features[f'ind_{name}'] = value
        
        # Indicator crossovers (if we have MAs)
        ma_indicators = [(name, value) for name, value in self.indicator_values.items() 
                        if 'MA' in name]
        
        for i, (name1, val1) in enumerate(ma_indicators):
            for name2, val2 in ma_indicators[i+1:]:
                if val2 > 0:
                    features[f'ratio_{name1}_{name2}'] = val1 / val2
                    features[f'diff_{name1}_{name2}'] = val1 - val2
        
        # RSI-based features
        if 'RSI_14' in self.indicator_values:
            rsi = self.indicator_values['RSI_14']
            features['rsi_overbought'] = 1.0 if rsi > 70 else 0.0
            features['rsi_oversold'] = 1.0 if rsi < 30 else 0.0
            features['rsi_neutral'] = 1.0 if 30 <= rsi <= 70 else 0.0
        
        # MACD-based features
        if 'MACD' in self.indicator_values:
            macd = self.indicator_values['MACD']
            features['macd_positive'] = 1.0 if macd > 0 else 0.0
            features['macd_strength'] = abs(macd)
        
        self.features = features
        return features
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return list(self.features.keys())
    
    @property
    def ready(self) -> bool:
        """Whether extractor is ready."""
        # Ready when we have at least some indicators
        return len(self.indicator_values) > 0
    
    def reset(self) -> None:
        """Reset extractor state."""
        self.features.clear()
        self.indicator_values.clear()


class TechnicalFeatureExtractor:
    """
    Extracts advanced technical analysis features.
    
    Combines multiple indicators to create composite features
    that capture market conditions.
    """
    
    def __init__(self, name: str = "technical_features"):
        self.name = name
        self.features: Dict[str, float] = {}
        self.indicator_values: Dict[str, float] = {}
        self.price_data: Optional[Dict[str, Any]] = None
        
        # Capabilities
        self._events = None
    
    def setup_subscriptions(self) -> None:
        """Subscribe to events."""
        if self._events:
            self._events.subscribe('INDICATOR_UPDATE', self.on_indicator_update)
            self._events.subscribe('BAR', self.on_bar)
    
    def on_indicator_update(self, event: Event) -> None:
        """Process indicator update."""
        data = event.payload
        self.indicator_values = data.get('all_indicators', {}).copy()
    
    def on_bar(self, event: Event) -> None:
        """Process bar data."""
        self.price_data = event.payload
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract technical features."""
        features = {}
        
        # Trend alignment
        features.update(self._extract_trend_features())
        
        # Momentum features
        features.update(self._extract_momentum_features())
        
        # Volatility features
        features.update(self._extract_volatility_features())
        
        # Support/Resistance features
        features.update(self._extract_sr_features())
        
        self.features = features
        return features
    
    def _extract_trend_features(self) -> Dict[str, float]:
        """Extract trend-related features."""
        features = {}
        
        # MA alignment
        ma_values = [(name, value) for name, value in self.indicator_values.items() 
                     if name.startswith('MA_')]
        ma_values.sort(key=lambda x: int(x[0].split('_')[1]))
        
        if len(ma_values) >= 2:
            # Check if MAs are aligned (bullish or bearish)
            aligned_up = all(ma_values[i][1] < ma_values[i+1][1] 
                            for i in range(len(ma_values)-1))
            aligned_down = all(ma_values[i][1] > ma_values[i+1][1] 
                              for i in range(len(ma_values)-1))
            
            features['ma_aligned_bullish'] = 1.0 if aligned_up else 0.0
            features['ma_aligned_bearish'] = 1.0 if aligned_down else 0.0
            features['ma_aligned'] = 1.0 if aligned_up or aligned_down else 0.0
        
        # Price vs MAs
        if self.price_data:
            close = self.price_data.get('close', 0)
            above_ma_count = sum(1 for name, value in ma_values if close > value)
            features['price_above_ma_pct'] = above_ma_count / len(ma_values) if ma_values else 0
        
        return features
    
    def _extract_momentum_features(self) -> Dict[str, float]:
        """Extract momentum-related features."""
        features = {}
        
        # RSI momentum
        if 'RSI_14' in self.indicator_values:
            rsi = self.indicator_values['RSI_14']
            features['rsi_momentum'] = (rsi - 50) / 50  # Normalized to [-1, 1]
        
        # MACD momentum
        if 'MACD' in self.indicator_values:
            macd = self.indicator_values['MACD']
            # Normalize MACD (rough approximation)
            features['macd_momentum'] = max(-1, min(1, macd / 0.01))
        
        # Combined momentum
        momentum_values = []
        if 'rsi_momentum' in features:
            momentum_values.append(features['rsi_momentum'])
        if 'macd_momentum' in features:
            momentum_values.append(features['macd_momentum'])
        
        if momentum_values:
            features['combined_momentum'] = sum(momentum_values) / len(momentum_values)
        
        return features
    
    def _extract_volatility_features(self) -> Dict[str, float]:
        """Extract volatility-related features."""
        features = {}
        
        # ATR-based features
        if 'ATR_14' in self.indicator_values and self.price_data:
            atr = self.indicator_values['ATR_14']
            close = self.price_data.get('close', 1)
            
            # Normalized ATR
            features['atr_normalized'] = atr / close if close > 0 else 0
            
            # Volatility regime
            if 'ATR_14' in self.indicator_values and 'ATR_50' in self.indicator_values:
                atr_ratio = atr / self.indicator_values['ATR_50']
                features['volatility_expanding'] = 1.0 if atr_ratio > 1.2 else 0.0
                features['volatility_contracting'] = 1.0 if atr_ratio < 0.8 else 0.0
        
        return features
    
    def _extract_sr_features(self) -> Dict[str, float]:
        """Extract support/resistance features."""
        features = {}
        
        if not self.price_data:
            return features
        
        close = self.price_data.get('close', 0)
        high = self.price_data.get('high', close)
        low = self.price_data.get('low', close)
        
        # Distance from key MAs (potential support/resistance)
        for ma_name, ma_value in self.indicator_values.items():
            if ma_name.startswith('MA_') and ma_value > 0:
                distance_pct = (close - ma_value) / ma_value
                features[f'distance_from_{ma_name}'] = distance_pct
                
                # Near support/resistance
                features[f'near_{ma_name}'] = 1.0 if abs(distance_pct) < 0.01 else 0.0
        
        return features
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return list(self.features.keys())
    
    @property
    def ready(self) -> bool:
        """Whether extractor is ready."""
        return len(self.indicator_values) > 0 and self.price_data is not None
    
    def reset(self) -> None:
        """Reset extractor state."""
        self.features.clear()
        self.indicator_values.clear()
        self.price_data = None


class CompositeFeatureExtractor:
    """
    Combines multiple feature extractors.
    """
    
    def __init__(self, extractors: List[FeatureExtractor], name: str = "composite_features"):
        self.name = name
        self.extractors = extractors
        self.features: Dict[str, float] = {}
        
        # Capabilities
        self._events = None
        self._lifecycle = None
    
    def setup_subscriptions(self) -> None:
        """Set up subscriptions for all extractors."""
        for extractor in self.extractors:
            if hasattr(extractor, 'setup_subscriptions'):
                extractor.setup_subscriptions()
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from all extractors."""
        all_features = {}
        
        for extractor in self.extractors:
            try:
                features = extractor.extract(data)
                # Prefix features with extractor name to avoid conflicts
                for fname, fvalue in features.items():
                    all_features[f"{extractor.name}_{fname}"] = fvalue
            except Exception as e:
                logger.error(f"Error in extractor {extractor.name}: {e}")
        
        self.features = all_features
        return all_features
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return list(self.features.keys())
    
    @property
    def ready(self) -> bool:
        """Whether all extractors are ready."""
        return all(extractor.ready for extractor in self.extractors)
    
    def reset(self) -> None:
        """Reset all extractors."""
        self.features.clear()
        for extractor in self.extractors:
            if hasattr(extractor, 'reset'):
                extractor.reset()