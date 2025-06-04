"""
Stateless market classifiers for unified architecture.

These classifiers implement the StatelessClassifier protocol for use as 
lightweight services in the event-driven architecture. All state is passed 
as parameters - no internal state is maintained.
"""

from typing import Dict, Any, List
from enum import Enum

from ...core.components.protocols import StatelessClassifier


class MarketRegime(Enum):
    """Common market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class StatelessTrendClassifier:
    """
    Stateless trend classifier for unified architecture.
    
    Classifies market trend based on moving average relationships
    without maintaining any internal state.
    """
    
    def __init__(self):
        """Initialize stateless trend classifier."""
        # No configuration stored - everything comes from params
        pass
    
    def classify_regime(
        self,
        features: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify market trend from features.
        
        Args:
            features: Calculated indicators from FeatureHub
                - sma_fast: Fast moving average
                - sma_slow: Slow moving average
            params: Classifier parameters
                - trend_threshold: Min MA difference for trend (default: 0.02)
                
        Returns:
            Regime dict with classification and confidence
        """
        # Extract parameters with defaults
        trend_threshold = params.get('trend_threshold', 0.02)
        
        # Get required features
        fast_ma = features.get('sma_fast')
        slow_ma = features.get('sma_slow')
        
        # Return default if features missing
        if fast_ma is None or slow_ma is None or slow_ma == 0:
            return {
                'regime': MarketRegime.RANGING.value,
                'confidence': 0.0,
                'metadata': {'reason': 'Missing required features'}
            }
        
        # Calculate trend strength
        ma_diff = (fast_ma - slow_ma) / slow_ma
        
        # Classify based on MA relationship
        if ma_diff > trend_threshold:
            return {
                'regime': MarketRegime.TRENDING_UP.value,
                'confidence': min(ma_diff / (trend_threshold * 2), 1.0),
                'metadata': {
                    'ma_diff': ma_diff,
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma,
                    'reason': 'Fast MA above slow MA'
                }
            }
        elif ma_diff < -trend_threshold:
            return {
                'regime': MarketRegime.TRENDING_DOWN.value,
                'confidence': min(abs(ma_diff) / (trend_threshold * 2), 1.0),
                'metadata': {
                    'ma_diff': ma_diff,
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma,
                    'reason': 'Fast MA below slow MA'
                }
            }
        else:
            return {
                'regime': MarketRegime.RANGING.value,
                'confidence': 1.0 - (abs(ma_diff) / trend_threshold),
                'metadata': {
                    'ma_diff': ma_diff,
                    'reason': 'No clear trend'
                }
            }
    
    @property
    def required_features(self) -> List[str]:
        """List of feature names this classifier requires."""
        return ['sma_fast', 'sma_slow']


class StatelessVolatilityClassifier:
    """
    Stateless volatility classifier for unified architecture.
    
    Classifies market volatility based on ATR or standard deviation
    without maintaining any internal state.
    """
    
    def __init__(self):
        """Initialize stateless volatility classifier."""
        # No configuration stored - everything comes from params
        pass
    
    def classify_regime(
        self,
        features: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify market volatility from features.
        
        Args:
            features: Calculated indicators from FeatureHub
                - atr: Average True Range
                - atr_sma: Moving average of ATR
                - volatility: Standard deviation
            params: Classifier parameters
                - high_vol_threshold: Multiplier for high vol (default: 1.5)
                - low_vol_threshold: Multiplier for low vol (default: 0.5)
                
        Returns:
            Regime dict with classification and confidence
        """
        # Extract parameters with defaults
        high_vol_threshold = params.get('high_vol_threshold', 1.5)
        low_vol_threshold = params.get('low_vol_threshold', 0.5)
        
        # Get volatility measure (prefer ATR if available)
        current_vol = features.get('atr') or features.get('volatility')
        avg_vol = features.get('atr_sma') or features.get('volatility_sma')
        
        # Return default if features missing
        if current_vol is None or avg_vol is None or avg_vol == 0:
            return {
                'regime': MarketRegime.RANGING.value,
                'confidence': 0.0,
                'metadata': {'reason': 'Missing volatility features'}
            }
        
        # Calculate volatility ratio
        vol_ratio = current_vol / avg_vol
        
        # Classify based on volatility level
        if vol_ratio > high_vol_threshold:
            return {
                'regime': MarketRegime.HIGH_VOLATILITY.value,
                'confidence': min((vol_ratio - 1.0) / (high_vol_threshold - 1.0), 1.0),
                'metadata': {
                    'vol_ratio': vol_ratio,
                    'current_vol': current_vol,
                    'avg_vol': avg_vol,
                    'reason': 'Volatility above average'
                }
            }
        elif vol_ratio < low_vol_threshold:
            return {
                'regime': MarketRegime.LOW_VOLATILITY.value,
                'confidence': min((1.0 - vol_ratio) / (1.0 - low_vol_threshold), 1.0),
                'metadata': {
                    'vol_ratio': vol_ratio,
                    'current_vol': current_vol,
                    'avg_vol': avg_vol,
                    'reason': 'Volatility below average'
                }
            }
        else:
            return {
                'regime': MarketRegime.RANGING.value,
                'confidence': 0.5,
                'metadata': {
                    'vol_ratio': vol_ratio,
                    'reason': 'Normal volatility'
                }
            }
    
    @property
    def required_features(self) -> List[str]:
        """List of feature names this classifier requires."""
        return ['atr', 'atr_sma']  # Or volatility, volatility_sma


class StatelessCompositeClassifier:
    """
    Stateless composite classifier that combines multiple classifiers.
    
    This demonstrates how different classifiers can work together
    in a stateless manner.
    """
    
    def __init__(self):
        """Initialize stateless composite classifier."""
        # Create sub-classifiers
        self.trend_classifier = StatelessTrendClassifier()
        self.volatility_classifier = StatelessVolatilityClassifier()
    
    def classify_regime(
        self,
        features: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify market using multiple sub-classifiers.
        
        Args:
            features: Calculated indicators from FeatureHub
            params: Classifier parameters for all sub-classifiers
                
        Returns:
            Regime dict with combined classification
        """
        # Get sub-classifications
        trend_result = self.trend_classifier.classify_regime(
            features, 
            params.get('trend_params', {})
        )
        
        vol_result = self.volatility_classifier.classify_regime(
            features,
            params.get('volatility_params', {})
        )
        
        # Combine classifications
        trend_regime = trend_result['regime']
        vol_regime = vol_result['regime']
        
        # Create composite regime string
        if trend_regime in ['trending_up', 'trending_down']:
            if vol_regime == 'high_volatility':
                composite_regime = f"{trend_regime}_volatile"
            else:
                composite_regime = trend_regime
        else:
            composite_regime = vol_regime
        
        # Average confidence
        avg_confidence = (trend_result['confidence'] + vol_result['confidence']) / 2
        
        return {
            'regime': composite_regime,
            'confidence': avg_confidence,
            'metadata': {
                'trend': trend_result,
                'volatility': vol_result,
                'reason': 'Composite classification'
            }
        }
    
    @property
    def required_features(self) -> List[str]:
        """List of feature names this classifier requires."""
        # Combine requirements from sub-classifiers
        features = []
        features.extend(self.trend_classifier.required_features)
        features.extend(self.volatility_classifier.required_features)
        return list(set(features))  # Remove duplicates


# Factory functions
def create_stateless_trend_classifier() -> StatelessTrendClassifier:
    """Create a stateless trend classifier instance."""
    return StatelessTrendClassifier()


def create_stateless_volatility_classifier() -> StatelessVolatilityClassifier:
    """Create a stateless volatility classifier instance."""
    return StatelessVolatilityClassifier()


def create_stateless_composite_classifier() -> StatelessCompositeClassifier:
    """Create a stateless composite classifier instance."""
    return StatelessCompositeClassifier()