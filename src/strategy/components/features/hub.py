"""
Stateful Feature Management - FeatureHub.

The FeatureHub is the Tier 2 stateful component that manages incremental 
feature computation, allowing strategies to remain completely stateless.

This is THE canonical FeatureHub implementation using O(1) incremental updates.
Uses protocol + composition architecture - no inheritance.
"""

from typing import Optional, Dict, List, Any
import logging

from .protocols import Feature, FeatureState
from .indicators import ALL_INDICATOR_FEATURES

logger = logging.getLogger(__name__)


# All features are now imported from organized modules


# Feature registry - consolidated from all indicator modules
FEATURE_REGISTRY = ALL_INDICATOR_FEATURES.copy()


class FeatureHub:
    """
    THE canonical FeatureHub implementation.
    
    Centralized stateful feature computation engine using O(1) incremental updates.
    This Tier 2 component manages ALL incremental feature calculation,
    allowing strategies to remain completely stateless.
    
    Key responsibilities:
    - Maintain rolling windows for all features
    - Provide O(1) incremental updates for streaming data
    - Cache computed features for efficiency
    - Manage dependencies between features
    """
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize FeatureHub.
        
        Args:
            symbols: List of symbols to track (optional)
        """
        self.symbols = symbols or []
        
        # Feature storage: symbol -> feature_name -> Feature
        self._features: Dict[str, Dict[str, Any]] = {}
        self._feature_configs: Dict[str, Dict[str, Any]] = {}
        
        # Feature cache - computed feature values per symbol
        self.feature_cache: Dict[str, Dict[str, Any]] = {}
        
        # State tracking
        self.bar_count: Dict[str, int] = {}
        
        logger.debug("FeatureHub initialized with INCREMENTAL mode (O(1) updates)")
        logger.debug("FeatureHub initialized for symbols: %s", self.symbols)
    
    def configure_features(self, feature_configs: Dict[str, Dict[str, Any]]) -> None:
        """
        Configure which features to compute.
        
        Args:
            feature_configs: Dict mapping feature names to their configurations
            
        Example:
            {
                "sma_20": {"type": "sma", "period": 20},
                "rsi": {"type": "rsi", "period": 14},
                "bollinger": {"type": "bollinger", "period": 20, "std_dev": 2.0}
            }
        """
        self._feature_configs = feature_configs
        logger.debug("FeatureHub configured with features: %s", list(feature_configs.keys()))
    
    def _create_feature(self, feature_name: str, config: Dict[str, Any]) -> Any:
        """Create a feature instance based on configuration."""
        feature_type = config.get("type", "")
        
        if feature_type not in FEATURE_REGISTRY:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        feature_class = FEATURE_REGISTRY[feature_type]
        
        # Extract parameters (remove 'type' key)
        params = {k: v for k, v in config.items() if k != "type"}
        params["name"] = feature_name
        
        return feature_class(**params)
    
    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to track."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.debug("Added symbol to FeatureHub: %s", symbol)
    
    def update_bar(self, symbol: str, bar: Dict[str, float]) -> None:
        """
        Update with new bar data for incremental feature calculation.
        
        Args:
            symbol: Symbol to update
            bar: Bar data with open, high, low, close, volume
        """
        self.bar_count[symbol] = self.bar_count.get(symbol, 0) + 1
        
        # Initialize features for symbol if needed
        if symbol not in self._features:
            self._features[symbol] = {}
            for name, config in self._feature_configs.items():
                self._features[symbol][name] = self._create_feature(name, config)
        
        # Update all features
        results = {}
        
        # Add raw OHLCV data to results first
        # This allows classifiers and strategies to access raw price data
        for key in ['open', 'high', 'low', 'close', 'volume']:
            if key in bar:
                results[key] = bar[key]
        
        for name, feature in self._features[symbol].items():
            try:
                # Special handling for ParabolicSAR which needs positional args
                if hasattr(feature, '__class__') and feature.__class__.__name__ == 'ParabolicSAR':
                    # ParabolicSAR requires high, low, close as positional arguments
                    high = bar.get("high")
                    low = bar.get("low")
                    close = bar.get("close", 0)
                    if high is not None and low is not None:
                        value = feature.update(high, low, close)
                    else:
                        logger.warning(f"ParabolicSAR {name} skipped - missing high/low data")
                        continue
                else:
                    # Standard keyword argument update for other features
                    value = feature.update(
                        price=bar.get("close", 0),
                        high=bar.get("high"),
                        low=bar.get("low"),
                        volume=bar.get("volume")
                    )
                
                if value is not None:
                    if isinstance(value, dict):
                        # Multi-value features
                        for sub_name, sub_value in value.items():
                            results[f"{name}_{sub_name}"] = sub_value
                    else:
                        # Single-value features
                        results[name] = value
                        
            except Exception as e:
                logger.error(f"Error updating feature {name} for {symbol}: {e}")
        
        self.feature_cache[symbol] = results
        logger.debug("Updated bar for %s, total bars: %d", symbol, self.bar_count[symbol])
    
    def get_features(self, symbol: str) -> Dict[str, Any]:
        """
        Get current feature values for a symbol.
        
        Args:
            symbol: Symbol to get features for
            
        Returns:
            Dict of feature_name -> feature_value
        """
        return self.feature_cache.get(symbol, {}).copy()
    
    def get_all_features(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current feature values for all symbols.
        
        Returns:
            Dict mapping symbol -> feature_dict
        """
        return {
            symbol: self.get_features(symbol)
            for symbol in self.symbols
        }
    
    def has_sufficient_data(self, symbol: str, min_bars: int = 50) -> bool:
        """
        Check if symbol has sufficient data for feature calculation.
        
        Args:
            symbol: Symbol to check
            min_bars: Minimum number of bars required
            
        Returns:
            True if sufficient data available
        """
        if symbol not in self._features:
            return False
        
        return all(f.is_ready for f in self._features[symbol].values())
    
    def reset(self, symbol: Optional[str] = None) -> None:
        """
        Reset feature computation state.
        
        Args:
            symbol: Symbol to reset (None for all symbols)
        """
        if symbol:
            if symbol in self._features:
                for feature in self._features[symbol].values():
                    feature.reset()
            self.feature_cache[symbol] = {}
            self.bar_count[symbol] = 0
            logger.debug("Reset FeatureHub state for symbol: %s", symbol)
        else:
            for symbol_features in self._features.values():
                for feature in symbol_features.values():
                    feature.reset()
            self.feature_cache.clear()
            self.bar_count.clear()
            logger.debug("Reset FeatureHub state for all symbols")
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of FeatureHub state.
        
        Returns:
            Dict with summary information
        """
        return {
            "symbols": len(self.symbols),
            "configured_features": len(self._feature_configs),
            "bar_counts": dict(self.bar_count),
            "feature_counts": {
                symbol: len(features)
                for symbol, features in self.feature_cache.items()
            }
        }


# Factory function for creating FeatureHub
def create_feature_hub(symbols: Optional[List[str]] = None,
                      feature_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> FeatureHub:
    """
    Factory function for creating FeatureHub instances.
    
    Args:
        symbols: List of symbols to track
        feature_configs: Feature configurations
        
    Returns:
        Configured FeatureHub instance
    """
    hub = FeatureHub(symbols)
    
    if feature_configs:
        hub.configure_features(feature_configs)
    
    return hub


# Default feature configurations for common strategies
DEFAULT_MOMENTUM_FEATURES = {
    "sma_fast": {"type": "sma", "period": 10},
    "sma_slow": {"type": "sma", "period": 20},
    "rsi": {"type": "rsi", "period": 14},
    "macd": {"type": "macd", "fast_period": 12, "slow_period": 26, "signal_period": 9}
}

DEFAULT_MEAN_REVERSION_FEATURES = {
    "bollinger": {"type": "bollinger", "period": 20, "std_dev": 2.0},
    "rsi": {"type": "rsi", "period": 14}
}

DEFAULT_VOLATILITY_FEATURES = {
    "atr": {"type": "atr", "period": 14},
    "bollinger": {"type": "bollinger", "period": 20, "std_dev": 2.0}
}

DEFAULT_STRUCTURE_FEATURES = {
    "pivot_points": {"type": "pivot_points"},
    "support_resistance": {"type": "support_resistance", "lookback": 50, "min_touches": 2},
    "swing_points": {"type": "swing_points", "lookback": 5}
}

DEFAULT_VOLUME_FEATURES = {
    "volume_sma": {"type": "volume_sma", "period": 20},
    "obv": {"type": "obv"},
    "vpt": {"type": "vpt"}
}