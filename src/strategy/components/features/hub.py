"""
Stateful Feature Management - FeatureHub.

The FeatureHub is the Tier 2 stateful component that manages incremental 
feature computation, allowing strategies to remain completely stateless.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Set, List, Deque
from datetime import datetime
from collections import deque, defaultdict
import logging

# Import all feature functions for computation
from .trend import sma_feature, ema_feature, dema_feature, tema_feature
from .oscillators import rsi_feature, stochastic_feature, williams_r_feature, cci_feature
from .momentum import macd_feature, adx_feature, momentum_feature, vortex_feature
from .volatility import atr_feature, bollinger_bands_feature, keltner_channel_feature, donchian_channel_feature, volatility_feature
from .volume import volume_feature, volume_sma_feature, volume_ratio_feature
from .complex import ichimoku_feature
from .price import high_feature, low_feature, atr_sma_feature, volatility_sma_feature

logger = logging.getLogger(__name__)


# Feature registry for easy discovery and dynamic creation
FEATURE_REGISTRY = {
    # Trend features
    "sma": sma_feature,
    "ema": ema_feature,
    "dema": dema_feature,
    "tema": tema_feature,
    
    # Oscillator features
    "rsi": rsi_feature,
    "stochastic": stochastic_feature,
    "williams_r": williams_r_feature,
    "cci": cci_feature,
    
    # Momentum features
    "macd": macd_feature,
    "adx": adx_feature,
    "momentum": momentum_feature,
    "vortex": vortex_feature,
    
    # Volatility features
    "atr": atr_feature,
    "bollinger_bands": bollinger_bands_feature,
    "bollinger": bollinger_bands_feature,  # Alias for bollinger_bands
    "keltner_channel": keltner_channel_feature,
    "donchian_channel": donchian_channel_feature,
    "volatility": volatility_feature,
    
    # Volume features
    "volume": volume_feature,
    "volume_sma": volume_sma_feature,
    "volume_ratio": volume_ratio_feature,
    
    # Price features
    "high": high_feature,
    "low": low_feature,
    "atr_sma": atr_sma_feature,
    "volatility_sma": volatility_sma_feature,
    
    # Complex features
    "ichimoku": ichimoku_feature
}


def compute_feature(feature_name: str, data: Union[pd.DataFrame, pd.Series], **kwargs) -> Union[pd.Series, Dict[str, pd.Series]]:
    """
    Compute a single feature by name.
    
    Pure function dispatcher for feature calculation.
    
    Args:
        feature_name: Name of feature to compute
        data: Price data (DataFrame with OHLCV or Series)
        **kwargs: Additional parameters for feature calculation
        
    Returns:
        Feature values as Series or Dict of Series
    """
    if feature_name not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature: {feature_name}")
    
    feature_func = FEATURE_REGISTRY[feature_name]
    
    # Handle different data input types
    if isinstance(data, pd.DataFrame):
        # Extract common price series from DataFrame
        if feature_name in ["sma", "ema", "dema", "tema", "rsi", "momentum", "volatility"]:
            return feature_func(data["close"], **kwargs)
        elif feature_name in ["macd", "bollinger_bands", "bollinger"]:
            return feature_func(data["close"], **kwargs)
        elif feature_name in ["atr", "stochastic", "adx", "williams_r", "cci", "vortex", "keltner_channel", "donchian_channel", "ichimoku"]:
            return feature_func(data["high"], data["low"], data["close"], **kwargs)
        elif feature_name in ["volume", "volume_sma"]:
            if feature_name == "volume":
                return feature_func(data["volume"], data["close"], **kwargs)
            else:
                return feature_func(data["volume"], **kwargs)
        elif feature_name == "volume_ratio":
            return feature_func(data["volume"], **kwargs)
        elif feature_name in ["high", "low"]:
            return feature_func(data, **kwargs)
        elif feature_name in ["atr_sma", "volatility_sma"]:
            return feature_func(data, **kwargs)
    else:
        # Single series input
        if feature_name in ["sma", "ema", "dema", "tema", "rsi", "momentum", "volatility"]:
            return feature_func(data, **kwargs)
        else:
            raise ValueError(f"Feature {feature_name} requires OHLC data, not single series")
    
    return feature_func(data, **kwargs)


def compute_multiple_features(feature_configs: Dict[str, Dict[str, Any]], 
                            data: pd.DataFrame) -> Dict[str, Union[pd.Series, Dict[str, pd.Series]]]:
    """
    Compute multiple features in one call.
    
    Pure function for batch feature computation.
    
    Args:
        feature_configs: Dict mapping feature names to their configurations
        data: OHLCV DataFrame
        
    Returns:
        Dict mapping feature names to their computed values
    
    Example:
        configs = {
            "sma_20": {"feature": "sma", "period": 20},
            "rsi": {"feature": "rsi", "period": 14},
            "macd": {"feature": "macd", "fast": 12, "slow": 26, "signal": 9}
        }
        features = compute_multiple_features(configs, data)
    """
    results = {}
    
    for name, config in feature_configs.items():
        feature_type = config.pop("feature")
        results[name] = compute_feature(feature_type, data, **config)
    
    return results


class FeatureHub:
    """
    Centralized stateful feature computation engine.
    
    This Tier 2 component manages ALL incremental feature calculation,
    allowing strategies to remain completely stateless.
    
    Key responsibilities:
    - Maintain rolling windows for all features
    - Provide incremental updates for streaming data
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
        
        # Stateful data storage - rolling windows per symbol
        self.price_data: Dict[str, Dict[str, Deque[float]]] = defaultdict(lambda: {
            'open': deque(maxlen=1000),
            'high': deque(maxlen=1000), 
            'low': deque(maxlen=1000),
            'close': deque(maxlen=1000),
            'volume': deque(maxlen=1000)
        })
        
        # Feature cache - computed feature values per symbol
        self.feature_cache: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Feature configurations - what features to compute
        self.feature_configs: Dict[str, Dict[str, Any]] = {}
        
        # State tracking
        self.bar_count: Dict[str, int] = defaultdict(int)
        
        logger.info("FeatureHub initialized for symbols: %s", self.symbols)
    
    def configure_features(self, feature_configs: Dict[str, Dict[str, Any]]) -> None:
        """
        Configure which features to compute.
        
        Args:
            feature_configs: Dict mapping feature names to their configurations
            
        Example:
            {
                "sma_20": {"feature": "sma", "period": 20},
                "rsi": {"feature": "rsi", "period": 14},
                "bollinger": {"feature": "bollinger_bands", "period": 20, "std_dev": 2.0}
            }
        """
        self.feature_configs = feature_configs
        logger.info("FeatureHub configured with features: %s", list(feature_configs.keys()))
    
    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to track."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.info("Added symbol to FeatureHub: %s", symbol)
    
    def update_bar(self, symbol: str, bar: Dict[str, float]) -> None:
        """
        Update with new bar data for incremental feature calculation.
        
        Args:
            symbol: Symbol to update
            bar: Bar data with open, high, low, close, volume
        """
        # Store new bar data
        for field in ['open', 'high', 'low', 'close', 'volume']:
            if field in bar:
                self.price_data[symbol][field].append(bar[field])
        
        self.bar_count[symbol] += 1
        
        # Recompute features incrementally
        self._update_features(symbol)
        
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
        return self.bar_count.get(symbol, 0) >= min_bars
    
    def reset(self, symbol: Optional[str] = None) -> None:
        """
        Reset feature computation state.
        
        Args:
            symbol: Symbol to reset (None for all symbols)
        """
        if symbol:
            # Reset specific symbol
            for field in self.price_data[symbol]:
                self.price_data[symbol][field].clear()
            self.feature_cache[symbol].clear()
            self.bar_count[symbol] = 0
            logger.info("Reset FeatureHub state for symbol: %s", symbol)
        else:
            # Reset all symbols
            self.price_data.clear()
            self.feature_cache.clear()
            self.bar_count.clear()
            logger.info("Reset FeatureHub state for all symbols")
    
    def _update_features(self, symbol: str) -> None:
        """
        Update features for a symbol using current data.
        
        This converts the rolling window data to pandas and computes
        features using the stateless feature functions.
        """
        # Convert deques to pandas DataFrame for feature computation
        data_dict = {}
        for field, deque_data in self.price_data[symbol].items():
            if len(deque_data) > 0:
                data_dict[field] = list(deque_data)
        
        if not data_dict or len(data_dict['close']) < 2:
            return
        
        # Create DataFrame for feature computation
        df = pd.DataFrame(data_dict)
        
        # Compute all configured features
        symbol_features = {}
        
        for feature_name, config in self.feature_configs.items():
            try:
                feature_type = config.get('feature')
                if feature_type not in FEATURE_REGISTRY:
                    logger.warning("Unknown feature type: %s", feature_type)
                    continue
                
                # Create config copy without 'feature' key
                feature_params = {k: v for k, v in config.items() if k != 'feature'}
                
                # Compute feature using stateless function
                result = compute_feature(feature_type, df, **feature_params)
                
                # Store latest feature value(s)
                if isinstance(result, dict):
                    # Multi-value features (e.g., MACD, Bollinger Bands)
                    for sub_name, series in result.items():
                        if len(series) > 0 and not pd.isna(series.iloc[-1]):
                            symbol_features[f"{feature_name}_{sub_name}"] = float(series.iloc[-1])
                else:
                    # Single-value features (e.g., SMA, RSI)
                    if len(result) > 0 and not pd.isna(result.iloc[-1]):
                        symbol_features[feature_name] = float(result.iloc[-1])
                        
            except Exception as e:
                logger.error("Error computing feature %s for %s: %s", feature_name, symbol, e)
        
        # Update cache
        self.feature_cache[symbol].update(symbol_features)
        
        logger.debug("Updated %d features for %s", len(symbol_features), symbol)
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of FeatureHub state.
        
        Returns:
            Dict with summary information
        """
        return {
            "symbols": len(self.symbols),
            "configured_features": len(self.feature_configs),
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
    "sma_fast": {"feature": "sma", "period": 10},
    "sma_slow": {"feature": "sma", "period": 20},
    "rsi": {"feature": "rsi", "period": 14},
    "macd": {"feature": "macd", "fast": 12, "slow": 26, "signal": 9}
}

DEFAULT_MEAN_REVERSION_FEATURES = {
    "bollinger": {"feature": "bollinger_bands", "period": 20, "std_dev": 2.0},
    "rsi": {"feature": "rsi", "period": 14},
    "stochastic": {"feature": "stochastic", "k_period": 14, "d_period": 3}
}

DEFAULT_VOLATILITY_FEATURES = {
    "atr": {"feature": "atr", "period": 14},
    "bollinger": {"feature": "bollinger_bands", "period": 20, "std_dev": 2.0},
    "adx": {"feature": "adx", "period": 14}
}