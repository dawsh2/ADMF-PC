"""
Stateful Feature Management - FeatureHub.

The FeatureHub is the Tier 2 stateful component that manages incremental 
feature computation, allowing strategies to remain completely stateless.

This is THE canonical FeatureHub implementation using O(1) incremental updates.
Uses protocol + composition architecture - no inheritance.
"""

from typing import Optional, Dict, List, Any, Set, Tuple
import logging
from collections import defaultdict

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
        
        # Dependency tracking
        self._feature_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._computation_order: Dict[str, List[str]] = {}  # Cache computation order per symbol
        
        logger.debug("FeatureHub initialized with INCREMENTAL mode (O(1) updates)")
        logger.debug("FeatureHub initialized for symbols: %s", self.symbols)
    
    def configure_features(self, feature_configs: Dict[str, Dict[str, Any]], 
                          required_features: Optional[List[str]] = None) -> None:
        """
        Configure which features to compute.
        
        Args:
            feature_configs: Dict mapping feature names to their configurations
            required_features: Optional list of features actually needed (for optimization)
            
        Example:
            {
                "sma_20": {"type": "sma", "period": 20},
                "rsi": {"type": "rsi", "period": 14},
                "bollinger": {"type": "bollinger", "period": 20, "std_dev": 2.0}
            }
        """
        # If required_features is specified, only configure those
        if required_features:
            original_count = len(feature_configs)
            filtered_configs = {
                name: config 
                for name, config in feature_configs.items() 
                if name in required_features
            }
            
            if len(filtered_configs) < original_count:
                logger.info(f"🎯 Feature optimization: configuring {len(filtered_configs)} of "
                           f"{original_count} features (skipping {original_count - len(filtered_configs)} unused)")
                
            self._feature_configs = filtered_configs
        else:
            self._feature_configs = feature_configs
            
        logger.debug("FeatureHub configured with features: %s", list(self._feature_configs.keys()))
        
        # Log bollinger bands features specifically
        bollinger_features = [name for name in self._feature_configs.keys() if 'bollinger' in name]
        if bollinger_features:
            logger.info(f"Bollinger bands features configured: {len(bollinger_features)} total")
            # Log first 5 and last 5 for debugging
            if len(bollinger_features) > 10:
                logger.info(f"  First 5: {bollinger_features[:5]}")
                logger.info(f"  Last 5: {bollinger_features[-5:]}")
            else:
                logger.info(f"  All: {bollinger_features}")
    
    def _create_feature(self, feature_name: str, config: Dict[str, Any]) -> Any:
        """Create a feature instance based on configuration."""
        feature_type = config.get("type", "")
        
        # Debug logging
        if feature_type == "standard":
            logger.error(f"Feature {feature_name} has type 'standard' - config: {config}")
        
        if feature_type not in FEATURE_REGISTRY:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        feature_class = FEATURE_REGISTRY[feature_type]
        
        # Extract parameters (remove 'type' key)
        # Handle both formats: {type: X, params: {...}} and {type: X, ...params}
        if 'params' in config:
            # New format with nested params
            params = config['params'].copy()
        else:
            # Old format with params at top level
            params = {k: v for k, v in config.items() if k not in ["type", "component"]}
        
        params["name"] = feature_name
        
        # Create feature instance
        feature = feature_class(**params)
        
        # Check if feature declares dependencies
        if hasattr(feature, 'dependencies'):
            deps = feature.dependencies
            if isinstance(deps, property):
                # If it's a property, call it
                deps = deps.fget(feature)
            self._feature_dependencies[feature_name] = set(deps) if deps else set()
            logger.debug(f"Feature {feature_name} declares dependencies: {deps}")
        
        return feature
    
    def _topological_sort(self, features: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """
        Sort features by dependencies using topological sort.
        
        Returns features in order such that dependencies are computed first.
        """
        # Build dependency graph
        in_degree = defaultdict(int)
        adj_list = defaultdict(list)
        
        for name in features:
            deps = self._feature_dependencies.get(name, set())
            # Map simple dependency names to actual feature names
            active_deps = set()
            for dep in deps:
                # Find features that start with the dependency name
                for feature_name in features.keys():
                    if feature_name.startswith(dep + '_') or feature_name == dep:
                        active_deps.add(feature_name)
            
            in_degree[name] = len(active_deps)
            for dep in active_deps:
                adj_list[dep].append(name)
        
        # Find all nodes with no dependencies
        queue = [name for name in features if in_degree[name] == 0]
        sorted_features = []
        
        while queue:
            current = queue.pop(0)
            sorted_features.append((current, features[current]))
            
            # Reduce in-degree for dependent features
            for dependent in adj_list[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for cycles
        if len(sorted_features) != len(features):
            logger.error("Cycle detected in feature dependencies!")
            # Return features in original order as fallback
            return list(features.items())
            
        return sorted_features
    
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
        for key in ['open', 'high', 'low', 'close', 'volume', 'timestamp']:
            if key in bar:
                results[key] = bar[key]
        
        # Sort features by dependencies
        sorted_features = self._topological_sort(self._features[symbol])
        
        # Debug: log the sort order once
        if symbol not in getattr(self, '_logged_sort_order', set()):
            if not hasattr(self, '_logged_sort_order'):
                self._logged_sort_order = set()
            self._logged_sort_order.add(symbol)
            logger.info(f"Feature computation order for {symbol}: {[name for name, _ in sorted_features]}")
        
        for name, feature in sorted_features:
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
                elif hasattr(feature, 'update_with_features'):
                    # Feature declares it needs access to computed features
                    # Merge bar data and computed features
                    all_data = {**bar, **results}
                    value = feature.update_with_features(all_data)
                elif name in self._feature_dependencies and self._feature_dependencies[name]:
                    # Feature has dependencies - pass computed features in kwargs
                    # Collect all computed dependency values
                    dep_values = {}
                    for dep in self._feature_dependencies[name]:
                        if dep in results:
                            dep_values[dep] = results[dep]
                    
                    # Update with dependencies
                    value = feature.update(
                        price=bar.get("close", 0),
                        high=bar.get("high"),
                        low=bar.get("low"),
                        volume=bar.get("volume"),
                        **dep_values
                    )
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
                        # Check if a specific component was requested in the config
                        feature_config = self._feature_configs.get(name, {})
                        component = feature_config.get('component')
                        
                        if component and component in value:
                            # Store just the requested component value under the canonical name
                            results[name] = value[component]
                        else:
                            # Store all components with sub-keys
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