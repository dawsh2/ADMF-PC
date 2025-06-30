"""
Enhanced Feature Hub with dependency resolution.

This version properly handles features that depend on other features,
computing them in the correct order.
"""

from typing import Optional, Dict, List, Any, Set, Tuple
import logging
from collections import defaultdict

from .protocols import Feature, FeatureState
from .hub import FeatureHub

logger = logging.getLogger(__name__)


class DependentFeature:
    """Wrapper for features that depend on other features."""
    
    def __init__(self, feature: Any, dependencies: List[str]):
        self.feature = feature
        self.dependencies = dependencies
        
    def update(self, bar_data: Dict[str, Any], computed_features: Dict[str, Any]) -> Any:
        """Update feature with both bar data and computed features."""
        # Merge bar data and computed features for the update
        all_data = {**bar_data, **computed_features}
        
        # Check if feature has custom update method for dependencies
        if hasattr(self.feature, 'update_with_features'):
            return self.feature.update_with_features(all_data)
        else:
            # Standard update with computed features in kwargs
            return self.feature.update(
                price=bar_data.get("close", 0),
                high=bar_data.get("high"),
                low=bar_data.get("low"),
                volume=bar_data.get("volume"),
                **computed_features
            )


class EnhancedFeatureHub(FeatureHub):
    """
    Enhanced Feature Hub that supports feature dependencies.
    
    Features can depend on other features, and the hub will compute
    them in the correct order using topological sorting.
    """
    
    def __init__(self, symbols: Optional[List[str]] = None):
        super().__init__(symbols)
        
        # Track feature dependencies
        self._feature_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
    def register_dependent_feature(self, name: str, feature: Any, dependencies: List[str]) -> None:
        """
        Register a feature that depends on other features.
        
        Args:
            name: Feature name
            feature: Feature instance
            dependencies: List of feature names this depends on
        """
        wrapped = DependentFeature(feature, dependencies)
        
        for symbol in self.symbols:
            if symbol not in self._features:
                self._features[symbol] = {}
            self._features[symbol][name] = wrapped
            
        self._feature_dependencies[name] = set(dependencies)
        logger.info(f"Registered dependent feature {name} with dependencies: {dependencies}")
        
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
            in_degree[name] = len(deps)
            for dep in deps:
                if dep in features:  # Only consider dependencies that exist
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
    
    def update_bar(self, symbol: str, bar: Dict[str, Any]) -> None:
        """
        Update all features for a symbol with new bar data.
        
        Computes features in dependency order.
        """
        if symbol not in self._features:
            logger.warning(f"No features configured for symbol {symbol}")
            return
        
        self.bar_count[symbol] = self.bar_count.get(symbol, 0) + 1
        results = {}
        
        # Add raw OHLCV data to results first
        for key in ['open', 'high', 'low', 'close', 'volume', 'timestamp']:
            if key in bar:
                results[key] = bar[key]
        
        # Sort features by dependencies
        sorted_features = self._topological_sort(self._features[symbol])
        
        # Compute features in order
        for name, feature in sorted_features:
            try:
                if isinstance(feature, DependentFeature):
                    # Dependent feature - pass computed results
                    value = feature.update(bar, results)
                else:
                    # Regular feature - standard update
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
                import traceback
                traceback.print_exc()
        
        self.feature_cache[symbol] = results
        logger.debug("Updated bar for %s with %d features", symbol, len(results))