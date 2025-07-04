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
        
        # Feature states per symbol - Dict[symbol, Dict[feature_name, FeatureState]]
        self._feature_states: Dict[str, Dict[str, FeatureState]] = defaultdict(dict)
        
        # Features to compute per symbol
        self._feature_configs: Dict[str, Dict[str, Any]] = {}
        
        # Bar count per symbol for warmup tracking
        self._bar_counts: Dict[str, int] = defaultdict(int)
        
        # Cache of latest features per symbol
        self._feature_cache: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Track which features have been initialized
        self._initialized_features: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance metrics
        self._update_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"FeatureHub initialized for symbols: {symbols}")
    
    def configure_features(self, feature_configs: Dict[str, Dict[str, Any]], 
                          required_features: Optional[List[str]] = None) -> None:
        """
        Configure which features to compute.
        
        Args:
            feature_configs: Dict mapping feature names to their configurations
            required_features: Optional list of features actually needed (for optimization)
        """
        if required_features:
            # Only configure features that are actually needed
            self._feature_configs = {
                name: config 
                for name, config in feature_configs.items() 
                if name in required_features
            }
            logger.info(f"Configured {len(self._feature_configs)} features "
                       f"(optimized from {len(feature_configs)} total)")
        else:
            self._feature_configs = feature_configs.copy()
            logger.info(f"Configured {len(self._feature_configs)} features")
        
        # Initialize feature states for existing symbols
        for symbol in self.symbols:
            self._initialize_features_for_symbol(symbol)
    
    def _initialize_features_for_symbol(self, symbol: str) -> None:
        """Initialize feature states for a symbol."""
        if symbol not in self._feature_states:
            self._feature_states[symbol] = {}
        
        for feature_name, config in self._feature_configs.items():
            if feature_name not in self._initialized_features[symbol]:
                # Get the base feature type from the config
                feature_type = config.get('feature', feature_name.split('_')[0])
                
                # Create feature instance from registry
                if feature_type in FEATURE_REGISTRY:
                    feature_class = FEATURE_REGISTRY[feature_type]
                    feature_instance = feature_class(**config)
                    
                    # Create feature state
                    self._feature_states[symbol][feature_name] = FeatureState(
                        feature=feature_instance,
                        config=config
                    )
                    
                    self._initialized_features[symbol].add(feature_name)
                    logger.debug(f"Initialized feature {feature_name} for {symbol}")
                else:
                    logger.warning(f"Unknown feature type: {feature_type}")
    
    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to track."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self._initialize_features_for_symbol(symbol)
            logger.info(f"Added symbol {symbol} to FeatureHub")
    
    def update_bar(self, symbol: str, bar: Dict[str, float]) -> None:
        """
        Update features with new bar data.
        
        Uses O(1) incremental computation for all features.
        
        Args:
            symbol: Symbol to update
            bar: Bar data with 'open', 'high', 'low', 'close', 'volume'
        """
        if symbol not in self.symbols:
            self.add_symbol(symbol)
        
        # Ensure features are initialized
        if symbol not in self._initialized_features or not self._initialized_features[symbol]:
            self._initialize_features_for_symbol(symbol)
        
        # Increment bar count
        self._bar_counts[symbol] += 1
        self._update_count += 1
        
        # Clear cache for this symbol (will be recomputed on demand)
        self._feature_cache[symbol].clear()
        
        # Update all feature states with new bar
        for feature_name, feature_state in self._feature_states[symbol].items():
            try:
                feature_state.update(bar)
            except Exception as e:
                logger.error(f"Error updating feature {feature_name} for {symbol}: {e}")
        
        # Log progress periodically
        if self._update_count % 1000 == 0:
            logger.debug(f"FeatureHub processed {self._update_count} bars. "
                        f"Cache hit rate: {self._get_cache_hit_rate():.1%}")
    
    def get_features(self, symbol: str) -> Dict[str, Any]:
        """
        Get current features for a symbol.
        
        Returns cached values if available, otherwise computes and caches.
        
        Args:
            symbol: Symbol to get features for
            
        Returns:
            Dict of feature_name -> feature_value
        """
        # Return cached features if available
        if symbol in self._feature_cache and self._feature_cache[symbol]:
            self._cache_hits += 1
            return self._feature_cache[symbol]
        
        self._cache_misses += 1
        
        # Compute all features
        features = {}
        
        if symbol in self._feature_states:
            for feature_name, feature_state in self._feature_states[symbol].items():
                try:
                    value = feature_state.get_value()
                    # Handle multi-output features
                    if isinstance(value, dict):
                        # For multi-output features, add each output with suffix
                        for output_name, output_value in value.items():
                            full_name = f"{feature_name}_{output_name}"
                            features[full_name] = output_value
                    else:
                        # Single output feature
                        features[feature_name] = value
                except Exception as e:
                    logger.error(f"Error getting feature {feature_name} for {symbol}: {e}")
                    features[feature_name] = None
        
        # Cache the computed features
        self._feature_cache[symbol] = features
        
        return features
    
    def get_all_features(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all features for all symbols.
        
        Returns:
            Dict mapping symbol -> feature_dict
        """
        return {symbol: self.get_features(symbol) for symbol in self.symbols}
    
    def has_sufficient_data(self, symbol: str, min_bars: int = 50) -> bool:
        """
        Check if symbol has sufficient data for feature calculation.
        
        Args:
            symbol: Symbol to check
            min_bars: Minimum number of bars required
            
        Returns:
            True if sufficient data available
        """
        return self._bar_counts.get(symbol, 0) >= min_bars
    
    def reset(self, symbol: Optional[str] = None) -> None:
        """
        Reset feature computation state.
        
        Args:
            symbol: Symbol to reset (None for all symbols)
        """
        if symbol is None:
            # Reset all symbols
            self._feature_states.clear()
            self._feature_cache.clear()
            self._bar_counts.clear()
            self._initialized_features.clear()
            logger.info("Reset all feature states")
        else:
            # Reset specific symbol
            if symbol in self._feature_states:
                del self._feature_states[symbol]
            if symbol in self._feature_cache:
                del self._feature_cache[symbol]
            if symbol in self._bar_counts:
                del self._bar_counts[symbol]
            if symbol in self._initialized_features:
                del self._initialized_features[symbol]
            logger.info(f"Reset feature states for {symbol}")
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of feature computation state."""
        feature_counts = {}
        for symbol in self.symbols:
            if symbol in self._feature_states:
                feature_counts[symbol] = len(self._feature_states[symbol])
        
        return {
            'symbols': self.symbols,
            'feature_counts': feature_counts,
            'total_features': sum(feature_counts.values()),
            'bar_counts': dict(self._bar_counts),
            'update_count': self._update_count,
            'cache_hit_rate': self._get_cache_hit_rate()
        }
    
    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0
    
    def get_feature_dependencies(self) -> Dict[str, List[str]]:
        """
        Get feature dependency graph.
        
        Returns:
            Dict mapping feature_name -> list of dependencies
        """
        dependencies = {}
        
        # For now, most features are independent
        # In the future, we could add explicit dependencies
        # e.g., MACD depends on EMA values
        
        return dependencies
    
    def validate_features(self) -> Tuple[bool, List[str]]:
        """
        Validate that all configured features can be computed.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for feature_name, config in self._feature_configs.items():
            feature_type = config.get('feature', feature_name.split('_')[0])
            
            if feature_type not in FEATURE_REGISTRY:
                errors.append(f"Unknown feature type: {feature_type} for {feature_name}")
            else:
                # Could add parameter validation here
                pass
        
        return len(errors) == 0, errors