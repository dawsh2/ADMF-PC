"""
Feature discovery system for topology builder.

This module handles the discovery and validation of features required by strategies,
supporting both static requirements and dynamic discovery based on parameters.
"""

from typing import Dict, Any, List, Set, Optional, Callable
import logging
import re
from ..features.feature_spec import FeatureSpec, FEATURE_REGISTRY
from ..components.discovery import get_component_registry

# Import strategies to ensure they're registered
try:
    import src.strategy.strategies
except ImportError:
    pass  # Strategies might not be available in all contexts

logger = logging.getLogger(__name__)


class FeatureDiscovery:
    """Handles feature discovery and validation for strategies."""
    
    def __init__(self):
        self.registry = get_component_registry()
    
    def discover_all_features(self, strategies: List[Dict[str, Any]], 
                            classifiers: List[Dict[str, Any]]) -> Dict[str, FeatureSpec]:
        """
        Discover all features required by strategies and classifiers.
        
        Returns dict mapping canonical feature names to FeatureSpec objects.
        """
        all_features = {}
        
        # Discover from strategies
        for strategy_config in strategies:
            features = self.discover_strategy_features(strategy_config)
            for spec in features:
                all_features[spec.canonical_name] = spec
        
        # Discover from classifiers
        for classifier_config in classifiers:
            features = self.discover_classifier_features(classifier_config)
            for spec in features:
                all_features[spec.canonical_name] = spec
        
        logger.info(f"Discovered {len(all_features)} unique features")
        return all_features
    
    def discover_strategy_features(self, config: Dict[str, Any]) -> List[FeatureSpec]:
        """
        Discover features for a single strategy configuration.
        
        Handles:
        1. New-style FeatureSpec with static/dynamic discovery
        2. Old-style feature_config and param_feature_mapping
        3. Ensemble strategies (recursive discovery)
        4. Filter-based feature discovery
        """
        strategy_type = config.get('type') or config.get('name')
        if not strategy_type:
            logger.warning(f"Strategy config missing type/name: {config}")
            return []
        
        # Get strategy metadata from registry
        component_info = self.registry.get_component(strategy_type)
        if not component_info:
            logger.warning(f"Strategy '{strategy_type}' not found in registry")
            return []
        
        metadata = component_info.metadata
        
        # Start with features from metadata
        features = []
        
        # Handle new-style FeatureSpec
        if metadata.get('required_features') or metadata.get('feature_discovery'):
            features.extend(self._discover_with_feature_spec(config, metadata))
        
        # Handle ensemble strategies
        elif strategy_type in ['ensemble', 'universal_ensemble', 'two_layer_ensemble']:
            features.extend(self._discover_ensemble_features(config))
        
        # Old-style strategies must be migrated!
        elif metadata.get('feature_config') or metadata.get('param_feature_mapping'):
            raise ValueError(
                f"Strategy '{strategy_type}' uses legacy feature system. "
                f"Must be migrated to use FeatureSpec with required_features or feature_discovery. "
                f"See crossovers_migrated.py for examples."
            )
        
        # NEW: Discover features from filter/threshold/constraints expressions
        filter_expr = config.get('constraints') or config.get('threshold') or config.get('filter')
        if filter_expr:
            filter_features = self._discover_features_from_filter(filter_expr)
            logger.info(f"Discovered {len(filter_features)} features from constraints/filter expression")
            features.extend(filter_features)
        
        # Deduplicate by canonical name
        unique_features = {}
        for spec in features:
            unique_features[spec.canonical_name] = spec
        
        return list(unique_features.values())
    
    def _discover_with_feature_spec(self, config: Dict[str, Any], 
                                   metadata: Dict[str, Any]) -> List[FeatureSpec]:
        """Handle new-style FeatureSpec discovery."""
        features = []
        
        # Static requirements
        if metadata.get('required_features'):
            features.extend(metadata['required_features'])
        
        # Dynamic discovery
        if metadata.get('feature_discovery'):
            discovery_func = metadata['feature_discovery']
            # Try multiple parameter locations - param_overrides is used by expanded strategies
            params = config.get('params') or config.get('param_overrides') or {}
            
            # Log for debugging
            if 'bollinger' in config.get('type', ''):
                logger.debug(f"Discovering features for {config.get('name', 'unknown')} with params: {params}")
            
            try:
                discovered = discovery_func(params)
                features.extend(discovered)
            except Exception as e:
                logger.error(f"Feature discovery failed: {e}")
        
        return features
    
    def _discover_legacy_features(self, config: Dict[str, Any], 
                                 metadata: Dict[str, Any]) -> List[FeatureSpec]:
        """
        Convert old-style feature requirements to FeatureSpec.
        
        This handles the existing param_feature_mapping patterns.
        """
        features = []
        params = config.get('params', {})
        
        # Get feature types from feature_config
        feature_types = []
        if isinstance(metadata.get('feature_config'), list):
            feature_types = metadata['feature_config']
        elif isinstance(metadata.get('feature_config'), dict):
            feature_types = list(metadata['feature_config'].keys())
        
        # For each feature type, try to determine parameters
        for feature_type in feature_types:
            if feature_type not in FEATURE_REGISTRY:
                logger.warning(f"Unknown feature type: {feature_type}")
                continue
            
            registry_entry = FEATURE_REGISTRY[feature_type]
            
            # Try to extract parameters from strategy params
            feature_params = {}
            for param_name in registry_entry.required_params:
                # Look for common parameter patterns
                value = self._find_parameter_value(param_name, params, feature_type)
                if value is not None:
                    feature_params[param_name] = value
            
            # Only create FeatureSpec if we found all required parameters
            if len(feature_params) == len(registry_entry.required_params):
                try:
                    spec = FeatureSpec(feature_type, feature_params)
                    features.append(spec)
                except Exception as e:
                    logger.warning(f"Failed to create FeatureSpec: {e}")
        
        return features
    
    def _find_parameter_value(self, param_name: str, strategy_params: Dict[str, Any], 
                             feature_type: str) -> Optional[Any]:
        """
        Try to find a parameter value using common naming patterns.
        
        This handles the mess of inconsistent parameter naming.
        """
        # Direct match
        if param_name in strategy_params:
            return strategy_params[param_name]
        
        # Feature-prefixed (e.g., 'sma_period' for 'period')
        prefixed = f"{feature_type}_{param_name}"
        if prefixed in strategy_params:
            return strategy_params[prefixed]
        
        # Common aliases
        aliases = {
            'period': ['lookback', 'window', 'length'],
            'fast_period': ['fast', 'fast_ma', 'fast_ema'],
            'slow_period': ['slow', 'slow_ma', 'slow_ema'],
            'signal_period': ['signal', 'signal_ema'],
            'k_period': ['k', 'fast_k'],
            'd_period': ['d', 'slow_d'],
            'multiplier': ['mult', 'factor', 'std_dev'],
            'threshold': ['thresh', 'level'],
        }
        
        for alias in aliases.get(param_name, []):
            if alias in strategy_params:
                return strategy_params[alias]
            # Also try with feature prefix
            prefixed_alias = f"{feature_type}_{alias}"
            if prefixed_alias in strategy_params:
                return strategy_params[prefixed_alias]
        
        return None
    
    def _discover_ensemble_features(self, config: Dict[str, Any]) -> List[FeatureSpec]:
        """Recursively discover features for ensemble strategies."""
        all_features = []
        
        # Check for sub-strategies
        for key in ['strategies', 'baseline_strategies', 'sub_strategies']:
            if key in config:
                for sub_config in config[key]:
                    features = self.discover_strategy_features(sub_config)
                    all_features.extend(features)
        
        # Check for regime-specific strategies
        if 'regime_strategies' in config or 'regime_boosters' in config:
            regime_dict = config.get('regime_strategies') or config.get('regime_boosters', {})
            for regime, regime_strategies in regime_dict.items():
                if isinstance(regime_strategies, list):
                    for sub_config in regime_strategies:
                        features = self.discover_strategy_features(sub_config)
                        all_features.extend(features)
        
        # Deduplicate by canonical name
        unique_features = {}
        for spec in all_features:
            unique_features[spec.canonical_name] = spec
        
        return list(unique_features.values())
    
    def _discover_features_from_filter(self, filter_expr) -> List[FeatureSpec]:
        """
        Extract features from a filter expression.
        
        Args:
            filter_expr: Either a string expression or a list of expressions
            
        Examples:
        - "signal == 0 or (rsi_14 < 50)" -> [FeatureSpec for rsi_14]
        - "signal == 0 or (volume > volume_sma_20 * 1.2)" -> [FeatureSpecs for volume, volume_sma_20]
        - ["intraday", "volume > sma(volume, 20)"] -> [FeatureSpecs for volume, sma_volume_20]
        """
        if not filter_expr:
            return []
        
        # Handle list of constraints - combine them with AND
        if isinstance(filter_expr, list):
            combined_expr = " and ".join(f"({expr})" for expr in filter_expr)
            filter_expr = combined_expr
        
        features = []
        
        # First, find function-style features like sma(volume, 20), atr(14), rsi(14)
        function_pattern = r'(\w+)\s*\(\s*(?:(\w+)\s*,\s*)?(\d+)\s*\)'
        function_matches = re.findall(function_pattern, filter_expr)
        
        # Also find simple feature names (word characters with optional underscore and digits)
        # This matches: rsi_14, volume_sma_20, atr_14, close, vwap, etc.
        feature_pattern = r'\b([a-z_]+(?:_\d+)?)\b'
        simple_matches = re.findall(feature_pattern, filter_expr, re.IGNORECASE)
        
        # Filter out common operators, keywords, and non-features
        keywords = {
            'signal', 'or', 'and', 'not', 'if', 'else', 'true', 'false', 
            'abs', 'min', 'max', 'log', 'exp', 'sqrt',
            'intraday', 'time', 'hour', 'minute',  # Special context variables
        }
        
        # Process function-style features
        function_features = []
        for func_name, arg1, arg2 in function_matches:
            if func_name.lower() in keywords:
                continue
            if arg1:  # e.g., sma(volume, 20)
                function_features.append((func_name.lower(), arg1.lower(), int(arg2)))
            else:  # e.g., atr(14), rsi(14)
                function_features.append((func_name.lower(), None, int(arg2)))
        
        # Process simple feature names
        feature_names = {m for m in simple_matches if m not in keywords and not m.isdigit()}
        
        logger.debug(f"Found function features in filter: {function_features}")
        logger.debug(f"Found simple features in filter: {feature_names}")
        
        # Convert function features to FeatureSpec objects
        for func_info in function_features:
            feature_spec = self._create_feature_spec_from_function(func_info)
            if feature_spec:
                features.append(feature_spec)
        
        # Convert simple feature names to FeatureSpec objects
        for feature_name in feature_names:
            feature_spec = self._create_feature_spec_from_name(feature_name)
            if feature_spec:
                features.append(feature_spec)
        
        return features
    
    def _create_feature_spec_from_function(self, func_info: tuple) -> Optional[FeatureSpec]:
        """
        Create a FeatureSpec from a function-style feature.
        
        Args:
            func_info: Tuple of (function_name, optional_arg1, period)
            
        Examples:
            ('atr', None, 14) -> FeatureSpec(feature_type='atr', params={'period': 14})
            ('sma', 'volume', 20) -> FeatureSpec(feature_type='volume_sma', params={'period': 20})
            ('rsi', None, 14) -> FeatureSpec(feature_type='rsi', params={'period': 14})
        """
        func_name, arg1, period = func_info
        
        # Skip if it's actually raw data
        if func_name in ['open', 'high', 'low', 'close', 'volume']:
            return None
        
        # Handle different function patterns
        if arg1:
            # Pattern: func(data_source, period) -> data_source_func
            # e.g., sma(volume, 20) -> volume_sma with period 20
            if arg1 == 'volume' and func_name == 'sma':
                return FeatureSpec(
                    feature_type='volume_sma',
                    params={'period': period}
                )
            else:
                # Generic pattern
                return FeatureSpec(
                    feature_type=f'{arg1}_{func_name}',
                    params={'period': period}
                )
        else:
            # Pattern: func(period) -> func with period
            # e.g., atr(14) -> atr with period 14
            return FeatureSpec(
                feature_type=func_name,
                params={'period': period}
            )
    
    def _create_feature_spec_from_name(self, feature_name: str) -> Optional[FeatureSpec]:
        """
        Create a FeatureSpec from a feature name found in a filter.
        
        Examples:
        - "rsi_14" -> FeatureSpec(name="rsi", params={"period": 14})
        - "volume_sma_20" -> FeatureSpec(name="volume_sma", params={"period": 20})
        - "close" -> FeatureSpec(name="close", params={})
        """
        # Handle raw data features
        if feature_name in ['open', 'high', 'low', 'close', 'volume']:
            # These are provided directly in bar data
            logger.debug(f"Skipping raw data field: {feature_name}")
            return None
        
        # VWAP is special - it needs to be computed as a feature
        if feature_name == 'vwap':
            return FeatureSpec(
                feature_type='vwap',
                params={}
            )
        
        # bar_of_day is handled in the filter context, not as a feature
        if feature_name == 'bar_of_day':
            logger.debug(f"Skipping bar_of_day - handled in filter context")
            return None
        
        # Parse indicator features with parameters
        # Pattern: indicator_param or indicator_type_param
        parts = feature_name.split('_')
        
        # Common patterns:
        # rsi_14 -> rsi with period 14
        # volume_sma_20 -> sma of volume with period 20
        # atr_14 -> atr with period 14
        # atr_sma_50 -> sma of atr with period 50
        
        if len(parts) == 2 and parts[1].isdigit():
            # Simple pattern: indicator_period
            indicator = parts[0]
            period = int(parts[1])
            
            if indicator in ['rsi', 'atr', 'ema', 'sma']:
                return FeatureSpec(
                    feature_type=indicator,
                    params={'period': period}
                )
        
        elif len(parts) == 3:
            # Complex pattern: source_indicator_period
            source = parts[0]
            indicator = parts[1]
            
            if parts[2].isdigit():
                period = int(parts[2])
                
                if indicator == 'sma' and source == 'atr':
                    # atr_sma_50 -> SMA of ATR (composite feature)
                    # Until we have proper composite feature support, ensure base feature exists
                    logger.info(
                        f"Composite feature {feature_name} detected. "
                        f"Ensuring base feature 'atr' is available. "
                        f"Note: Filter will use current ATR value, not true SMA."
                    )
                    # Return ATR feature so at least the base is available
                    return FeatureSpec(
                        feature_type='atr',
                        params={'period': 14}  # Standard ATR period
                    )
                elif indicator == 'sma' and source == 'volume':
                    # volume_sma_20 -> volume SMA
                    return FeatureSpec(
                        feature_type='volume_sma',
                        params={'period': period}
                    )
        
        # If we can't parse it, log a warning
        logger.warning(f"Could not create FeatureSpec for: {feature_name}")
        return None
    
    def discover_classifier_features(self, config: Dict[str, Any]) -> List[FeatureSpec]:
        """
        Discover features for classifiers.
        
        Similar to strategies but with classifier-specific patterns.
        """
        classifier_type = config.get('type') or config.get('name')
        params = config.get('params', {})
        
        # Common classifier features based on type
        features = []
        
        # Most classifiers need these base features
        base_features = {
            'atr': ['atr_period'],
            'rsi': ['rsi_period'],
            'sma': ['sma_short', 'sma_long', 'sma_period'],
            'ema': ['ema_period'],
        }
        
        for feature_type, param_names in base_features.items():
            for param_name in param_names:
                if param_name in params:
                    try:
                        period = params[param_name]
                        spec = FeatureSpec(feature_type, {'period': period})
                        features.append(spec)
                    except Exception as e:
                        logger.warning(f"Failed to create {feature_type} spec: {e}")
        
        return features
    
    def validate_features_available(self, required_specs: Dict[str, FeatureSpec],
                                  available_features: Set[str]) -> List[str]:
        """
        Validate that all required features are available.
        
        Returns list of missing feature names.
        """
        missing = []
        for canonical_name, spec in required_specs.items():
            if canonical_name not in available_features:
                missing.append(canonical_name)
        
        return missing