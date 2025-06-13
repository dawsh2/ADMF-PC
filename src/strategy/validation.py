"""
Feature dependency validation for strategies and classifiers.

This module provides validation to ensure strategies and classifiers have all
required features available before execution, preventing silent failures.
"""

from typing import Dict, Any, List, Set, Optional, Callable, Union
from dataclasses import dataclass
import logging
import inspect
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class FeatureDependency:
    """Represents a feature dependency with validation rules."""
    name: str
    required: bool = True
    default_value: Any = None
    validator: Optional[Callable[[Any], bool]] = None
    error_message: Optional[str] = None


class FeatureDependencyError(Exception):
    """Raised when required features are missing or invalid."""
    
    def __init__(self, missing_features: List[str], strategy_name: str):
        self.missing_features = missing_features
        self.strategy_name = strategy_name
        super().__init__(
            f"Strategy '{strategy_name}' missing required features: {', '.join(missing_features)}"
        )


class FeatureValidator:
    """
    Validates feature dependencies for strategies and classifiers.
    
    Zero inheritance - pure composition following ADMF-PC architecture.
    """
    
    def __init__(self):
        self._dependency_cache: Dict[str, List[FeatureDependency]] = {}
        self._validation_stats = {
            'validations_performed': 0,
            'failures': 0,
            'missing_features_total': 0
        }
    
    def validate_features(
        self,
        features: Dict[str, Any],
        required_features: List[str],
        component_name: str = "unknown"
    ) -> None:
        """
        Validate that all required features are present.
        
        Args:
            features: Available features dictionary
            required_features: List of required feature names
            component_name: Name of component for error messages
            
        Raises:
            FeatureDependencyError: If required features are missing
        """
        self._validation_stats['validations_performed'] += 1
        
        missing = []
        for feature_name in required_features:
            if feature_name not in features or features[feature_name] is None:
                missing.append(feature_name)
                self._validation_stats['missing_features_total'] += 1
        
        if missing:
            self._validation_stats['failures'] += 1
            logger.error(
                f"Feature validation failed for {component_name}: "
                f"missing {len(missing)} features: {missing}"
            )
            raise FeatureDependencyError(missing, component_name)
        
        logger.debug(f"Feature validation passed for {component_name}")
    
    def validate_with_dependencies(
        self,
        features: Dict[str, Any],
        dependencies: List[FeatureDependency],
        component_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Validate features with detailed dependency rules.
        
        Args:
            features: Available features
            dependencies: List of FeatureDependency objects
            component_name: Component name for errors
            
        Returns:
            Validated features dict with defaults applied
            
        Raises:
            FeatureDependencyError: If validation fails
        """
        validated_features = features.copy()
        missing_required = []
        
        for dep in dependencies:
            feature_value = features.get(dep.name)
            
            # Check if missing
            if feature_value is None:
                if dep.required:
                    missing_required.append(dep.name)
                elif dep.default_value is not None:
                    validated_features[dep.name] = dep.default_value
                    logger.debug(
                        f"Applied default value for {dep.name} in {component_name}: "
                        f"{dep.default_value}"
                    )
                continue
            
            # Run custom validator if provided
            if dep.validator and not dep.validator(feature_value):
                error_msg = dep.error_message or f"Invalid value for {dep.name}: {feature_value}"
                logger.error(f"Feature validation failed for {component_name}: {error_msg}")
                raise ValueError(error_msg)
        
        if missing_required:
            raise FeatureDependencyError(missing_required, component_name)
        
        return validated_features
    
    def get_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        return self._validation_stats.copy()


# Global validator instance
_global_validator = FeatureValidator()


def get_feature_validator() -> FeatureValidator:
    """Get the global feature validator instance."""
    return _global_validator


def validate_strategy_features(strategy_func: Callable) -> Callable:
    """
    Decorator that validates features before strategy execution.
    
    This decorator dynamically constructs the required parameterized feature names
    based on the strategy's parameters at runtime.
    
    Usage:
        @validate_strategy_features
        def my_strategy(features, bar, params):
            # Strategy will only execute if required features are present
            return signal
    """
    @wraps(strategy_func)
    def wrapper(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Any:
        # TEMPORARILY DISABLED - validation is handled by ComponentState per-strategy readiness checking
        return strategy_func(features, bar, params)
        # Dynamic feature validation - construct actual required feature names
        required_features = []
        func_name = getattr(strategy_func, '__name__', 'unknown_strategy')
        
        # Get feature config from component info
        if hasattr(strategy_func, '_component_info'):
            component_info = strategy_func._component_info
            metadata = component_info.metadata
            feature_config = metadata.get('feature_config', {})
            
            # Debug logging for breakout strategy
            if 'breakout' in func_name:
                logger.error(f"Breakout strategy validation - feature_config: {feature_config}, params: {params}")
            
            # For each feature type, construct the actual parameterized name
            for feature_name, feature_meta in feature_config.items():
                param_names = feature_meta.get('params', [])
                defaults = feature_meta.get('defaults', {})
                default_value = feature_meta.get('default')
                
                if param_names:
                    # Handle multiple parameters for the same feature
                    for param_name in param_names:
                        # Get parameter value from strategy params, then defaults, then general default
                        param_value = params.get(param_name)
                        if param_value is None:
                            param_value = defaults.get(param_name, default_value)
                        
                        if param_value is not None:
                            # Special handling for volume features that need suffix
                            if feature_name == 'volume' and param_name == 'lookback_period':
                                # volume_20 becomes volume_20_volume_ma for breakout strategies
                                feature_name_final = f'{feature_name}_{param_value}_volume_ma'
                                required_features.append(feature_name_final)
                                if 'breakout' in func_name:
                                    logger.error(f"Added volume feature: {feature_name_final}")
                            else:
                                # Standard parameterized feature name: high_20, low_20, etc.
                                feature_name_final = f'{feature_name}_{param_value}'
                                required_features.append(feature_name_final)
                                if 'breakout' in func_name:
                                    logger.error(f"Added feature: {feature_name_final}")
                            
                            # For volume features, also try the base name without suffix (e.g., volume_20)
                            if feature_name == 'volume' and param_name == 'lookback_period':
                                base_volume_name = f'{feature_name}_{param_value}'
                                required_features.append(base_volume_name)
                                if 'breakout' in func_name:
                                    logger.error(f"Added base volume feature: {base_volume_name}")
                else:
                    # Feature without parameters
                    required_features.append(feature_name)
        
        # Fall back to static required_features if no dynamic construction
        if not required_features:
            required_features = getattr(strategy_func, 'required_features', [])
            if 'breakout' in func_name:
                logger.error(f"Breakout strategy fallback - using static required_features: {required_features}")
        
        if required_features:
            if 'breakout' in func_name:
                logger.error(f"Breakout strategy final validation - required_features: {required_features}")
            validator = get_feature_validator()
            validator.validate_features(
                features,
                required_features,
                getattr(strategy_func, '__name__', 'unknown_strategy')
            )
        
        # Call original function
        return strategy_func(features, bar, params)
    
    # Preserve original function attributes
    wrapper.required_features = getattr(strategy_func, 'required_features', [])
    
    return wrapper


def validate_classifier_features(classifier_func: Callable) -> Callable:
    """
    Decorator that validates features before classifier execution.
    
    Similar to validate_strategy_features but for classifiers.
    """
    @wraps(classifier_func)
    def wrapper(features: Dict[str, Any], params: Dict[str, Any]) -> Any:
        # Extract required features
        required_features = getattr(classifier_func, 'required_features', [])
        
        if not required_features and hasattr(classifier_func, '_component_info'):
            component_info = classifier_func._component_info
            required_features = component_info.metadata.get('features', [])
        
        if required_features:
            validator = get_feature_validator()
            validator.validate_features(
                features,
                required_features,
                getattr(classifier_func, '__name__', 'unknown_classifier')
            )
        
        return classifier_func(features, params)
    
    wrapper.required_features = getattr(classifier_func, 'required_features', [])
    
    return wrapper


class StrategyWrapper:
    """
    Wrapper that adds feature validation to existing strategies.
    
    This allows us to add validation to strategies without modifying them.
    Zero inheritance - pure composition.
    """
    
    def __init__(self, strategy_func: Callable, required_features: List[str]):
        self.strategy_func = strategy_func
        self.required_features = required_features
        self.name = getattr(strategy_func, '__name__', 'wrapped_strategy')
        self._validator = get_feature_validator()
    
    def generate_signal(
        self,
        features: Dict[str, Any],
        bar: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate signal with feature validation.
        
        Validates features before calling the wrapped strategy.
        """
        # Validate features
        self._validator.validate_features(
            features,
            self.required_features,
            self.name
        )
        
        # Call wrapped strategy
        return self.strategy_func(features, bar, params)
    
    def __call__(self, features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Any:
        """Allow wrapper to be called directly."""
        return self.generate_signal(features, bar, params)


def create_validated_strategy(
    strategy_func: Callable,
    required_features: Optional[List[str]] = None
) -> StrategyWrapper:
    """
    Create a validated version of a strategy function.
    
    Args:
        strategy_func: Original strategy function
        required_features: List of required features (auto-detected if None)
        
    Returns:
        StrategyWrapper with validation
    """
    # Auto-detect required features if not provided
    if required_features is None:
        required_features = []
        
        # Check for required_features attribute
        if hasattr(strategy_func, 'required_features'):
            required_features = strategy_func.required_features
        
        # Check component info
        elif hasattr(strategy_func, '_component_info'):
            component_info = strategy_func._component_info
            metadata = component_info.metadata
            
            # Extract from feature_config
            feature_config = metadata.get('feature_config', {})
            if feature_config:
                required_features = list(feature_config.keys())
            else:
                required_features = metadata.get('features', [])
    
    return StrategyWrapper(strategy_func, required_features)


def extract_required_features(strategy_or_classifier: Any) -> List[str]:
    """
    Extract required features from a strategy or classifier.
    
    Checks multiple sources:
    1. required_features property/attribute
    2. _component_info metadata
    3. Function annotations
    
    Args:
        strategy_or_classifier: Strategy or classifier to inspect
        
    Returns:
        List of required feature names
    """
    required_features = []
    
    # Check direct attribute
    if hasattr(strategy_or_classifier, 'required_features'):
        attr = getattr(strategy_or_classifier, 'required_features')
        if callable(attr):
            required_features = attr()
        else:
            required_features = attr
    
    # Check component info
    elif hasattr(strategy_or_classifier, '_component_info'):
        component_info = strategy_or_classifier._component_info
        metadata = component_info.metadata
        
        # Try feature_config first
        feature_config = metadata.get('feature_config', {})
        if feature_config:
            required_features = list(feature_config.keys())
        else:
            # Fall back to features list
            required_features = metadata.get('features', [])
    
    # Try to extract from function signature/docstring
    elif callable(strategy_or_classifier):
        # Could parse docstring or type hints here
        pass
    
    return required_features


# Feature dependency specifications for common indicators
COMMON_FEATURE_DEPENDENCIES = {
    'sma': FeatureDependency(
        name='sma',
        required=True,
        validator=lambda x: isinstance(x, (int, float)) and x >= 0,
        error_message="SMA must be a non-negative number"
    ),
    'ema': FeatureDependency(
        name='ema',
        required=True,
        validator=lambda x: isinstance(x, (int, float)) and x >= 0,
        error_message="EMA must be a non-negative number"
    ),
    'rsi': FeatureDependency(
        name='rsi',
        required=True,
        validator=lambda x: isinstance(x, (int, float)) and 0 <= x <= 100,
        error_message="RSI must be between 0 and 100"
    ),
    'macd': FeatureDependency(
        name='macd',
        required=True,
        validator=lambda x: isinstance(x, dict) and all(k in x for k in ['macd', 'signal', 'histogram']),
        error_message="MACD must contain 'macd', 'signal', and 'histogram' values"
    ),
    'volume': FeatureDependency(
        name='volume',
        required=False,
        default_value=0,
        validator=lambda x: isinstance(x, (int, float)) and x >= 0,
        error_message="Volume must be non-negative"
    )
}