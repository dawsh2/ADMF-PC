"""
Feature Inference Logic for ADMF-PC

This module contains the core logic for automatically determining required features
based on strategy configurations. Moved from coordinator module to follow proper
separation of concerns - strategy analysis belongs in the strategy module.

This is part of Step 10.0.1 coordinator consolidation to eliminate ADMF-PC violations.
"""

import logging
from typing import Dict, Any, List, Set

logger = logging.getLogger(__name__)


def infer_features_from_strategies(strategies: List[Dict[str, Any]]) -> Set[str]:
    """Infer required features from strategy configurations using discovery system.
    
    This uses the discovery registry to automatically determine what features
    are needed based on strategy metadata and parameter values.
    
    Args:
        strategies: List of strategy configuration dictionaries
        
    Returns:
        Set of required feature identifiers
    """
    from ...core.containers.discovery import get_component_registry
    
    required_features = set()
    registry = get_component_registry()
    
    for strategy_config in strategies:
        strategy_type = strategy_config.get('type', strategy_config.get('class'))
        strategy_params = strategy_config.get('parameters', {})
        
        # Get strategy metadata from registry
        strategy_info = registry.get_component(strategy_type)
        
        if strategy_info:
            # Extract feature requirements from metadata
            feature_config = strategy_info.metadata.get('feature_config', {})
            
            # For each feature type the strategy needs
            for feature_name, feature_meta in feature_config.items():
                param_names = feature_meta.get('params', [])
                defaults = feature_meta.get('defaults', {})
                default_value = feature_meta.get('default')
                
                # Handle parameter lists (for grid search)
                if param_names:
                    for param_name in param_names:
                        if param_name in strategy_params:
                            param_values = strategy_params[param_name]
                            # Handle both single values and lists
                            if isinstance(param_values, list):
                                for value in param_values:
                                    required_features.add(f'{feature_name}_{value}')
                            else:
                                required_features.add(f'{feature_name}_{param_values}')
                        elif param_name in defaults:
                            # Use specific default for this param
                            required_features.add(f'{feature_name}_{defaults[param_name]}')
                        elif default_value is not None:
                            # Use general default
                            required_features.add(f'{feature_name}_{default_value}')
                else:
                    # Feature with no params, just add it
                    required_features.add(feature_name)
            
            logger.info(f"Strategy '{strategy_type}' requires features: {sorted(required_features)}")
            
        else:
            # Fallback for strategies not in registry
            logger.warning(f"Strategy '{strategy_type}' not found in registry, using hardcoded inference")
            
            # Legacy hardcoded logic as fallback
            if strategy_type in ['MomentumStrategy', 'momentum']:
                lookback_period = strategy_params.get('lookback_period', 20)
                rsi_period = strategy_params.get('rsi_period', 14)
                required_features.add(f'sma_{lookback_period}')
                required_features.add(f'rsi_{rsi_period}')
            elif strategy_type in ['MeanReversionStrategy', 'mean_reversion']:
                period = strategy_params.get('period', 20)
                required_features.add(f'bollinger_{period}')
                required_features.add('rsi_14')
            else:
                # Default features
                required_features.update(['sma_20', 'rsi_14'])
    
    # If no strategies found, add default features
    if not required_features:
        logger.warning("No strategies found, using default features")
        required_features.update(['sma_20', 'rsi_14'])
        
    return required_features


def get_strategy_requirements(strategy_config: Dict[str, Any]) -> Dict[str, Any]:
    """Get comprehensive requirements for a single strategy.
    
    Args:
        strategy_config: Strategy configuration dictionary
        
    Returns:
        Dictionary with indicators, dependencies, and other requirements
    """
    strategy_class = strategy_config.get('class', strategy_config.get('type'))
    strategy_params = strategy_config.get('parameters', {})
    
    requirements = {
        'indicators': set(),
        'dependencies': [],
        'risk_requirements': {},
        'data_requirements': {}
    }
    
    if strategy_class in ['MomentumStrategy', 'momentum']:
        lookback_period = strategy_params.get('lookback_period', 20)
        requirements['indicators'].update([f'SMA_{lookback_period}', 'RSI'])
        requirements['data_requirements']['min_history'] = max(lookback_period, 14) + 5
        
    elif strategy_class in ['MeanReversionStrategy', 'mean_reversion']:
        period = strategy_params.get('period', 20)
        requirements['indicators'].update([f'BB_{period}', 'RSI'])
        requirements['data_requirements']['min_history'] = max(period, 14) + 5
        
    elif strategy_class in ['moving_average_crossover', 'momentum_crossover']:
        for param_name, param_value in strategy_params.items():
            if 'fast_period' in param_name:
                requirements['indicators'].add(f'SMA_{param_value}')
            elif 'slow_period' in param_name:
                requirements['indicators'].add(f'SMA_{param_value}')
                requirements['data_requirements']['min_history'] = max(
                    requirements['data_requirements'].get('min_history', 0),
                    param_value + 5
                )
            elif 'rsi_period' in param_name:
                requirements['indicators'].add('RSI')
    
    # Convert set to list for JSON serialization
    requirements['indicators'] = list(requirements['indicators'])
    
    return requirements


def validate_strategy_configuration(strategy_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a strategy configuration and return validation results.
    
    Args:
        strategy_config: Strategy configuration to validate
        
    Returns:
        Dictionary with validation results
    """
    errors = []
    warnings = []
    
    # Check required fields
    if not strategy_config.get('type') and not strategy_config.get('class'):
        errors.append("Strategy configuration missing 'type' or 'class' field")
    
    strategy_class = strategy_config.get('class', strategy_config.get('type'))
    strategy_params = strategy_config.get('parameters', {})
    
    # Strategy-specific validation
    if strategy_class in ['MomentumStrategy', 'momentum']:
        lookback_period = strategy_params.get('lookback_period', 20)
        if lookback_period < 5:
            warnings.append(f"Momentum lookback period {lookback_period} is very short")
        elif lookback_period > 200:
            warnings.append(f"Momentum lookback period {lookback_period} is very long")
            
    elif strategy_class in ['MeanReversionStrategy', 'mean_reversion']:
        period = strategy_params.get('period', 20)
        if period < 5:
            warnings.append(f"Mean reversion period {period} is very short")
            
    elif strategy_class in ['moving_average_crossover', 'momentum_crossover']:
        fast_period = None
        slow_period = None
        
        for param_name, param_value in strategy_params.items():
            if 'fast_period' in param_name:
                fast_period = param_value
            elif 'slow_period' in param_name:
                slow_period = param_value
                
        if fast_period and slow_period:
            if fast_period >= slow_period:
                errors.append(f"Fast period ({fast_period}) must be less than slow period ({slow_period})")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'strategy_class': strategy_class,
        'requirements': get_strategy_requirements(strategy_config)
    }