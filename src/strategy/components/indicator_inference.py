"""
Indicator Inference Logic for ADMF-PC

This module contains the core logic for automatically determining required indicators
based on strategy configurations. Moved from coordinator module to follow proper
separation of concerns - strategy analysis belongs in the strategy module.

This is part of Step 10.0.1 coordinator consolidation to eliminate ADMF-PC violations.
"""

import logging
from typing import Dict, Any, List, Set

logger = logging.getLogger(__name__)


def infer_indicators_from_strategies(strategies: List[Dict[str, Any]]) -> Set[str]:
    """Infer required indicators from strategy configurations.
    
    This is the core indicator inference logic that automatically determines
    what indicators are needed based on strategy configurations.
    
    Args:
        strategies: List of strategy configuration dictionaries
        
    Returns:
        Set of required indicator identifiers
    """
    required_indicators = set()
    
    for strategy_config in strategies:
        strategy_class = strategy_config.get('class', strategy_config.get('type'))
        strategy_params = strategy_config.get('parameters', {})
        
        if strategy_class in ['MomentumStrategy', 'momentum']:
            # MomentumStrategy needs SMA for momentum and RSI for signals
            lookback_period = strategy_params.get('lookback_period', 20)
            rsi_period = strategy_params.get('rsi_period', 14)
            
            required_indicators.add(f'SMA_{lookback_period}')
            required_indicators.add('RSI')
            
            logger.info(f"MomentumStrategy requires: SMA_{lookback_period}, RSI")
            
        elif strategy_class in ['MeanReversionStrategy', 'mean_reversion']:
            # MeanReversionStrategy typically needs Bollinger Bands, RSI
            period = strategy_params.get('period', 20)
            required_indicators.add(f'BB_{period}')
            required_indicators.add('RSI')
            
            logger.info(f"MeanReversionStrategy requires: BB_{period}, RSI")
            
        elif strategy_class in ['moving_average_crossover', 'momentum_crossover']:
            # For crossover strategies, infer from parameter names
            for param_name, param_value in strategy_params.items():
                if 'fast_period' in param_name:
                    required_indicators.add(f'SMA_{param_value}')
                elif 'slow_period' in param_name:
                    required_indicators.add(f'SMA_{param_value}')
                elif 'rsi_period' in param_name:
                    required_indicators.add('RSI')
                    
        else:
            # Default indicators for unknown strategies
            logger.warning(f"Unknown strategy class {strategy_class}, using default indicators")
            required_indicators.update(['SMA_20', 'RSI'])
    
    # If no strategies found, add default indicators to prevent empty indicator hub
    if not required_indicators:
        logger.warning("No strategies found, using default indicators")
        required_indicators.update(['SMA_20', 'RSI'])
        
    return required_indicators


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