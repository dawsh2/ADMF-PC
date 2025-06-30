"""
Example showing how parameter granularity should be defined in strategy metadata.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


# CURRENT APPROACH - No granularity info
@strategy(
    name='rsi_bands_current',
    parameter_space={
        'overbought': {'type': 'float', 'range': (60, 90), 'default': 70},
        'oversold': {'type': 'float', 'range': (10, 40), 'default': 30},
        'rsi_period': {'type': 'int', 'range': (7, 30), 'default': 14}
    }
)
def rsi_bands_current(features, bar, params):
    pass


# BETTER APPROACH - Granularity in parameter definition
@strategy(
    name='rsi_bands_improved',
    parameter_space={
        'overbought': {
            'type': 'float', 
            'range': (60, 90), 
            'default': 70,
            'granularity': 4,  # Test 60, 70, 80, 90 - critical levels
            'description': 'RSI overbought threshold'
        },
        'oversold': {
            'type': 'float', 
            'range': (10, 40), 
            'default': 30,
            'granularity': 4,  # Test 10, 20, 30, 40 - critical levels
            'description': 'RSI oversold threshold'
        },
        'rsi_period': {
            'type': 'int', 
            'range': (7, 30), 
            'default': 14,
            'granularity': 3,  # Test 7, 14, 30 - less sensitive
            'description': 'RSI calculation period'
        }
    }
)
def rsi_bands_improved(features, bar, params):
    pass


# EVEN BETTER - Granularity with rationale
@strategy(
    name='bollinger_bands_smart',
    parameter_space={
        'period': {
            'type': 'int', 
            'range': (10, 50), 
            'default': 20,
            'granularity': 3,  # Coarse - period changes slowly affect behavior
            'optimization_hint': 'low_sensitivity'
        },
        'num_std': {
            'type': 'float', 
            'range': (1.0, 3.0), 
            'default': 2.0,
            'granularity': 5,  # Fine - small changes significantly affect signals
            'optimization_hint': 'high_sensitivity',
            'recommended_values': [1.0, 1.5, 2.0, 2.5, 3.0]  # Explicit values
        }
    }
)
def bollinger_bands_smart(features, bar, params):
    pass


# ADVANCED - Conditional granularity based on other parameters
@strategy(
    name='adaptive_strategy',
    parameter_space={
        'base_period': {
            'type': 'int',
            'range': (10, 100),
            'default': 20,
            'granularity': 3
        },
        'multiplier': {
            'type': 'float',
            'range': (0.5, 5.0),
            'default': 2.0,
            'granularity': lambda params: 5 if params.get('base_period', 20) < 50 else 3
            # Fine granularity for short periods, coarse for long periods
        }
    },
    optimization_hints={
        'parameter_relationships': {
            'multiplier': 'inversely_proportional_to_base_period'
        },
        'search_strategy': 'adaptive',  # Could hint at smart search algorithms
        'estimated_runtime': 'O(n²)'    # Help optimizer plan resources
    }
)
def adaptive_strategy(features, bar, params):
    pass


# COMPREHENSIVE EXAMPLE - Full metadata approach
@strategy(
    name='keltner_breakout_complete',
    feature_discovery=lambda params: [
        FeatureSpec('keltner_channel', {
            'period': params.get('period', 20),
            'multiplier': params.get('multiplier', 2.0)
        }, outputs=['upper', 'middle', 'lower'])  # Clean multi-output syntax
    ],
    parameter_space={
        'period': {
            'type': 'int',
            'range': (10, 50),
            'default': 20,
            'granularity': 3,
            'values': None,  # Use auto-generated values
            'description': 'MA and ATR calculation period',
            'optimization_group': 'timing'  # Group related parameters
        },
        'multiplier': {
            'type': 'float',
            'range': (1.0, 4.0),
            'default': 2.0,
            'granularity': 7,  # More granular - key parameter
            'values': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],  # Explicit preferred values
            'description': 'ATR multiplier for band width',
            'optimization_group': 'sensitivity'
        },
        'exit_on_middle': {
            'type': 'bool',
            'default': False,
            'description': 'Exit positions when price crosses middle band',
            'optimization_group': 'risk_management'
        }
    },
    strategy_type='trend_following',
    tags=['breakout', 'volatility', 'keltner'],
    optimization_config={
        'priority_parameters': ['multiplier'],  # Optimize this first
        'parameter_groups': {
            'timing': {'search_together': True},
            'sensitivity': {'search_together': False}
        },
        'early_stopping': {
            'enabled': True,
            'min_trades': 20,
            'confidence_threshold': 0.95
        }
    }
)
def keltner_breakout_complete(features: Dict[str, Any], bar: Dict[str, Any], 
                             params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Complete Keltner Channel breakout strategy with full metadata.
    
    The strategy metadata includes:
    - Per-parameter granularity
    - Optimization hints and groups
    - Parameter relationships
    - Explicit value recommendations
    """
    # Strategy implementation
    upper = features.get('keltner_channel_upper')
    middle = features.get('keltner_channel_middle') 
    lower = features.get('keltner_channel_lower')
    price = bar.get('close', 0)
    
    if upper is None or lower is None:
        return None
    
    # Signal logic
    if price > upper:
        signal_value = 1  # Breakout above
    elif price < lower:
        signal_value = -1  # Breakout below
    elif params.get('exit_on_middle', False) and middle:
        # Optional exit on middle band touch
        if abs(price - middle) < (upper - middle) * 0.1:
            signal_value = 0
        else:
            signal_value = None  # Maintain previous signal
    else:
        signal_value = 0  # In channel
    
    if signal_value is None:
        return None
        
    return {
        'signal_value': signal_value,
        'metadata': {
            'price': price,
            'upper_band': upper,
            'lower_band': lower,
            'band_width': upper - lower
        }
    }


# How the parameter expander would use this information
def expand_parameters_with_metadata(strategy_metadata: Dict[str, Any]) -> list:
    """
    Expand parameters using strategy-defined granularity.
    """
    import numpy as np
    from itertools import product
    
    param_space = strategy_metadata.get('parameter_space', {})
    combinations = []
    
    # Build value lists for each parameter
    param_values = {}
    for param_name, param_config in param_space.items():
        param_type = param_config.get('type')
        
        if param_type == 'bool':
            param_values[param_name] = [True, False]
            continue
            
        # Check for explicit values first
        if 'values' in param_config and param_config['values']:
            param_values[param_name] = param_config['values']
            continue
        
        # Otherwise use granularity
        param_range = param_config.get('range')
        granularity = param_config.get('granularity', 3)  # Default to 3
        
        # Handle callable granularity
        if callable(granularity):
            # Would need current parameter context here
            granularity = 3  # Fallback for example
        
        if param_type == 'int':
            values = np.linspace(param_range[0], param_range[1], granularity)
            values = sorted(list(set(int(round(v)) for v in values)))
        else:  # float
            values = np.linspace(param_range[0], param_range[1], granularity)
            values = [round(float(v), 4) for v in values]
        
        param_values[param_name] = values
    
    # Generate combinations respecting optimization groups
    optimization_config = strategy_metadata.get('optimization_config', {})
    param_groups = optimization_config.get('parameter_groups', {})
    
    # Simple product for now (could be smarter with groups)
    for values in product(*param_values.values()):
        combo = dict(zip(param_values.keys(), values))
        combinations.append(combo)
    
    return combinations