"""
Generic Feature Inference Implementation

This module provides a generic approach to inferring feature requirements
from strategy parameters, eliminating the need for hardcoded feature-specific logic.
"""

# Feature parameter mapping - defines how to extract parameters for each feature type
FEATURE_PARAM_MAPPING = {
    # Single parameter features (period-based)
    'sma': ['period', 'sma_period', 'fast_period', 'slow_period'],
    'ema': ['period', 'ema_period', 'fast_ema_period', 'slow_ema_period'],
    'dema': ['period', 'dema_period', 'fast_dema_period', 'slow_dema_period'],
    'tema': ['period', 'tema_period'],
    'rsi': ['period', 'rsi_period'],
    'cci': ['period', 'cci_period'],
    'williams_r': ['period', 'williams_period'],
    'roc': ['period', 'roc_period'],
    'vortex': ['period', 'vortex_period'],
    'atr': ['period', 'atr_period'],
    'adx': ['period', 'adx_period'],
    'di': ['period', 'di_period'],
    'obv': ['period', 'obv_sma_period'],
    'mfi': ['period', 'mfi_period'],
    'cmf': ['period'],
    'chaikin_money_flow': ['period'],
    'ad': ['period', 'ad_ema_period'],
    'accumulation_distribution': ['period', 'ad_ema_period'],
    'aroon': ['period'],
    'linear_regression': ['period'],
    'fibonacci_retracement': ['period'],
    'fibonacci': ['period'],
    'support_resistance': ['period'],
    'swing_points': ['period'],
    'donchian_channel': ['period'],
    'donchian': ['period'],
    
    # Two parameter features
    'bollinger_bands': [('period', 'bb_period'), ('std_dev', 'bb_std')],
    'bollinger': [('period', 'bb_period'), ('std_dev', 'bb_std')],
    'keltner_channel': [('period',), ('multiplier',)],
    'keltner': [('period',), ('multiplier',)],
    'stochastic': [('k_period',), ('d_period',)],
    'stochastic_rsi': [('rsi_period',), ('stoch_period',)],
    'parabolic_sar': [('af_start',), ('af_max',)],
    'psar': [('af_start',), ('af_max',)],
    'supertrend': [('period',), ('multiplier',)],
    'ichimoku': [('conversion_period',), ('base_period',)],
    
    # Three parameter features
    'macd': [('fast_ema',), ('slow_ema',), ('signal_ema',)],
    'ultimate_oscillator': [('period1',), ('period2',), ('period3',)],
    'ultimate': [('period1',), ('period2',), ('period3',)],
    
    # Special features
    'vwap': [],  # No parameters
    'pivot_points': [('pivot_type',)],  # String parameter
}

def infer_feature_requirements_generic(feature_name, strategy_params):
    """
    Generic feature requirement inference.
    
    Args:
        feature_name: Name of the feature (e.g., 'sma', 'bollinger_bands')
        strategy_params: Dictionary of strategy parameters
        
    Returns:
        Set of required feature strings (e.g., {'sma_10', 'sma_20'})
    """
    required_features = set()
    
    # Get parameter mapping for this feature
    param_mapping = FEATURE_PARAM_MAPPING.get(feature_name, None)
    
    if param_mapping is None:
        # Unknown feature - try generic single-parameter approach
        for param_name, param_values in strategy_params.items():
            if 'period' in param_name and feature_name in param_name:
                if isinstance(param_values, list):
                    for value in param_values:
                        required_features.add(f'{feature_name}_{value}')
                else:
                    required_features.add(f'{feature_name}_{param_values}')
        return required_features
    
    # Handle special cases
    if feature_name == 'vwap':
        required_features.add('vwap')
        return required_features
    
    # Check if it's a simple single-parameter feature
    if isinstance(param_mapping, list) and all(isinstance(p, str) for p in param_mapping):
        # Single parameter feature - look for any matching parameter names
        for param_name in param_mapping:
            if param_name in strategy_params:
                param_values = strategy_params[param_name]
                if isinstance(param_values, list):
                    for value in param_values:
                        required_features.add(f'{feature_name}_{value}')
                else:
                    required_features.add(f'{feature_name}_{param_values}')
        return required_features
    
    # Multi-parameter feature
    if isinstance(param_mapping, list) and all(isinstance(p, tuple) for p in param_mapping):
        # Collect values for each parameter
        param_value_lists = []
        
        for param_group in param_mapping:
            values_found = []
            for param_name in param_group:
                if param_name in strategy_params:
                    param_values = strategy_params[param_name]
                    if isinstance(param_values, list):
                        values_found.extend(param_values)
                    else:
                        values_found.append(param_values)
                    break  # Found values for this parameter group
            
            if values_found:
                param_value_lists.append(values_found)
            else:
                # Use default if no values found
                if 'period' in param_group[0]:
                    param_value_lists.append([20])  # Default period
                elif 'multiplier' in param_group[0] or 'std_dev' in param_group[0]:
                    param_value_lists.append([2.0])  # Default multiplier
                else:
                    param_value_lists.append([None])  # Unknown default
        
        # Generate all combinations
        import itertools
        if all(vals[0] is not None for vals in param_value_lists):
            for combination in itertools.product(*param_value_lists):
                feature_str = f'{feature_name}_' + '_'.join(str(v) for v in combination)
                required_features.add(feature_str)
    
    return required_features


# Example test
if __name__ == "__main__":
    # Test various feature types
    test_cases = [
        # Single parameter
        ('sma', {'fast_period': [10, 20], 'slow_period': [30, 40]}),
        ('rsi', {'rsi_period': [14, 21]}),
        
        # Two parameters
        ('bollinger_bands', {'period': [20, 30], 'std_dev': [2.0, 2.5]}),
        ('stochastic', {'k_period': [5, 14], 'd_period': [3, 5]}),
        
        # Three parameters
        ('macd', {'fast_ema': [12], 'slow_ema': [26], 'signal_ema': [9]}),
        ('ultimate_oscillator', {'period1': [7], 'period2': [14], 'period3': [28]}),
        
        # Special
        ('vwap', {}),
        ('pivot_points', {'pivot_type': ['standard', 'fibonacci']}),
    ]
    
    for feature_name, params in test_cases:
        features = infer_feature_requirements_generic(feature_name, params)
        print(f"{feature_name}: {sorted(features)}")