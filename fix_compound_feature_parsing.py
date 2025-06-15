#!/usr/bin/env python3
"""Fix for compound feature name parsing in topology.py"""

def create_feature_config_from_id(feature_id):
    """
    Parse a feature ID and create the appropriate feature configuration.
    
    Handles compound feature names like 'bollinger_bands_20_2.0' correctly.
    """
    # Dictionary of compound feature names and their expected parameter patterns
    compound_features = {
        'bollinger_bands': {'params': ['period', 'std_dev'], 'defaults': [20, 2.0]},
        'donchian_channel': {'params': ['period'], 'defaults': [20]},
        'keltner_channel': {'params': ['period', 'multiplier'], 'defaults': [20, 2.0]},
        'linear_regression': {'params': ['period'], 'defaults': [20]},
        'parabolic_sar': {'params': ['af_start', 'af_max'], 'defaults': [0.02, 0.2]},
        'ultimate_oscillator': {'params': ['period1', 'period2', 'period3'], 'defaults': [7, 14, 28]},
        'support_resistance': {'params': ['period'], 'defaults': [50]},
        'fibonacci_retracement': {'params': ['period'], 'defaults': [50]},
        'swing_points': {'params': ['period'], 'defaults': [5]},
        'pivot_points': {'params': [], 'defaults': []},  # No parameters
        'stochastic_rsi': {'params': ['rsi_period', 'stoch_period'], 'defaults': [14, 14]},
        'atr_sma': {'params': ['atr_period', 'sma_period'], 'defaults': [14, 20]},
        'volatility_sma': {'params': ['vol_period', 'sma_period'], 'defaults': [20, 20]},
    }
    
    # First check if it's a compound feature
    for compound_name, info in compound_features.items():
        if feature_id.startswith(compound_name + '_') or feature_id == compound_name:
            # Extract parameter values
            if feature_id == compound_name:
                # No parameters provided, use defaults
                config = {'feature': compound_name}
                for i, param_name in enumerate(info['params']):
                    config[param_name] = info['defaults'][i]
                return config
            
            # Parse parameters after the compound name
            param_str = feature_id[len(compound_name) + 1:]  # +1 for underscore
            param_parts = param_str.split('_')
            
            config = {'feature': compound_name}
            
            # Map parameters to their names
            for i, param_name in enumerate(info['params']):
                if i < len(param_parts):
                    try:
                        # Try to convert to appropriate type
                        if '.' in param_parts[i]:
                            value = float(param_parts[i])
                        else:
                            value = int(param_parts[i])
                        config[param_name] = value
                    except ValueError:
                        # Use default if conversion fails
                        config[param_name] = info['defaults'][i] if i < len(info['defaults']) else None
                else:
                    # Use default if not enough parameters
                    config[param_name] = info['defaults'][i] if i < len(info['defaults']) else None
            
            return config
    
    # Not a compound feature, parse as simple feature
    parts = feature_id.split('_')
    
    if len(parts) == 1:
        # Simple feature without parameters (e.g., 'vwap', 'ad')
        return {'feature': feature_id}
    
    feature_type = parts[0]
    
    # Handle MACD specially (3 parameters)
    if feature_type == 'macd' and len(parts) >= 4:
        try:
            return {
                'feature': 'macd',
                'fast': int(parts[1]),
                'slow': int(parts[2]),
                'signal': int(parts[3])
            }
        except ValueError:
            return {'feature': 'macd', 'fast': 12, 'slow': 26, 'signal': 9}
    
    # Handle stochastic (2 parameters: k_period, d_period)
    elif feature_type == 'stochastic' and len(parts) >= 3:
        try:
            return {
                'feature': 'stochastic',
                'k_period': int(parts[1]),
                'd_period': int(parts[2])
            }
        except ValueError:
            return {'feature': 'stochastic', 'k_period': 14, 'd_period': 3}
    
    # Handle features with standard 'period' parameter
    elif len(parts) >= 2:
        try:
            period = int(parts[1])
            return {'feature': feature_type, 'period': period}
        except ValueError:
            # Default period for most indicators
            return {'feature': feature_type, 'period': 20}
    
    # Fallback
    return {'feature': feature_type}


# Test the function
test_features = [
    'bollinger_bands_20_2.0',
    'bollinger_bands_20_2',
    'bollinger_bands',
    'stochastic_14_3',
    'stochastic_rsi_14_14',
    'macd_12_26_9',
    'sma_20',
    'rsi_14',
    'adx_14',
    'vwap',
    'ad',
    'donchian_channel_20',
    'keltner_channel_20_2.0',
    'linear_regression_20',
    'parabolic_sar_0.02_0.2',
    'ultimate_oscillator_7_14_28',
    'support_resistance_50',
    'fibonacci_retracement_50',
    'swing_points_5',
    'pivot_points',
]

print("=== TESTING IMPROVED FEATURE CONFIG CREATION ===\n")

for feature_id in test_features:
    config = create_feature_config_from_id(feature_id)
    print(f"{feature_id:30} -> {config}")