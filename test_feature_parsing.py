#!/usr/bin/env python3
"""Test feature ID parsing logic."""

# Test feature IDs
test_features = [
    'bollinger_bands_20_2.0',
    'bollinger_bands_20_2',
    'stochastic_14_3',
    'macd_12_26_9',
    'sma_20',
    'rsi_14',
    'adx_14',
    'vwap',
    'ad'
]

print("=== TESTING FEATURE ID PARSING ===\n")

for feature_id in test_features:
    print(f"Feature ID: {feature_id}")
    parts = feature_id.split('_')
    print(f"  Parts: {parts}")
    
    if len(parts) >= 2:
        feature_type = parts[0]
        print(f"  Feature type: {feature_type}")
        
        # Check for compound feature names
        if feature_type == 'bollinger' and len(parts) > 1 and parts[1] == 'bands':
            print("  -> This is bollinger_bands, needs special handling!")
            actual_parts = [parts[0] + '_' + parts[1]] + parts[2:]
            print(f"  Actual parts: {actual_parts}")
    else:
        print(f"  Feature type: {parts[0] if parts else feature_id}")
    
    print()

# Now let's see how the actual code would handle it
print("\n=== SIMULATING ACTUAL PARSING ===\n")

def parse_feature_id(feature_id):
    """Simulate the parsing logic."""
    parts = feature_id.split('_')
    
    # Special handling for compound feature names
    if feature_id.startswith('bollinger_bands_'):
        feature_type = 'bollinger_bands'
        param_parts = feature_id[len('bollinger_bands_'):].split('_')
        return feature_type, param_parts
    elif feature_id.startswith('donchian_channel_'):
        feature_type = 'donchian_channel'
        param_parts = feature_id[len('donchian_channel_'):].split('_')
        return feature_type, param_parts
    elif feature_id.startswith('keltner_channel_'):
        feature_type = 'keltner_channel'
        param_parts = feature_id[len('keltner_channel_'):].split('_')
        return feature_type, param_parts
    elif feature_id.startswith('linear_regression_'):
        feature_type = 'linear_regression'
        param_parts = feature_id[len('linear_regression_'):].split('_')
        return feature_type, param_parts
    elif feature_id.startswith('parabolic_sar_'):
        feature_type = 'parabolic_sar'
        param_parts = feature_id[len('parabolic_sar_'):].split('_')
        return feature_type, param_parts
    elif feature_id.startswith('ultimate_oscillator_'):
        feature_type = 'ultimate_oscillator' 
        param_parts = feature_id[len('ultimate_oscillator_'):].split('_')
        return feature_type, param_parts
    elif feature_id.startswith('support_resistance_'):
        feature_type = 'support_resistance'
        param_parts = feature_id[len('support_resistance_'):].split('_')
        return feature_type, param_parts
    elif feature_id.startswith('fibonacci_retracement_'):
        feature_type = 'fibonacci_retracement'
        param_parts = feature_id[len('fibonacci_retracement_'):].split('_')
        return feature_type, param_parts
    elif feature_id.startswith('swing_points_'):
        feature_type = 'swing_points'
        param_parts = feature_id[len('swing_points_'):].split('_')
        return feature_type, param_parts
    elif feature_id.startswith('pivot_points_'):
        feature_type = 'pivot_points'
        param_parts = feature_id[len('pivot_points_'):].split('_')
        return feature_type, param_parts
    elif feature_id.startswith('stochastic_rsi_'):
        feature_type = 'stochastic_rsi'
        param_parts = feature_id[len('stochastic_rsi_'):].split('_')
        return feature_type, param_parts
    else:
        # Simple feature name
        if len(parts) >= 2:
            feature_type = parts[0]
            param_parts = parts[1:]
        else:
            feature_type = parts[0] if parts else feature_id
            param_parts = []
        
    return feature_type, param_parts

for feature_id in test_features:
    feature_type, param_parts = parse_feature_id(feature_id)
    print(f"{feature_id} -> type='{feature_type}', params={param_parts}")