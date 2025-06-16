#!/usr/bin/env python3
"""Debug feature mapping logic."""

# Test the logic used in topology.py

strategies_to_test = [
    {
        'feature_config': ['obv', 'sma'],
        'param_feature_mapping': {'obv_sma_period': 'sma_{obv_sma_period}'}
    },
    {
        'feature_config': ['ad', 'ema'],
        'param_feature_mapping': {'ad_ema_period': 'ema_{ad_ema_period}'}
    },
    {
        'feature_config': ['vwap'],
        'param_feature_mapping': {'std_multiplier': 'vwap_{std_multiplier}'}
    }
]

for i, strategy in enumerate(strategies_to_test):
    print(f"\nStrategy {i+1}:")
    feature_config = strategy['feature_config']
    param_feature_mapping = strategy['param_feature_mapping']
    
    print(f"  feature_config: {feature_config}")
    print(f"  param_feature_mapping: {param_feature_mapping}")
    
    for feature_name in feature_config:
        # Check if this feature is covered by param_feature_mapping
        is_parameterized = any(
            template.startswith(feature_name + '_') or template == feature_name
            for template in param_feature_mapping.values()
        )
        print(f"  Feature '{feature_name}': is_parameterized = {is_parameterized}")
        
        if not is_parameterized:
            print(f"    -> Would add '{feature_name}' as base feature")
        else:
            print(f"    -> Parameterized, handled by mapping")