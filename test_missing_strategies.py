#\!/usr/bin/env python3
"""Test missing strategies to see why they don't generate signals."""

import sys
sys.path.insert(0, '/Users/daws/ADMF-PC')

from src.core.components.discovery import get_component_registry
from src.strategy.strategies.indicators import crossovers, oscillators, volatility, volume, trend, structure
import numpy as np

# Create dummy bar data
bar = {
    'timestamp': '2023-01-01T09:30:00',
    'symbol': 'SPY',
    'timeframe': '1m',
    'open': 100.0,
    'high': 100.5,
    'low': 99.5,
    'close': 100.2,
    'volume': 1000000
}

# Test strategies and their required features
test_configs = [
    ('dema_crossover', {'fast_dema_period': 3, 'slow_dema_period': 15}, ['dema_3', 'dema_15']),
    ('stochastic_crossover', {'k_period': 5, 'd_period': 3}, ['stochastic_k_5', 'stochastic_d_5_3']),
    ('vortex_crossover', {'vortex_period': 11}, ['vortex_vi_plus_11', 'vortex_vi_minus_11']),
    ('macd_crossover', {'fast_ema': 5, 'slow_ema': 20, 'signal_ema': 7}, ['macd_value_5_20_7', 'macd_signal_5_20_7']),
]

print("Testing missing strategies with minimal features...\n")

for strategy_name, params, required_features in test_configs:
    print(f"\n{'='*60}")
    print(f"Testing: {strategy_name}")
    print(f"Params: {params}")
    print(f"Required features: {required_features}")
    
    # Create minimal features dict
    features = {}
    for feat in required_features:
        # Assign dummy values
        if 'dema' in feat:
            features[feat] = 100.0
        elif 'stochastic' in feat:
            features[feat] = 50.0
        elif 'vortex' in feat:
            features[feat] = 1.0
        elif 'macd' in feat:
            features[feat] = 0.1
    
    print(f"Features provided: {features}")
    
    # Get the strategy function
    registry = get_component_registry()
    all_strategies = registry.get_components_by_type('strategy')
    
    # Find the strategy by name
    strategy_meta = None
    for s in all_strategies:
        if s.name == strategy_name:
            strategy_meta = s
            break
    
    if strategy_meta:
        print(f"Strategy found: {strategy_meta.name}")
        
        # Call the strategy
        try:
            result = strategy_meta.metadata['function'](features, bar, params)
            print(f"Result: {result}")
        except Exception as e:
            print(f"ERROR: {e}")
    else:
        print(f"Strategy '{strategy_name}' not found in registry!")

print("\n\nChecking what features these strategies actually expect...")

# Let's check the actual feature configs
for strategy_name, _, _ in test_configs:
    strategy_meta = None
    for s in all_strategies:
        if s.name == strategy_name:
            strategy_meta = s
            break
    
    if strategy_meta:
        feature_config = strategy_meta.metadata.get('feature_config', [])
        print(f"\n{strategy_name} feature_config: {feature_config}")