#!/usr/bin/env python3
"""Debug structure strategy readiness and feature access."""

import yaml
import sys
sys.path.append('.')

from src.core.coordinator.topology import TopologyBuilder
from src.strategy.components.features.hub import FeatureHub
from src.strategy.state import ComponentState

# Test linear regression strategy specifically
print("Testing linear_regression_slope strategy...")

# Load config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Find linear regression strategy
lr_strategies = [s for s in config.get('strategies', []) if s.get('type') == 'linear_regression_slope']
if not lr_strategies:
    print("No linear regression strategy found in config")
    sys.exit(1)

lr_config = lr_strategies[0]
print(f"Config: {lr_config}")

# Expand parameters
builder = TopologyBuilder()
expanded = builder._expand_strategy_parameters([lr_config])
print(f"Expanded to {len(expanded)} instances")

# Test first instance
first_instance = expanded[0]
print(f"Testing instance: {first_instance}")

# Infer features
inferred = builder._infer_features_from_strategies([first_instance])
print(f"Inferred features: {sorted(inferred)}")

# Create FeatureHub and configure features
feature_hub = FeatureHub(['SPY'])

# Convert inferred features to feature configs
feature_configs = {}
for feature_name in inferred:
    if feature_name.startswith('linear_regression_'):
        period = feature_name.split('_')[-1]
        feature_configs[feature_name] = {
            'type': 'linear_regression',
            'period': int(period)
        }

print(f"Feature configs: {feature_configs}")
feature_hub.configure_features(feature_configs)

# Create test bars
import time
test_bars = []
base_price = 100.0
for i in range(50):  # 50 bars for testing
    price = base_price + (i * 0.5) + (i % 10) * 0.2  # Trending with noise
    bar = {
        'symbol': 'SPY',
        'timestamp': int(time.time()) + i,
        'open': price - 0.1,
        'high': price + 0.2,
        'low': price - 0.2,
        'close': price,
        'volume': 1000000 + i * 1000
    }
    test_bars.append(bar)

print(f"Created {len(test_bars)} test bars")

# Process bars through FeatureHub
print("Processing bars through FeatureHub...")
for i, bar in enumerate(test_bars):
    feature_hub.update_bar('SPY', bar)
    
    # Check feature readiness every 10 bars
    if i % 10 == 9:
        features = feature_hub.get_features('SPY')
        lr_feature_name = list(inferred)[0]  # First linear regression feature
        lr_value = features.get(lr_feature_name)
        print(f"  Bar {i+1}: {lr_feature_name} = {lr_value}")

# Final feature check
final_features = feature_hub.get_features('SPY')
print(f"\nFinal features available: {list(final_features.keys())}")
for feature_name in inferred:
    value = final_features.get(feature_name)
    print(f"  {feature_name}: {value}")

# Test strategy readiness
print(f"\nTesting strategy readiness...")

# Import strategy function
from src.strategy.strategies.indicators.trend import linear_regression_slope

# Create ComponentState to test readiness
from src.core.components.discovery import get_component_registry
import importlib
importlib.import_module('src.strategy.strategies.indicators.trend')

registry = get_component_registry()
strategy_info = registry.get_component('linear_regression_slope')
print(f"Strategy info: {strategy_info}")

if strategy_info:
    # Test with final features
    try:
        result = linear_regression_slope(final_features, test_bars[-1], first_instance['params'])
        print(f"Strategy result: {result}")
    except Exception as e:
        print(f"Strategy execution failed: {e}")
else:
    print("Strategy not registered")