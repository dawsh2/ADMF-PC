#!/usr/bin/env python3
"""Debug why strategies aren't ready."""

import sys
sys.path.append('.')

from src.strategy.components.features.hub import FeatureHub
from src.core.coordinator.topology import TopologyBuilder
import yaml

# Load config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Build topology to get feature configs
builder = TopologyBuilder()
expanded_strategies = builder._expand_strategy_parameters(config.get('strategies', []))

# Get all required features
all_features = builder._infer_features_from_strategies(expanded_strategies)
print(f"Total features inferred: {len(all_features)}")

# Create feature hub and configure
hub = FeatureHub(['SPY'])

# Convert inferred features to feature configs
feature_configs = {}
for feature_str in all_features:
    # Parse feature string like "psar_0.02_0.2"
    parts = feature_str.split('_')
    if parts[0] == 'psar' and len(parts) == 3:
        feature_configs[feature_str] = {
            'type': 'psar',
            'af_start': float(parts[1]),
            'af_max': float(parts[2])
        }
    elif parts[0] == 'supertrend' and len(parts) == 3:
        feature_configs[feature_str] = {
            'type': 'supertrend',
            'period': int(parts[1]),
            'multiplier': float(parts[2])
        }
    elif parts[0] == 'adx' and len(parts) == 2:
        feature_configs[feature_str] = {
            'type': 'adx',
            'period': int(parts[1])
        }
    # Add more as needed...

hub.configure_features(feature_configs)

# Simulate some bars
print("\nSimulating bar updates...")
for i in range(20):
    bar = {
        'symbol': 'SPY',
        'open': 100 + i * 0.1,
        'high': 100.5 + i * 0.1,
        'low': 99.5 + i * 0.1,
        'close': 100 + i * 0.1,
        'volume': 100000
    }
    hub.update_bar('SPY', bar)

# Get features
features = hub.get_features('SPY')
print(f"\nAfter 20 bars, available features: {len(features)}")

# Check for specific features
print("\nChecking for missing strategy features:")
test_features = [
    'psar_0.02_0.2',
    'supertrend_10_3.0', 
    'supertrend_10_3.0_supertrend',
    'supertrend_10_3.0_trend',
    'adx_14',
    'adx_14_adx',
    'adx_14_di_plus',
    'adx_14_di_minus'
]

for feat in test_features:
    if feat in features:
        print(f"  ✓ {feat} = {features[feat]}")
    else:
        print(f"  ✗ {feat} NOT FOUND")

# Show actual feature names
print(f"\nSample of actual feature names:")
for i, (name, value) in enumerate(features.items()):
    if i < 20:
        print(f"  {name} = {value}")