#!/usr/bin/env python3
"""Test new FeatureHub implementation."""

import sys
sys.path.insert(0, '/Users/daws/ADMF-PC')

# Direct import from hub.py
from src.strategy.components.features.hub import FeatureHub
from src.strategy.strategies.indicators.crossovers import stochastic_crossover
import numpy as np

print("Testing new FeatureHub implementation")
print("=" * 50)

# Create FeatureHub
hub = FeatureHub(symbols=['SPY'])

# Configure with stochastic feature
feature_configs = {
    'stochastic_5_3': {
        'type': 'stochastic',
        'k_period': 5,
        'd_period': 3
    }
}

hub.configure_features(feature_configs)

# Add test data
n_bars = 20
np.random.seed(42)
for i in range(n_bars):
    bar = {
        'open': 100.0,
        'high': 101.0,
        'low': 99.0,
        'close': 100.0,
        'volume': 1000000
    }
    hub.update_bar('SPY', bar)

# Check generated features
features = hub.get_features('SPY')
print("Generated features:")
for name, value in sorted(features.items()):
    print(f"  {name}: {value}")

# Test strategy expectations
expected_k = 'stochastic_5_3_k'
expected_d = 'stochastic_5_3_d'

print(f"\nStrategy expects: {expected_k}, {expected_d}")
print(f"K available: {expected_k in features}")
print(f"D available: {expected_d in features}")

if expected_k in features and expected_d in features:
    print("✅ SUCCESS: New FeatureHub generates expected features!")
else:
    print("❌ Features missing - checking what's available:")
    stoch_features = [k for k in features.keys() if 'stoch' in k.lower()]
    print(f"Stochastic-related features: {stoch_features}")