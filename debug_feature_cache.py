#!/usr/bin/env python3
"""Debug what features are actually available."""

import sys
sys.path.insert(0, '/Users/daws/ADMF-PC')

# Create a minimal test that replicates the signal generation setup
from src.strategy.components.features.hub import FeatureHub
import pandas as pd
import numpy as np

print("Testing FeatureHub feature generation:")
print("=" * 60)

# Create minimal test data
n_bars = 100
np.random.seed(42)
data = {
    'open': np.random.uniform(99, 101, n_bars),
    'high': np.random.uniform(100, 102, n_bars),
    'low': np.random.uniform(98, 100, n_bars),
    'close': np.random.uniform(99, 101, n_bars),
    'volume': np.random.uniform(1000000, 2000000, n_bars)
}
df = pd.DataFrame(data)

# Create FeatureHub
hub = FeatureHub(symbols=['SPY'])

# Configure stochastic features like the topology would
feature_configs = {
    'stochastic_5_3': {
        'feature': 'stochastic',
        'k_period': 5,
        'd_period': 3
    }
}

hub.configure_features(feature_configs)

# Add data bar by bar
for i in range(n_bars):
    bar = {
        'open': data['open'][i],
        'high': data['high'][i], 
        'low': data['low'][i],
        'close': data['close'][i],
        'volume': data['volume'][i]
    }
    hub.update_bar('SPY', bar)

# Check what features are available
features = hub.get_features('SPY')
print(f"Available features: {list(features.keys())}")

# Check for stochastic specifically
stoch_features = {k: v for k, v in features.items() if 'stochastic' in k}
print(f"Stochastic features: {stoch_features}")

# Test what strategy would look for
expected_k = 'stochastic_5_3_k'
expected_d = 'stochastic_5_3_d'
print(f"\nStrategy expects:")
print(f"  {expected_k}: {'✓' if expected_k in features else '✗'}")
print(f"  {expected_d}: {'✓' if expected_d in features else '✗'}")

if expected_k in features and expected_d in features:
    print(f"\n✓ Features found!")
    print(f"  K value: {features[expected_k]}")
    print(f"  D value: {features[expected_d]}")
else:
    print(f"\n✗ Features missing!")
    print("Available feature keys that contain 'stochastic':")
    for key in features.keys():
        if 'stochastic' in key.lower():
            print(f"  {key}")
    
    print("\nAll available features:")
    for key in sorted(features.keys()):
        print(f"  {key}: {features[key]}")