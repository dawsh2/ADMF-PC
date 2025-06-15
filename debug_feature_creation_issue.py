#!/usr/bin/env python3
"""Debug the specific issue with feature creation in FeatureHub."""

import logging
from src.strategy.components.features.hub import FeatureHub

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test the _create_feature method directly
hub = FeatureHub(['SPY'])

# Test correct config format (from topology.py)
correct_config = {
    'sma_20': {'type': 'sma', 'period': 20},
    'rsi_14': {'type': 'rsi', 'period': 14},
    'atr_14': {'type': 'atr', 'period': 14}
}

print("=== TESTING FEATURE CREATION ===\n")
print("Correct config format from topology:")
for name, config in correct_config.items():
    print(f"  {name}: {config}")

print("\nTesting _create_feature directly:")
for name, config in correct_config.items():
    try:
        feature = hub._create_feature(name, config)
        print(f"  ✓ {name} -> {type(feature).__name__}")
    except Exception as e:
        print(f"  ✗ {name} -> ERROR: {e}")

# Check what's in the registry
from src.strategy.components.features.hub import FEATURE_REGISTRY
print(f"\nFEATURE_REGISTRY contains {len(FEATURE_REGISTRY)} features:")
for key in sorted(FEATURE_REGISTRY.keys())[:10]:
    print(f"  {key}: {FEATURE_REGISTRY[key].__name__}")

# Now test configuring the hub
print("\nConfiguring FeatureHub with correct configs...")
hub.configure_features(correct_config)

print(f"FeatureHub configured with {len(hub._feature_configs)} features")

# Test updating a bar to trigger feature creation
bar_data = {
    'open': 100.0,
    'high': 101.0,
    'low': 99.0,
    'close': 100.5,
    'volume': 1000000
}

print("\nUpdating with test bar...")
try:
    hub.update_bar('SPY', bar_data)
    features = hub.get_features('SPY')
    print(f"Successfully computed {len(features)} features")
    for name, value in list(features.items())[:5]:
        print(f"  {name}: {value}")
except Exception as e:
    print(f"ERROR during bar update: {e}")
    import traceback
    traceback.print_exc()