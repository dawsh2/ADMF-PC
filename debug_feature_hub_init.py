#!/usr/bin/env python3
"""Debug script to trace how FeatureHub is initialized."""

import logging
from src.core.containers.components.feature_hub_component import create_feature_hub_component

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test container config with features
container_config = {
    'name': 'feature_hub',
    'type': 'feature_hub',
    'symbols': ['SPY'],
    'features': {
        'sma_10': {'feature': 'sma', 'period': 10},
        'sma_20': {'feature': 'sma', 'period': 20},
        'rsi_14': {'feature': 'rsi', 'period': 14},
        'bollinger_bands_20_2.0': {'feature': 'bollinger_bands', 'period': 20, 'std_dev': 2.0},
        'macd_12_26_9': {'feature': 'macd', 'fast': 12, 'slow': 26, 'signal': 9}
    }
}

print(f"Creating feature hub component with config:")
print(f"  symbols: {container_config['symbols']}")
print(f"  features: {len(container_config['features'])} features")

# Create component
component = create_feature_hub_component(container_config)

print(f"\nComponent created: {component.name}")
print(f"Feature hub initialized: {component._feature_hub is not None}")

# Check if features were configured
hub = component._feature_hub
print(f"\nFeature hub state:")
print(f"  feature_configs: {len(hub._feature_configs)} configs")
print(f"  configured features: {list(hub._feature_configs.keys())[:5]}")

# Check feature creation
print(f"\nTesting feature creation:")
for feature_name, config in list(hub._feature_configs.items())[:3]:
    print(f"  {feature_name}: {config}")
    try:
        feature = hub._create_feature(feature_name, config)
        print(f"    -> Created successfully: {type(feature).__name__}")
    except Exception as e:
        print(f"    -> ERROR: {e}")

# Test update with a bar
bar_data = {
    'open': 100.0,
    'high': 101.0,
    'low': 99.0,
    'close': 100.5,
    'volume': 1000000
}

print(f"\nUpdating with test bar...")
hub.update_bar('SPY', bar_data)

features = hub.get_features('SPY')
print(f"\nComputed features: {len(features)}")
for name, value in list(features.items())[:5]:
    print(f"  {name}: {value}")