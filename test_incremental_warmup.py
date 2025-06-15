#!/usr/bin/env python3
"""Test incremental feature warmup period."""

import logging
from src.core.containers.components.feature_hub_component import create_feature_hub_component

# Setup logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        'bollinger_bands_20_2.0': {'feature': 'bollinger_bands', 'period': 20, 'std_dev': 2.0}
    }
}

# Create component
component = create_feature_hub_component(container_config)
hub = component._feature_hub

print(f"Testing incremental feature warmup...")
print(f"Configured features: {list(hub.feature_configs.keys())}")

# Feed multiple bars to warm up features
for i in range(30):
    bar_data = {
        'open': 100.0 + i * 0.1,
        'high': 101.0 + i * 0.1,
        'low': 99.0 + i * 0.1,
        'close': 100.5 + i * 0.1,
        'volume': 1000000 + i * 1000
    }
    
    hub.update_bar('SPY', bar_data)
    
    if i % 5 == 0 or i == 29:
        features = hub.get_features('SPY')
        print(f"\nAfter bar {i+1}:")
        print(f"  Total features computed: {len(features)}")
        if features:
            for name, value in list(features.items())[:3]:
                print(f"    {name}: {value:.4f}")
        
        # Check if we have sufficient data
        sufficient = hub.has_sufficient_data('SPY')
        print(f"  Has sufficient data: {sufficient}")