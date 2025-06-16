#!/usr/bin/env python3
"""Debug ichimoku strategy specifically."""

import yaml
import sys
sys.path.append('.')

from src.core.coordinator.topology import TopologyBuilder

# Load config and find ichimoku strategy
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

ichimoku_strategies = [s for s in config.get('strategies', []) if s.get('type') == 'ichimoku_cloud_position']
print(f"Found {len(ichimoku_strategies)} ichimoku strategies in config")

if ichimoku_strategies:
    ichimoku_config = ichimoku_strategies[0]
    print(f"Ichimoku config: {ichimoku_config}")
    
    builder = TopologyBuilder()
    expanded = builder._expand_strategy_parameters([ichimoku_config])
    print(f"Expanded to {len(expanded)} ichimoku instances")
    
    # Show first few expanded instances
    for i, strategy in enumerate(expanded[:3]):
        print(f"\nIchimoku instance {i+1}:")
        print(f"  name: {strategy.get('name')}")
        print(f"  type: {strategy.get('type')}")
        print(f"  params: {strategy.get('params')}")
    
    # Test feature inference
    inferred = builder._infer_features_from_strategies(expanded[:3])
    print(f"\nInferred features: {sorted(inferred)}")
    
    # Look for ichimoku features specifically
    ichimoku_features = [f for f in inferred if 'ichimoku' in f.lower()]
    print(f"Ichimoku features: {ichimoku_features}")
    
    # Check if strategy is registered
    from src.core.components.discovery import get_component_registry
    import importlib
    importlib.import_module('src.strategy.strategies.indicators.crossovers')
    registry = get_component_registry()
    info = registry.get_component('ichimoku_cloud_position')
    if info:
        print(f"\nStrategy registered: ✓")
        print(f"  feature_config: {info.metadata.get('feature_config')}")
        print(f"  param_feature_mapping: {info.metadata.get('param_feature_mapping')}")
    else:
        print(f"\nStrategy registered: ✗")
        
    # Check if ichimoku feature exists
    from src.strategy.components.features.indicators.trend import TREND_FEATURES
    print(f"\nIchimoku in trend registry: {'ichimoku' in TREND_FEATURES}")
    if 'ichimoku' in TREND_FEATURES:
        print(f"Ichimoku class: {TREND_FEATURES['ichimoku']}")
else:
    print("No ichimoku strategies found in config")