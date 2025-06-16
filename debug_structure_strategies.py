#!/usr/bin/env python3
"""Debug structure strategies."""

import yaml
import sys
sys.path.append('.')

from src.core.coordinator.topology import TopologyBuilder

# Load config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Find structure strategies
structure_types = ['fibonacci_retracement', 'linear_regression_slope', 'price_action_swing']
structure_strategies = []

for strategy in config.get('strategies', []):
    if strategy.get('type') in structure_types:
        structure_strategies.append(strategy)

print(f"Found {len(structure_strategies)} structure strategies in config")

for strategy in structure_strategies:
    print(f"\n{strategy.get('type')}:")
    print(f"  config: {strategy}")

if structure_strategies:
    builder = TopologyBuilder()
    
    # Test feature inference for each type
    for strategy in structure_strategies:
        print(f"\n=== {strategy.get('type')} ===")
        expanded = builder._expand_strategy_parameters([strategy])
        print(f"Expanded to {len(expanded)} instances")
        
        if expanded:
            # Show first instance
            first = expanded[0]
            print(f"First instance: {first.get('name')}")
            print(f"Params: {first.get('params')}")
            
            # Test feature inference
            inferred = builder._infer_features_from_strategies([first])
            print(f"Inferred features: {sorted(inferred)}")
            
            # Check strategy registration
            from src.core.components.discovery import get_component_registry
            import importlib
            importlib.import_module('src.strategy.strategies.indicators.structure')
            registry = get_component_registry()
            info = registry.get_component(strategy.get('type'))
            if info:
                print(f"Strategy registered: ✓")
                print(f"  feature_config: {info.metadata.get('feature_config')}")
                print(f"  param_feature_mapping: {info.metadata.get('param_feature_mapping')}")
            else:
                print(f"Strategy registered: ✗")