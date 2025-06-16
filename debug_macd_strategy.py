#!/usr/bin/env python3
"""Debug MACD crossover strategy specifically."""

import yaml
import sys
sys.path.append('.')

from src.core.coordinator.topology import TopologyBuilder

# Load config and find MACD strategy
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

macd_strategies = [s for s in config.get('strategies', []) if s.get('type') == 'macd_crossover']
print(f"Found {len(macd_strategies)} MACD strategies in config")

if macd_strategies:
    macd_config = macd_strategies[0]
    print(f"MACD config: {macd_config}")
    
    builder = TopologyBuilder()
    expanded = builder._expand_strategy_parameters([macd_config])
    print(f"Expanded to {len(expanded)} MACD instances")
    
    # Show first few expanded instances
    for i, strategy in enumerate(expanded[:3]):
        print(f"\nMACD instance {i+1}:")
        print(f"  name: {strategy.get('name')}")
        print(f"  type: {strategy.get('type')}")
        print(f"  params: {strategy.get('params')}")
    
    # Test feature inference
    inferred = builder._infer_features_from_strategies(expanded[:3])
    print(f"\nInferred features: {sorted(inferred)}")
    
    # Look for MACD features specifically
    macd_features = [f for f in inferred if 'macd' in f.lower()]
    print(f"MACD features: {macd_features}")
    
    # Check if strategy is registered
    from src.core.components.discovery import get_component_registry
    import importlib
    importlib.import_module('src.strategy.strategies.indicators.crossovers')
    registry = get_component_registry()
    info = registry.get_component('macd_crossover')
    if info:
        print(f"\nStrategy registered: ✓")
        print(f"  feature_config: {info.metadata.get('feature_config')}")
        print(f"  param_feature_mapping: {info.metadata.get('param_feature_mapping')}")
    else:
        print(f"\nStrategy registered: ✗")
        
    # Check if MACD feature exists
    from src.strategy.components.features.indicators.momentum import MOMENTUM_FEATURES
    print(f"\nMACD in momentum registry: {'macd' in MOMENTUM_FEATURES}")
    if 'macd' in MOMENTUM_FEATURES:
        print(f"MACD class: {MOMENTUM_FEATURES['macd']}")
        
    # Also check the strategy param mapping
    if info and info.metadata.get('param_feature_mapping'):
        param_mapping = info.metadata['param_feature_mapping']
        print(f"\nParameter mapping:")
        for param, feature_template in param_mapping.items():
            print(f"  {param} -> {feature_template}")
        
        # Test with first instance params
        first_params = expanded[0]['params']
        print(f"\nFirst instance params: {first_params}")
        print("Expected feature names:")
        for param, value in first_params.items():
            if param in param_mapping:
                template = param_mapping[param]
                # Replace all param placeholders
                for p, v in first_params.items():
                    template = template.replace(f'{{{p}}}', str(v))
                print(f"  {param}={value} -> {template}")
else:
    print("No MACD strategies found in config")