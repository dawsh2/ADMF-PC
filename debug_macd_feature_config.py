#!/usr/bin/env python3
"""Debug MACD feature configuration generation."""

import yaml
import sys
sys.path.append('.')

from src.core.coordinator.topology import TopologyBuilder

# Load config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Find MACD strategies
macd_strategies = [s for s in config.get('strategies', []) if s.get('type') == 'macd_crossover']
print(f"Found {len(macd_strategies)} MACD strategies")

if macd_strategies:
    builder = TopologyBuilder()
    
    # Test with first MACD strategy
    macd_config = macd_strategies[0]
    expanded = builder._expand_strategy_parameters([macd_config])
    print(f"Expanded to {len(expanded)} instances")
    
    # Test feature inference
    first_instance = expanded[0]
    print(f"First instance: {first_instance}")
    
    # Get the feature configs that would be generated
    inferred_features = builder._infer_features_from_strategies([first_instance])
    print(f"Inferred features: {inferred_features}")
    
    # Now test the actual feature config generation
    print(f"\nTesting feature config generation...")
    
    # Get all features that will be configured
    all_strategies = builder._expand_strategy_parameters(config.get('strategies', []))
    all_features = builder._infer_features_from_strategies(all_strategies)
    
    # Look for MACD features specifically
    macd_features = [f for f in all_features if f.startswith('macd_')]
    print(f"MACD features found: {macd_features[:5]}...")  # First 5
    
    # Test the feature config generation method
    feature_configs = builder._build_feature_configs(all_features)
    
    # Check MACD feature configs specifically
    print(f"\nMACD feature configs:")
    for feature_name, config in feature_configs.items():
        if feature_name.startswith('macd_'):
            print(f"  {feature_name}: {config}")
            break  # Just show first one
            
    # Also check what the builder thinks the MACD parameters should be
    print(f"\nChecking parameter mapping for MACD...")
    from src.core.components.discovery import get_component_registry
    import importlib
    importlib.import_module('src.strategy.strategies.indicators.crossovers')
    
    registry = get_component_registry()
    macd_info = registry.get_component('macd_crossover')
    if macd_info:
        print(f"MACD strategy param mapping: {macd_info.metadata.get('param_feature_mapping')}")