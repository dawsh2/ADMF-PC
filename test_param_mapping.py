#!/usr/bin/env python3
"""Test param_feature_mapping implementation."""

import sys
sys.path.insert(0, '/Users/daws/ADMF-PC')

from src.core.components.discovery import get_component_registry

# Check if stochastic_crossover is registered with param_feature_mapping
registry = get_component_registry()

# Import the strategy module to trigger decorator registration
from src.strategy.strategies.indicators.crossovers import stochastic_crossover

# Now check the registry
strategy_info = registry.get_component('stochastic_crossover')

if strategy_info:
    print("✅ Strategy found in registry!")
    print(f"Strategy name: {strategy_info.name}")
    print(f"Component type: {strategy_info.component_type}")
    print(f"Metadata keys: {list(strategy_info.metadata.keys())}")
    
    # Check for param_feature_mapping
    param_mapping = strategy_info.metadata.get('param_feature_mapping')
    if param_mapping:
        print(f"✅ param_feature_mapping found: {param_mapping}")
    else:
        print("❌ No param_feature_mapping in metadata")
        print(f"Full metadata: {strategy_info.metadata}")
        
    # Check feature_config
    feature_config = strategy_info.metadata.get('feature_config')
    print(f"Feature config: {feature_config}")
    
else:
    print("❌ Strategy not found in registry")
    print("Available strategies:")
    for name in sorted(registry._components.keys()):
        component = registry._components[name]
        if component.component_type == 'strategy':
            print(f"  {name}")