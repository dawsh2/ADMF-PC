#!/usr/bin/env python3
"""Debug MACD feature generation."""

import yaml
import sys
sys.path.append('.')

from src.core.coordinator.topology import TopologyBuilder

# Load config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Find MACD strategy
macd_strategies = [s for s in config.get('strategies', []) if s.get('type') == 'macd_crossover']
print(f"Found {len(macd_strategies)} MACD strategies in config")

if macd_strategies:
    macd_config = macd_strategies[0]
    print(f"MACD config: {macd_config}")
    
    # Test expansion
    builder = TopologyBuilder()
    expanded = builder._expand_strategy_parameters([macd_config])
    print(f"Expanded to {len(expanded)} MACD instances")
    
    # Show first few expanded instances
    for i, strategy in enumerate(expanded[:3]):
        print(f"\nMACD instance {i+1}:")
        print(f"  name: {strategy.get('name')}")
        print(f"  params: {strategy.get('params')}")
    
    # Test feature inference
    inferred = builder._infer_features_from_strategies(expanded[:3])
    print(f"\nInferred features: {sorted(inferred)}")
    
    # Look for MACD features specifically
    macd_features = [f for f in inferred if 'macd' in f.lower()]
    print(f"MACD features: {macd_features}")
else:
    print("No MACD strategies found in config")