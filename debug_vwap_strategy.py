#!/usr/bin/env python3
"""Debug VWAP deviation strategy."""

import yaml
import sys
sys.path.append('.')

from src.core.coordinator.topology import TopologyBuilder

# Load config and find VWAP strategy
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

vwap_strategies = [s for s in config.get('strategies', []) if s.get('type') == 'vwap_deviation']
print(f"Found {len(vwap_strategies)} VWAP strategies in config")

if vwap_strategies:
    vwap_config = vwap_strategies[0]
    print(f"VWAP config: {vwap_config}")
    
    builder = TopologyBuilder()
    expanded = builder._expand_strategy_parameters([vwap_config])
    print(f"Expanded to {len(expanded)} VWAP instances")
    
    # Show first few expanded instances
    for i, strategy in enumerate(expanded[:3]):
        print(f"\nVWAP instance {i+1}:")
        print(f"  name: {strategy.get('name')}")
        print(f"  params: {strategy.get('params')}")
    
    # Test feature inference
    inferred = builder._infer_features_from_strategies(expanded[:3])
    print(f"\nInferred features: {sorted(inferred)}")
    
    # Look for VWAP features specifically
    vwap_features = [f for f in inferred if 'vwap' in f.lower()]
    print(f"VWAP features: {vwap_features}")
    
    # Check if VWAP feature is registered
    from src.strategy.components.features.indicators.volatility import VOLATILITY_FEATURES
    print(f"\nVWAP in volatility registry: {'vwap' in VOLATILITY_FEATURES}")
    if 'vwap' in VOLATILITY_FEATURES:
        print(f"VWAP class: {VOLATILITY_FEATURES['vwap']}")
else:
    print("No VWAP strategies found in config")