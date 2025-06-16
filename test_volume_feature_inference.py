#!/usr/bin/env python3
"""Test volume feature inference."""

import yaml
import sys
sys.path.append('.')

from src.core.coordinator.topology import TopologyBuilder

# Load config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

builder = TopologyBuilder()

# Find volume strategies in config
volume_strategies = []
for strategy in config.get('strategies', []):
    if strategy.get('type') in ['obv_trend', 'vwap_deviation', 'accumulation_distribution', 'chaikin_money_flow']:
        volume_strategies.append(strategy)

print(f"Found {len(volume_strategies)} volume strategies in config")

# Expand strategies
expanded_volume = builder._expand_strategy_parameters(volume_strategies)
print(f"Expanded to {len(expanded_volume)} volume strategy instances")

# Test feature inference
inferred_features = builder._infer_features_from_strategies(expanded_volume)
print(f"\nInferred features: {sorted(inferred_features)}")

# Check for volume features specifically
volume_features = [f for f in inferred_features if any(vf in f for vf in ['obv', 'vwap', 'ad', 'cmf'])]
print(f"\nVolume features found: {volume_features}")

# Check strategy registry for these strategies
from src.core.components.discovery import get_component_registry
import importlib

# Import volume module
importlib.import_module('src.strategy.strategies.indicators.volume')

registry = get_component_registry()

for strategy_type in ['obv_trend', 'vwap_deviation', 'accumulation_distribution', 'chaikin_money_flow']:
    info = registry.get_component(strategy_type)
    if info:
        print(f"\n{strategy_type}:")
        print(f"  feature_config: {info.metadata.get('feature_config')}")
        print(f"  param_feature_mapping: {info.metadata.get('param_feature_mapping')}")
    else:
        print(f"\n{strategy_type}: NOT FOUND in registry")