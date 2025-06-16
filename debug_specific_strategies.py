#!/usr/bin/env python3
"""Debug specific strategies that aren't working."""

import yaml
import sys
sys.path.append('.')

from src.core.coordinator.topology import TopologyBuilder
from src.core.components.discovery import get_component_registry
import importlib

# Import all strategy modules
strategy_modules = [
    'src.strategy.strategies.indicators.volume',
    'src.strategy.strategies.indicators.momentum',
    'src.strategy.strategies.indicators.structure',
    'src.strategy.strategies.indicators.crossovers'
]

for module in strategy_modules:
    try:
        importlib.import_module(module)
    except Exception as e:
        print(f"Error importing {module}: {e}")

# Get registry
registry = get_component_registry()

# Focus on problematic strategies
problem_strategies = ['obv_trend', 'vwap_deviation', 'macd_crossover', 'ichimoku_cloud_position']

print("Checking problematic strategies:")
for strategy_name in problem_strategies:
    info = registry.get_component(strategy_name)
    if info:
        print(f"\n✓ {strategy_name}: Found in registry")
        print(f"  feature_config: {info.metadata.get('feature_config')}")
        print(f"  param_feature_mapping: {info.metadata.get('param_feature_mapping')}")
    else:
        print(f"\n✗ {strategy_name}: NOT FOUND in registry")

# Load config and test feature inference
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

builder = TopologyBuilder()

# Find these strategies in config
print(f"\nChecking config for problematic strategies:")
for strategy in config.get('strategies', []):
    if strategy.get('type') in problem_strategies:
        print(f"\n{strategy.get('type')}:")
        print(f"  name: {strategy.get('name')}")  
        print(f"  params: {strategy.get('params')}")

# Test feature inference for these strategies
problem_configs = [s for s in config.get('strategies', []) if s.get('type') in problem_strategies]
expanded = builder._expand_strategy_parameters(problem_configs)
print(f"\nExpanded {len(problem_configs)} to {len(expanded)} instances")

# Test inference
inferred = builder._infer_features_from_strategies(expanded[:4])  # Test first 4
print(f"\nInferred features: {sorted(inferred)}")

# Check for specific volume features
volume_features = [f for f in inferred if any(vf in f for vf in ['obv', 'vwap', 'macd'])]
print(f"Volume/momentum features found: {volume_features}")