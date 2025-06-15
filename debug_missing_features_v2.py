#!/usr/bin/env python3
"""Debug which features are missing for strategies that aren't generating signals."""

import yaml
from collections import defaultdict

# Load the grid search config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get all strategy types from config
all_strategy_types = set()
for strategy in config['strategies']:
    strategy_type = strategy['type']
    all_strategy_types.add(strategy_type)

print(f"=== ALL STRATEGIES IN CONFIG ===")
print(f"Total: {len(all_strategy_types)} strategy types")
for strategy_type in sorted(all_strategy_types):
    print(f"  {strategy_type}")

# Import strategy modules to get feature requirements
import sys
sys.path.append('/Users/daws/ADMF-PC')

from src.core.components.discovery import get_component_registry

# Import all indicator modules
import importlib
indicator_modules = [
    'src.strategy.strategies.indicators.crossovers',
    'src.strategy.strategies.indicators.oscillators', 
    'src.strategy.strategies.indicators.volatility',
    'src.strategy.strategies.indicators.volume',
    'src.strategy.strategies.indicators.trend',
    'src.strategy.strategies.indicators.structure',
]

for module_path in indicator_modules:
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        print(f"Could not import {module_path}: {e}")

registry = get_component_registry()

# Analyze each strategy
print("\n=== STRATEGY FEATURE REQUIREMENTS ===")
feature_usage = defaultdict(set)
strategies_by_feature_count = defaultdict(list)

for strategy_type in sorted(all_strategy_types):
    strategy_info = registry.get_component(strategy_type)
    if strategy_info:
        feature_config = strategy_info.metadata.get('feature_config', [])
        if isinstance(feature_config, list):
            print(f"\n{strategy_type}:")
            print(f"  Features: {feature_config}")
            strategies_by_feature_count[len(feature_config)].append(strategy_type)
            for feature in feature_config:
                feature_usage[feature].add(strategy_type)
    else:
        print(f"\n{strategy_type}: NOT FOUND IN REGISTRY")

print("\n=== STRATEGIES BY FEATURE COUNT ===")
for count in sorted(strategies_by_feature_count.keys()):
    strategies = strategies_by_feature_count[count]
    print(f"\n{count} features: {len(strategies)} strategies")
    for s in sorted(strategies):
        print(f"  - {s}")

print("\n=== FEATURE USAGE SUMMARY ===")
for feature, strategies in sorted(feature_usage.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"\n{feature} (used by {len(strategies)} strategies):")
    for s in sorted(strategies):
        print(f"  - {s}")

# Check which features are in the incremental system
print("\n=== INCREMENTAL FEATURE AVAILABILITY ===")
from src.strategy.components.features.incremental import IncrementalFeatureHub

# Create a test hub to see what features it supports
hub = IncrementalFeatureHub()
test_configs = {}

# Test each feature type
for feature in sorted(feature_usage.keys()):
    test_config = {f"test_{feature}": {"type": feature}}
    try:
        hub.configure_features(test_config)
        hub._create_feature(f"test_{feature}", test_config[f"test_{feature}"])
        print(f"✓ {feature} - SUPPORTED")
    except Exception as e:
        print(f"✗ {feature} - NOT SUPPORTED: {str(e)}")