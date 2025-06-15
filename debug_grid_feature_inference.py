#!/usr/bin/env python3
"""Debug feature inference for grid search parameters."""

import yaml
from src.core.coordinator.topology import TopologyBuilder
from src.core.coordinator.config.pattern_loader import PatternLoader

# Load config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Focus on just bollinger_breakout
bollinger_config = None
for strat in config['strategies']:
    if strat['type'] == 'bollinger_breakout':
        bollinger_config = strat
        break

print("=== BOLLINGER BREAKOUT GRID CONFIG ===")
print(f"Config: {bollinger_config}")

# Expand the grid parameters
from itertools import product
param_combinations = []
period_values = bollinger_config['params']['period']
std_dev_values = bollinger_config['params']['std_dev']

for period, std_dev in product(period_values, std_dev_values):
    param_combinations.append({'period': period, 'std_dev': std_dev})

print(f"\nParameter combinations ({len(param_combinations)}):")
for params in param_combinations:
    print(f"  {params}")

# Now run feature inference
topology_builder = TopologyBuilder()

# Create expanded strategies as the topology builder would
expanded_strategies = []
for params in param_combinations:
    expanded_strategies.append({
        'type': 'bollinger_breakout',
        'name': f"bollinger_breakout_{params['period']}_{params['std_dev']}",
        'params': params
    })

print(f"\nRunning feature inference on {len(expanded_strategies)} strategies...")

# Infer features
inferred_features = topology_builder._infer_features_from_strategies(expanded_strategies)

print(f"\nInferred features ({len(inferred_features)}):")
bollinger_features = [f for f in sorted(inferred_features) if 'bollinger' in f]
print(f"Bollinger features ({len(bollinger_features)}):")
for feat in bollinger_features:
    print(f"  {feat}")

# Check if all combinations are covered
print("\n=== COVERAGE CHECK ===")
missing = []
for params in param_combinations:
    feature_id = f"bollinger_bands_{params['period']}_{params['std_dev']}"
    if feature_id not in inferred_features:
        missing.append(feature_id)
        print(f"❌ Missing: {feature_id}")
    else:
        print(f"✓ Found: {feature_id}")

if missing:
    print(f"\n{len(missing)} features missing!")
else:
    print("\n✅ All parameter combinations have features!")

# Check feature configs
print("\n=== FEATURE CONFIG GENERATION ===")
feature_configs = {}
for feature_id in bollinger_features[:3]:  # Just first 3
    config = topology_builder._create_feature_config_from_id(feature_id)
    feature_configs[feature_id] = config
    print(f"{feature_id} -> {config}")