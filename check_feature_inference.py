#!/usr/bin/env python3
"""Check if feature inference is working for missing strategies."""

import yaml
import sys
sys.path.append('.')

from src.core.coordinator.topology import TopologyBuilder

# Load config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

builder = TopologyBuilder()

# Expand strategies
expanded_strategies = builder._expand_strategy_parameters(config.get('strategies', []))

# Check a few strategies that aren't generating signals
missing_strategies = ['parabolic_sar', 'supertrend', 'adx_trend_strength', 'aroon_crossover']

print("Checking feature inference for missing strategies:\n")

for strat_type in missing_strategies:
    print(f"\n{strat_type}:")
    # Find one expanded version
    for strategy in expanded_strategies:
        if strategy.get('type') == strat_type:
            # Simulate feature inference
            required_features = builder._infer_features_from_strategies([strategy])
            print(f"  Example: {strategy.get('name')}")
            print(f"  Params: {strategy.get('params')}")
            print(f"  Inferred features: {sorted(required_features)}")
            break

# Also check what features are configured
print("\n\nChecking total feature configuration:")

# Simulate full build
context = {'config': config}
all_required_features = builder._infer_features_from_strategies(expanded_strategies)
print(f"\nTotal unique features required: {len(all_required_features)}")
print(f"First 10 features: {sorted(list(all_required_features))[:10]}")

# Check if PSAR features are included
psar_features = [f for f in all_required_features if 'psar' in f]
print(f"\nPSAR features: {psar_features}")

supertrend_features = [f for f in all_required_features if 'supertrend' in f]
print(f"SuperTrend features: {supertrend_features}")

adx_features = [f for f in all_required_features if 'adx' in f]
print(f"ADX features: {adx_features}")