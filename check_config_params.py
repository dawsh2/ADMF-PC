#!/usr/bin/env python3
"""Check complete config for strategies with missing params."""

import yaml

with open('config/complete_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Checking all strategies for missing/null params...")

for i, strategy in enumerate(config.get('strategies', [])):
    strategy_type = strategy.get('type')
    params = strategy.get('params')
    
    if params is None:
        print(f"❌ Strategy {i+1}: {strategy_type} - params is None")
        print(f"   Full strategy: {strategy}")
    elif not params:
        print(f"⚠️  Strategy {i+1}: {strategy_type} - params is empty")
        print(f"   Full strategy: {strategy}")
    else:
        print(f"✓ Strategy {i+1}: {strategy_type} - params OK")

print(f"\nTotal strategies checked: {len(config.get('strategies', []))}")