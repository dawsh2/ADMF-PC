#!/usr/bin/env python3
"""Show the ID ranges for each strategy type."""

import yaml
from itertools import product

# Load config
with open('config/complete_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Strategy ID Ranges:")
print("-" * 80)
print(f"{'Strategy Type':<30} {'Name':<30} {'ID Range':<20}")
print("-" * 80)

counter = 0

for strategy in config.get('strategies', []):
    params = strategy.get('params', {})
    
    # Count combinations
    num_combos = 1
    for param_name, param_values in params.items():
        if isinstance(param_values, list):
            num_combos *= len(param_values)
    
    start_id = counter
    end_id = counter + num_combos - 1
    
    print(f"{strategy['type']:<30} {strategy['name']:<30} strategy_{start_id:04d} - strategy_{end_id:04d}")
    
    counter += num_combos

print("-" * 80)
print(f"Total strategies: {counter}")
print(f"\nTo decode strategy_1029, it falls in the range shown above.")