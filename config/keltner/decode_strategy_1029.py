#!/usr/bin/env python3
"""
Decode strategy 1029 parameters based on the Keltner config grid
"""

import yaml
from itertools import product

# Load the Keltner config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract Keltner strategy config
keltner_config = config['strategy'][0]['keltner_bands']

# Get parameter grids
periods = keltner_config['period']  # [10, 15, 20, 30, 50]
multipliers = keltner_config['multiplier']  # [1.0, 1.5, 2.0, 2.5, 3.0]
filters = keltner_config['filter']  # Complex list of filters

print(f"Keltner parameter grid:")
print(f"Periods: {periods} ({len(periods)} options)")
print(f"Multipliers: {multipliers} ({len(multipliers)} options)")
print(f"Filters: {len(filters)} filter configurations")
print(f"\nTotal combinations: {len(periods) * len(multipliers) * len(filters)}")

# Strategy 1029 is compiled_strategy_1029
# We need to figure out which combination this maps to

# Generate all combinations in order
strategy_id = 1029
combinations = list(product(periods, multipliers, range(len(filters))))

if strategy_id < len(combinations):
    period, multiplier, filter_idx = combinations[strategy_id]
    
    print(f"\n=== Strategy 1029 Parameters ===")
    print(f"Period: {period}")
    print(f"Multiplier: {multiplier}")
    print(f"Filter index: {filter_idx}")
    
    # Show the filter
    filter_config = filters[filter_idx]
    print(f"\nFilter configuration:")
    if filter_config is None:
        print("  No filter (baseline)")
    else:
        import json
        print(json.dumps(filter_config, indent=2))
else:
    print(f"\nStrategy ID {strategy_id} is out of range!")
    print(f"Maximum strategy ID: {len(combinations) - 1}")