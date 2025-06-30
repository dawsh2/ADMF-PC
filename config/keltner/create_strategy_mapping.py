#!/usr/bin/env python3
"""
Create a mapping of strategy IDs to their parameters
Save it for future reference
"""

import yaml
import json
from itertools import product
import pandas as pd

# Load the Keltner config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract Keltner strategy config
keltner_config = config['strategy'][0]['keltner_bands']

# Get parameter grids
periods = keltner_config['period']  # [10, 15, 20, 30, 50]
multipliers = keltner_config['multiplier']  # [1.0, 1.5, 2.0, 2.5, 3.0]
filters = keltner_config['filter']  # Complex list of filters

# Generate all combinations
strategy_mapping = []
strategy_id = 0

for period in periods:
    for multiplier in multipliers:
        for filter_idx, filter_config in enumerate(filters):
            mapping = {
                'strategy_id': strategy_id,
                'period': period,
                'multiplier': multiplier,
                'filter_index': filter_idx,
                'filter_type': 'none' if filter_config is None else type(filter_config).__name__,
                'filter_config': json.dumps(filter_config)
            }
            strategy_mapping.append(mapping)
            strategy_id += 1

# Convert to DataFrame
df = pd.DataFrame(strategy_mapping)

# Save as CSV
df.to_csv('strategy_parameter_mapping.csv', index=False)

# Also save as JSON for easy loading
with open('strategy_parameter_mapping.json', 'w') as f:
    json.dump(strategy_mapping, f, indent=2)

print(f"Created mapping for {len(strategy_mapping)} strategies")
print(f"Saved to:")
print(f"  - strategy_parameter_mapping.csv")
print(f"  - strategy_parameter_mapping.json")

# Show strategy 1029
if 1029 < len(strategy_mapping):
    print(f"\n=== Strategy 1029 ===")
    s1029 = strategy_mapping[1029]
    print(f"Period: {s1029['period']}")
    print(f"Multiplier: {s1029['multiplier']}")
    print(f"Filter index: {s1029['filter_index']}")
    print(f"Filter type: {s1029['filter_type']}")

# Summary statistics
print(f"\n=== Parameter Distribution ===")
print(f"Periods: {sorted(df['period'].unique())}")
print(f"Multipliers: {sorted(df['multiplier'].unique())}")
print(f"Filter types: {df['filter_type'].value_counts().to_dict()}")

# Create a SQL-friendly version
print("\n=== For DuckDB ===")
print("You can now load this in DuckDB:")
print("CREATE TABLE strategy_params AS SELECT * FROM read_csv_auto('strategy_parameter_mapping.csv');")
print("Then join with your traces: SELECT * FROM traces JOIN strategy_params USING (strategy_id);")