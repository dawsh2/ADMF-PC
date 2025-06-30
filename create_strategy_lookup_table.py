#!/usr/bin/env python3
"""Create a CSV lookup table for strategy IDs to parameters."""

import yaml
from itertools import product
import csv
import json

# Load config
with open('config/complete_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Open output CSV
with open('strategy_lookup.csv', 'w', newline='') as csvfile:
    # We'll write: strategy_id, strategy_type, strategy_name, parameters_json
    writer = csv.writer(csvfile)
    writer.writerow(['strategy_id', 'strategy_type', 'strategy_name', 'parameters_json'])
    
    counter = 0
    
    for strategy in config.get('strategies', []):
        params = strategy.get('params', {})
        
        # Get all parameter names and their value lists
        param_names = list(params.keys())
        param_values = [params[name] if isinstance(params[name], list) else [params[name]] 
                       for name in param_names]
        
        # Generate all combinations
        for values in product(*param_values):
            strategy_id = f"strategy_{counter}"
            param_dict = dict(zip(param_names, values))
            
            writer.writerow([
                strategy_id,
                strategy['type'],
                strategy['name'],
                json.dumps(param_dict)
            ])
            
            # Check if this is our target
            if counter == 1029:
                print(f"\nFound strategy_1029:")
                print(f"Type: {strategy['type']}")
                print(f"Name: {strategy['name']}")
                print(f"Parameters: {json.dumps(param_dict, indent=2)}")
            
            counter += 1

print(f"\nTotal strategies written: {counter}")
print("Lookup table saved to: strategy_lookup.csv")