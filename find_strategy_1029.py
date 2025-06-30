#!/usr/bin/env python3
"""Find which strategy type contains strategy_1029."""

import yaml
from itertools import product

# Load config
with open('config/complete_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

target_id = 1029
counter = 0

for idx, strategy in enumerate(config.get('strategies', [])):
    params = strategy.get('params', {})
    
    # Count combinations
    num_combos = 1
    param_details = []
    for param_name, param_values in params.items():
        if isinstance(param_values, list):
            num_combos *= len(param_values)
            param_details.append(f"{param_name}: {param_values}")
        else:
            param_details.append(f"{param_name}: {param_values}")
    
    start_id = counter
    end_id = counter + num_combos - 1
    
    if start_id <= target_id <= end_id:
        print(f"Found strategy_1029!")
        print(f"Strategy type: {strategy['type']}")
        print(f"Strategy name: {strategy['name']}")
        print(f"ID range: strategy_{start_id} - strategy_{end_id}")
        print(f"Position within this strategy: {target_id - start_id} (0-indexed)")
        print(f"\nParameter grids:")
        for detail in param_details:
            print(f"  {detail}")
        
        # Now decode the specific combination
        print(f"\nDecoding specific combination for strategy_1029...")
        combo_index = target_id - start_id
        
        # Extract parameter values
        param_names = list(params.keys())
        param_values = [params[name] if isinstance(params[name], list) else [params[name]] 
                       for name in param_names]
        
        # Convert index to specific parameter values
        indices = []
        temp = combo_index
        for values in reversed(param_values):
            indices.append(temp % len(values))
            temp //= len(values)
        indices.reverse()
        
        print(f"\nStrategy_1029 parameters:")
        for i, param_name in enumerate(param_names):
            print(f"  {param_name}: {param_values[i][indices[i]]}")
        
        break
    
    counter += num_combos