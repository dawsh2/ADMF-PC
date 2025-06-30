#!/usr/bin/env python3
"""Quick script to decode a specific strategy ID."""

import yaml
from itertools import product
import json
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python quick_decode_strategy.py <strategy_id>")
        print("Example: python quick_decode_strategy.py strategy_1029")
        sys.exit(1)
    
    target_id = sys.argv[1]
    
    # Extract the numeric ID
    try:
        target_num = int(target_id.replace("strategy_", ""))
    except:
        print(f"Invalid strategy ID format: {target_id}")
        sys.exit(1)
    
    # Load config
    with open('config/complete_grid_search.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Count through all strategy combinations
    counter = 0
    
    for strategy in config.get('strategies', []):
        params = strategy.get('params', {})
        
        # Get all parameter names and their value lists
        param_names = list(params.keys())
        param_values = [params[name] if isinstance(params[name], list) else [params[name]] 
                       for name in param_names]
        
        # Count combinations for this strategy
        num_combos = 1
        for values in param_values:
            num_combos *= len(values)
        
        # Check if our target is in this strategy's range
        if counter <= target_num < counter + num_combos:
            # Found it! Now generate the specific combination
            combo_index = target_num - counter
            
            # Convert index to specific parameter values
            indices = []
            temp = combo_index
            for values in reversed(param_values):
                indices.append(temp % len(values))
                temp //= len(values)
            indices.reverse()
            
            # Build the result
            result = {
                'strategy_id': target_id,
                'type': strategy['type'],
                'name': strategy['name'],
                'params': {}
            }
            
            for i, param_name in enumerate(param_names):
                result['params'][param_name] = param_values[i][indices[i]]
            
            print(json.dumps(result, indent=2))
            return
        
        counter += num_combos
    
    print(f"Strategy ID {target_id} not found. Valid range: strategy_0 to strategy_{counter-1}")

if __name__ == "__main__":
    main()