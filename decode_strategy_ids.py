#!/usr/bin/env python3
"""
Script to decode strategy IDs from the complete grid search configuration.
Maps strategy_XXXX IDs to their actual parameter configurations.
"""

import yaml
from itertools import product
from pathlib import Path
import json
import argparse

def load_config(config_path="config/complete_grid_search.yaml"):
    """Load the grid search configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def expand_strategy_parameters(strategy_config):
    """Expand parameter grid for a single strategy."""
    params = strategy_config.get('params', {})
    
    # Get all parameter names and their value lists
    param_names = list(params.keys())
    param_values = [params[name] if isinstance(params[name], list) else [params[name]] 
                   for name in param_names]
    
    # Generate all combinations
    combinations = []
    for values in product(*param_values):
        combo = {
            'type': strategy_config['type'],
            'name': strategy_config['name'],
            'params': dict(zip(param_names, values))
        }
        combinations.append(combo)
    
    return combinations

def generate_strategy_mapping(config):
    """Generate mapping of strategy IDs to their configurations."""
    all_strategies = []
    
    # Expand all strategy parameters
    for strategy in config.get('strategies', []):
        expanded = expand_strategy_parameters(strategy)
        all_strategies.extend(expanded)
    
    # Create mapping with strategy_XXXX IDs
    strategy_mapping = {}
    for idx, strategy in enumerate(all_strategies):
        strategy_id = f"strategy_{idx}"
        strategy_mapping[strategy_id] = strategy
    
    return strategy_mapping

def find_strategy_by_id(strategy_id, mapping):
    """Find a specific strategy by its ID."""
    return mapping.get(strategy_id)

def main():
    parser = argparse.ArgumentParser(description='Decode strategy IDs from grid search')
    parser.add_argument('--strategy-id', type=str, help='Specific strategy ID to decode (e.g., strategy_1029)')
    parser.add_argument('--config', type=str, default='config/complete_grid_search.yaml',
                       help='Path to grid search config file')
    parser.add_argument('--output', type=str, help='Output file for full mapping (JSON)')
    parser.add_argument('--search-type', type=str, help='Search for all strategies of a specific type')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Generate mapping
    print("Generating strategy mapping...")
    mapping = generate_strategy_mapping(config)
    print(f"Total strategies: {len(mapping)}")
    
    # Handle specific requests
    if args.strategy_id:
        strategy = find_strategy_by_id(args.strategy_id, mapping)
        if strategy:
            print(f"\n{args.strategy_id}:")
            print(json.dumps(strategy, indent=2))
        else:
            print(f"\nStrategy {args.strategy_id} not found!")
            print(f"Valid range: strategy_0 to strategy_{len(mapping)-1}")
    
    if args.search_type:
        print(f"\nStrategies of type '{args.search_type}':")
        found = 0
        for sid, strategy in mapping.items():
            if strategy['type'] == args.search_type:
                print(f"\n{sid}:")
                print(json.dumps(strategy, indent=2))
                found += 1
        print(f"\nFound {found} strategies of type '{args.search_type}'")
    
    if args.output:
        # Save full mapping to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"\nFull mapping saved to: {args.output}")
    
    # If no specific action requested, show summary
    if not (args.strategy_id or args.search_type or args.output):
        print("\nStrategy type summary:")
        type_counts = {}
        for strategy in mapping.values():
            stype = strategy['type']
            type_counts[stype] = type_counts.get(stype, 0) + 1
        
        for stype, count in sorted(type_counts.items()):
            print(f"  {stype}: {count} combinations")

if __name__ == "__main__":
    main()