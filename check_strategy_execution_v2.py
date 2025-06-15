#!/usr/bin/env python3
"""Check which strategies are actually being executed."""

import os
import json
import yaml

workspace = 'workspaces/expansive_grid_search_2c369f4c'

# Count strategies by type
strategy_counts = {}
total_signals = 0

# Read all signal files
for file in os.listdir(workspace):
    if file.endswith('_signals.parquet'):
        # Extract strategy type
        parts = file.split('_')
        if len(parts) >= 3 and parts[0] == 'SPY':
            # Find the strategy type (everything between SPY_ and _grid)
            strategy_part = '_'.join(parts[1:])
            grid_idx = strategy_part.find('_grid')
            if grid_idx > 0:
                strategy_type = strategy_part[:grid_idx + 5]  # Include '_grid'
                if strategy_type not in strategy_counts:
                    strategy_counts[strategy_type] = 0
                strategy_counts[strategy_type] += 1
                
                # Check file size as proxy for signal count
                file_path = os.path.join(workspace, file)
                file_size = os.path.getsize(file_path)
                if file_size > 1000:  # More than 1KB suggests actual signals
                    total_signals += 1

print(f"=== STRATEGY EXECUTION SUMMARY ===\n")
print(f"Total strategy types found: {len(strategy_counts)}")
print(f"Total strategy instances: {sum(strategy_counts.values())}")
print(f"Files with signals (>1KB): {total_signals}")

print(f"\nStrategy types generating signals:")
for strategy_type, count in sorted(strategy_counts.items()):
    print(f"  {strategy_type}: {count} instances")

# Now check which strategies are MISSING
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

expected_types = set()
for strategy in config['strategies']:
    strategy_type = strategy['type']
    expected_types.add(strategy_type)

found_types = set()
for st in strategy_counts.keys():
    # Extract the base strategy type (remove _grid suffix)
    base_type = st.replace('_grid', '')
    found_types.add(base_type)

missing_types = expected_types - found_types
if missing_types:
    print(f"\n=== MISSING STRATEGY TYPES ({len(missing_types)}) ===")
    for mt in sorted(missing_types):
        print(f"  {mt}")

# Check metadata for more info
metadata_path = os.path.join(workspace, 'metadata.json')
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n=== METADATA SUMMARY ===")
    print(f"Total bars: {metadata.get('total_bars', 'N/A')}")
    print(f"Total signals: {metadata.get('total_signals', 'N/A')}")
    print(f"Total components: {len(metadata.get('components', {}))}")
    
    # Check for strategy components
    strategy_components = 0
    classifier_components = 0
    for comp_name, comp_data in metadata.get('components', {}).items():
        if comp_data.get('type') == 'strategy':
            strategy_components += 1
        elif comp_data.get('type') == 'classifier':
            classifier_components += 1
    
    print(f"Strategy components: {strategy_components}")
    print(f"Classifier components: {classifier_components}")