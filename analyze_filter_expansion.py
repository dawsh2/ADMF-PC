#!/usr/bin/env python3
"""Analyze filter expansion in keltner config."""

import yaml
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.coordinator.config.clean_syntax_parser import parse_clean_config

# Load the keltner config
with open('config/keltner/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Parse it
parsed = parse_clean_config(config)

# Analyze strategies
strategies = parsed['parameter_space']['strategies']

print(f"Total strategies generated: {len(strategies)}")

# Count filters by analyzing filter expressions
filter_counts = {}

for i, strategy in enumerate(strategies):
    filter_expr = strategy.get('filter', '')
    
    # Categorize the filter
    if not filter_expr:
        filter_type = 'baseline'
    elif 'rsi_14 < 40' in filter_expr:
        filter_type = 'rsi_40'
    elif 'rsi_14 < 50' in filter_expr:
        filter_type = 'rsi_50'
    elif 'rsi_14 < 60' in filter_expr:
        filter_type = 'rsi_60'
    elif 'rsi_14 < 70' in filter_expr:
        filter_type = 'rsi_70'
    elif 'volume > volume_sma_20 * 1.1' in filter_expr and 'rsi' not in filter_expr:
        filter_type = 'volume_1.1'
    elif 'volume > volume_sma_20 * 1.2' in filter_expr and 'rsi' not in filter_expr:
        filter_type = 'volume_1.2'
    elif 'volume > volume_sma_20 * 1.5' in filter_expr and 'rsi' not in filter_expr:
        filter_type = 'volume_1.5'
    elif 'volume > volume_sma_20 * 2.0' in filter_expr and 'rsi' not in filter_expr:
        filter_type = 'volume_2.0'
    else:
        filter_type = 'other'
    
    filter_counts[filter_type] = filter_counts.get(filter_type, 0) + 1

print("\nFilter type counts:")
for filter_type, count in sorted(filter_counts.items()):
    print(f"  {filter_type}: {count}")

# Check if RSI filters were expanded
rsi_count = sum(count for ftype, count in filter_counts.items() if ftype.startswith('rsi_'))
volume_count = sum(count for ftype, count in filter_counts.items() if ftype.startswith('volume_'))

print(f"\nRSI filter strategies: {rsi_count} (expected: at least 100 = 25 param combos × 4 thresholds)")
print(f"Volume filter strategies: {volume_count} (expected: at least 100 = 25 param combos × 4 multipliers)")

# Show first few strategies with RSI filters
print("\nFirst 5 strategies with RSI filters:")
count = 0
for i, strategy in enumerate(strategies):
    if 'rsi_14' in strategy.get('filter', '') and count < 5:
        print(f"\nStrategy {i}:")
        print(f"  Type: {strategy['type']}")
        print(f"  Params: {strategy.get('param_overrides', {})}")
        print(f"  Filter: {strategy.get('filter', 'No filter')[:100]}...")
        count += 1