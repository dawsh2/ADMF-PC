#!/usr/bin/env python3
"""Diagnose the parameter expansion issue."""

import yaml
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.coordinator.config.clean_syntax_parser import parse_clean_config
from src.core.coordinator.config.parameter_expander import ParameterSpaceExpander

# Load a simple test config
test_yaml = """
name: test_expansion
data: SPY_5m

strategy:
  - keltner_bands:
      period: [20, 30]  # 2 values
      multiplier: [2.0, 3.0]  # 2 values  
      filter: [
        null,  # Should create 2×2=4 strategies
        {rsi_below: {threshold: [40, 50]}}  # Should create 2×2×2=8 strategies
      ]
"""

config = yaml.safe_load(test_yaml)

# Step 1: Clean syntax parsing
print("=" * 60)
print("STEP 1: Clean Syntax Parsing")
print("=" * 60)
parsed = parse_clean_config(config)
strategies = parsed.get('parameter_space', {}).get('strategies', [])

print(f"\nStrategies after clean syntax parsing: {len(strategies)}")
for i, strategy in enumerate(strategies):
    print(f"\n{i+1}. Type: {strategy['type']}")
    print(f"   Params: {strategy.get('param_overrides', {})}")
    print(f"   Filter: {strategy.get('filter', 'No filter')[:80]}...")

# Step 2: Parameter expansion (if --optimize was used)
print("\n" + "=" * 60)
print("STEP 2: Parameter Space Expansion (--optimize mode)")
print("=" * 60)

expander = ParameterSpaceExpander(granularity=5)
expanded_config = expander.expand_parameter_space(parsed, optimize=True)

if 'parameter_combinations' in expanded_config:
    combos = expanded_config['parameter_combinations']
    print(f"\nParameter combinations after expansion: {len(combos)}")
    
    # Group by filter
    filter_groups = {}
    for combo in combos:
        # Extract filter info from strategy name or parameters
        filter_desc = "unknown"
        if 'filter' in combo.get('parameters', {}):
            filter_desc = combo['parameters']['filter'][:50]
        elif combo.get('strategy_name', '').endswith('_0'):
            filter_desc = "baseline"
        elif combo.get('strategy_name', '').endswith('_1'):
            filter_desc = "rsi_filter"
            
        if filter_desc not in filter_groups:
            filter_groups[filter_desc] = 0
        filter_groups[filter_desc] += 1
    
    print("\nCombinations by filter type:")
    for filter_desc, count in filter_groups.items():
        print(f"  {filter_desc}: {count}")

print("\n" + "=" * 60)
print("DIAGNOSIS")
print("=" * 60)
print("\nThe issue is that filter parameter arrays (like threshold: [40, 50])")
print("are creating only ONE strategy variant instead of expanding into")
print("multiple strategies for each threshold value.")
print("\nExpected: 4 baseline + 8 RSI variants = 12 total")
print(f"Actual: {len(strategies)} strategies")