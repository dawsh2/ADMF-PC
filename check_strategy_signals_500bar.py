#!/usr/bin/env python3
"""Check which strategy types generated signals in 500-bar run."""

import subprocess
import re
from collections import defaultdict

print("Running 500-bar grid search and analyzing signals...")

# Run the command and capture output
result = subprocess.run([
    'python', 'main.py', 
    '--config', 'config/expansive_grid_search.yaml',
    '--signal-generation', 
    '--bars', '500'
], capture_output=True, text=True)

# Extract signal outputs
signal_lines = [line for line in result.stderr.split('\n') if '📡' in line]

print(f"Found {len(signal_lines)} signal outputs")

# Extract strategy types from signal names
strategy_types = defaultdict(int)
for line in signal_lines:
    # Extract strategy name from format: 📡 SIGNAL: SPY_strategy_name_grid_params → status
    match = re.search(r'📡 SIGNAL: SPY_(.+?)_grid_', line)
    if match:
        strategy_type = match.group(1)
        strategy_types[strategy_type] += 1

print(f"\nStrategy types that generated signals:")
for strategy_type, count in sorted(strategy_types.items()):
    print(f"  {strategy_type}: {count} signals")

print(f"\nTotal unique strategy types generating signals: {len(strategy_types)}")

# Check specifically for structure strategies
structure_strategies = ['linear_regression_slope', 'fibonacci_retracement', 'price_action_swing']
structure_found = {s: strategy_types.get(s, 0) for s in structure_strategies}

print(f"\nStructure strategies status:")
for strategy, count in structure_found.items():
    status = "✓ WORKING" if count > 0 else "✗ NOT WORKING"
    print(f"  {strategy}: {count} signals - {status}")

# Also check the remaining problem strategies
problem_strategies = ['macd_crossover', 'ichimoku_cloud_position']
problem_found = {s: strategy_types.get(s, 0) for s in problem_strategies}

print(f"\nRemaining problem strategies status:")
for strategy, count in problem_found.items():
    status = "✓ WORKING" if count > 0 else "✗ NOT WORKING"
    print(f"  {strategy}: {count} signals - {status}")