#!/usr/bin/env python3
"""Quick test of strategies to see which are working."""

import subprocess
import re
from collections import defaultdict

print("Running quick 50-bar test to check strategy status...")

result = subprocess.run([
    'python', 'main.py', 
    '--config', 'config/expansive_grid_search.yaml',
    '--signal-generation', 
    '--bars', '50'
], capture_output=True, text=True)

all_output = result.stdout + "\n" + result.stderr
signal_lines = [line for line in all_output.split('\n') if '📡' in line]

print(f"Found {len(signal_lines)} signal outputs total")

# Check target strategies
target_patterns = {
    'linear_regression_slope': r'linear_regression_slope_grid',
    'fibonacci_retracement': r'fibonacci_retracement_grid', 
    'price_action_swing': r'price_action_swing_grid',
    'macd_crossover': r'macd_crossover_grid',
    'ichimoku': r'ichimoku_grid'
}

for strategy_name, pattern in target_patterns.items():
    matches = [line for line in signal_lines if re.search(pattern, line)]
    count = len(matches)
    status = "✓ WORKING" if count > 0 else "✗ NOT WORKING"
    print(f"  {strategy_name}: {count} signals - {status}")
    
    if count > 0:
        # Show first example
        example = matches[0]
        match = re.search(r'📡 SIGNAL: (SPY_[^→]+)', example)
        if match:
            print(f"    Example: {match.group(1)}")

# Count unique strategy types
strategy_types = defaultdict(int)
for line in signal_lines:
    match = re.search(r'📡 SIGNAL: SPY_(.+?)_grid_', line)
    if match:
        strategy_type = match.group(1)
        strategy_types[strategy_type] += 1

print(f"\nTop 10 working strategy types:")
for strategy_type, count in sorted(strategy_types.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {strategy_type}: {count}")

print(f"\nTotal unique strategy types: {len(strategy_types)}")

# Check if any errors occurred
if result.returncode != 0:
    print(f"\nError occurred (return code: {result.returncode})")
    if result.stderr:
        print("Error output:")
        print(result.stderr[-1000:])  # Last 1000 chars