#!/usr/bin/env python3
"""Check target strategies with precise naming."""

import subprocess
import re
from collections import defaultdict

print("Running 100-bar test to check target strategies...")

# Run the command and capture output
result = subprocess.run([
    'python', 'main.py', 
    '--config', 'config/expansive_grid_search.yaml',
    '--signal-generation', 
    '--bars', '100'
], capture_output=True, text=True)

# Combine stdout and stderr
all_output = result.stdout + "\n" + result.stderr

# Extract signal outputs
signal_lines = [line for line in all_output.split('\n') if '📡' in line]
print(f"Found {len(signal_lines)} signal outputs total")

# Look for target strategies specifically
target_patterns = {
    'linear_regression_slope': r'linear_regression_slope_grid',
    'fibonacci_retracement': r'fibonacci_retracement_grid', 
    'price_action_swing': r'price_action_swing_grid',
    'macd_crossover': r'macd_crossover_grid',
    'ichimoku_cloud_position': r'ichimoku_grid'  # Note: config uses 'ichimoku_grid' as name
}

# Count signals for each target strategy
target_counts = {}
matching_signals = {}

for strategy_name, pattern in target_patterns.items():
    matches = [line for line in signal_lines if re.search(pattern, line)]
    target_counts[strategy_name] = len(matches)
    matching_signals[strategy_name] = matches[:3]  # Store first 3 examples

print(f"\nTarget strategies status:")
for strategy, count in target_counts.items():
    status = "✓ WORKING" if count > 0 else "✗ NOT WORKING"
    print(f"  {strategy}: {count} signals - {status}")
    
    # Show examples if found
    if count > 0 and matching_signals[strategy]:
        print(f"    Examples:")
        for example in matching_signals[strategy]:
            # Extract just the signal name part
            match = re.search(r'📡 SIGNAL: (SPY_[^→]+)', example)
            if match:
                print(f"      {match.group(1)}")

# Also check for any unrecognized patterns that might be our missing strategies
print(f"\nLooking for unrecognized patterns...")

# Extract all unique signal prefixes
all_prefixes = set()
for line in signal_lines:
    match = re.search(r'📡 SIGNAL: SPY_(.+?)_grid_', line)
    if match:
        prefix = match.group(1)
        # Try to identify base strategy name
        parts = prefix.split('_')
        if len(parts) >= 2:
            all_prefixes.add('_'.join(parts[:2]))  # Take first two parts
        else:
            all_prefixes.add(parts[0])

unusual_prefixes = [p for p in sorted(all_prefixes) 
                   if any(target in p for target in ['linear', 'fibonacci', 'price', 'macd', 'ichimoku'])]

if unusual_prefixes:
    print(f"Potentially relevant prefixes found: {unusual_prefixes}")
else:
    print("No additional relevant prefixes found")

# Calculate success rate
working = sum(1 for count in target_counts.values() if count > 0)
total = len(target_counts)
print(f"\nSuccess rate: {working}/{total} = {working/total*100:.1f}%")