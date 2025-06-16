#!/usr/bin/env python3
"""Test structure strategies with different bar counts to find warmup requirements."""

import subprocess
import re
from collections import defaultdict

def test_structure_strategies(bars):
    """Test structure strategies with given number of bars."""
    print(f"Testing with {bars} bars...")
    
    result = subprocess.run([
        'python', 'main.py', 
        '--config', 'config/expansive_grid_search.yaml',
        '--signal-generation', 
        '--bars', str(bars)
    ], capture_output=True, text=True)
    
    all_output = result.stdout + "\n" + result.stderr
    signal_lines = [line for line in all_output.split('\n') if '📡' in line]
    
    # Look for structure strategies
    structure_patterns = {
        'linear_regression_slope': r'linear_regression_slope_grid',
        'fibonacci_retracement': r'fibonacci_retracement_grid', 
        'price_action_swing': r'price_action_swing_grid'
    }
    
    structure_counts = {}
    for strategy_name, pattern in structure_patterns.items():
        matches = [line for line in signal_lines if re.search(pattern, line)]
        structure_counts[strategy_name] = len(matches)
    
    return structure_counts

# Test different bar counts
bar_counts = [50, 100, 200, 300]
results = {}

for bars in bar_counts:
    results[bars] = test_structure_strategies(bars)
    print(f"  Results:")
    for strategy, count in results[bars].items():
        status = "✓" if count > 0 else "✗"
        print(f"    {strategy}: {count} signals {status}")
    print()

# Summary
print("Summary - Warmup Requirements:")
print("Bars\tLinear Reg\tFibonacci\tPrice Action")
print("-" * 50)
for bars in bar_counts:
    lr = results[bars]['linear_regression_slope']
    fib = results[bars]['fibonacci_retracement'] 
    pa = results[bars]['price_action_swing']
    print(f"{bars}\t{lr}\t\t{fib}\t\t{pa}")

# Determine minimum warmup needed
min_warmup = {}
for strategy in ['linear_regression_slope', 'fibonacci_retracement', 'price_action_swing']:
    min_bars = None
    for bars in bar_counts:
        if results[bars][strategy] > 0:
            min_bars = bars
            break
    min_warmup[strategy] = min_bars

print(f"\nMinimum warmup requirements:")
for strategy, min_bars in min_warmup.items():
    if min_bars:
        print(f"  {strategy}: {min_bars}+ bars")
    else:
        print(f"  {strategy}: >300 bars (needs more testing)")