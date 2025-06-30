#!/usr/bin/env python3
"""
Find the parameters for a specific strategy ID from a compiled run
"""

import sys
sys.path.append('../../src')

# Strategy 1029 is from a compiled grid
# The mapping would be in the compilation configuration
print("Strategy 1029 is from a compiled parameter grid.")
print("\nTo find exact parameters, check:")
print("1. The original compilation script that generated the strategies")
print("2. Any metadata files saved during compilation")
print("3. The strategy ID mapping in the compiled code")

# Common Keltner parameter ranges:
print("\nTypical Keltner parameter grid:")
print("- period: [10, 15, 20, 25, 30, 40, 50]")
print("- multiplier: [1.0, 1.5, 2.0, 2.5, 3.0]")
print("- atr_period: [10, 14, 20]")

# Calculate which combination 1029 might be
total_strategies = 2750
print(f"\nWith {total_strategies} strategies, this suggests a large parameter grid")
print("Strategy 1029 is approximately 37% through the grid")

# Estimate based on common grid sizes
periods = [10, 15, 20, 25, 30, 40, 50]
multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]  
atr_periods = [10, 14, 20]
filters = ['none', 'volatility', 'trend', 'volume']

print(f"\nIf using {len(periods)} periods × {len(multipliers)} multipliers × {len(atr_periods)} ATR periods × {len(filters)} filters")
print(f"That would be {len(periods) * len(multipliers) * len(atr_periods) * len(filters)} = 420 base combinations")
print("With additional filter parameters, could easily reach 2750 total")

# Strategy 1029 position in grid
strategy_idx = 1029
print(f"\nStrategy {strategy_idx} grid position analysis:")

# Reverse engineer possible parameters (this is approximate)
n_filters = len(filters)
n_atr = len(atr_periods)  
n_mult = len(multipliers)
n_period = len(periods)

filter_idx = strategy_idx // (n_period * n_mult * n_atr)
remainder = strategy_idx % (n_period * n_mult * n_atr)

atr_idx = remainder // (n_period * n_mult)
remainder = remainder % (n_period * n_mult)

mult_idx = remainder // n_period
period_idx = remainder % n_period

if filter_idx < len(filters) and atr_idx < len(atr_periods) and mult_idx < len(multipliers) and period_idx < len(periods):
    print(f"Possible parameters:")
    print(f"- period: {periods[period_idx]}")
    print(f"- multiplier: {multipliers[mult_idx]}")
    print(f"- atr_period: {atr_periods[atr_idx]}")
    print(f"- filter: {filters[filter_idx]}")
else:
    print("Grid structure is different than assumed")