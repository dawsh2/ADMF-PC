#!/usr/bin/env python3
"""Analyze which strategy types are working with 100 bars."""

import subprocess
import re
from collections import defaultdict, Counter
import sys

print("Running 100-bar grid search to get current working strategy list...")

# Run the command and capture output (using stdout and stderr)
result = subprocess.run([
    'python', 'main.py', 
    '--config', 'config/expansive_grid_search.yaml',
    '--signal-generation', 
    '--bars', '100'
], capture_output=True, text=True)

# Combine stdout and stderr to catch all output
all_output = result.stdout + "\n" + result.stderr

# Extract signal outputs
signal_lines = [line for line in all_output.split('\n') if '📡' in line]
print(f"Found {len(signal_lines)} signal outputs total")

if len(signal_lines) == 0:
    print("No signals found. Let's debug...")
    print("STDERR output:")
    print(result.stderr[:2000])
    print("\nSTDOUT output:")
    print(result.stdout[:1000])
    sys.exit(1)

# Extract strategy types from signal names
strategy_types = defaultdict(int)
strategy_instances = defaultdict(set)

for line in signal_lines:
    # Extract strategy name from format: 📡 SIGNAL: SPY_strategy_name_grid_params → status
    match = re.search(r'📡 SIGNAL: SPY_(.+?)_grid_', line)
    if match:
        full_name = match.group(1)
        
        # Split by underscores and try to find the base strategy name
        # Some strategies have compound names like 'stochastic_rsi', 'williams_r'
        
        # Known compound strategy names
        compound_strategies = [
            'stochastic_rsi', 'williams_r', 'ultimate_oscillator', 'roc_threshold',
            'cci_threshold', 'cci_bands', 'rsi_threshold', 'rsi_bands',
            'sma_crossover', 'ema_crossover', 'ema_sma_crossover', 'dema_crossover',
            'dema_sma_crossover', 'tema_sma_crossover', 'stochastic_crossover',
            'vortex_crossover', 'ichimoku_cloud_position', 'macd_crossover',
            'keltner_breakout', 'donchian_breakout', 'bollinger_breakout',
            'momentum_breakout', 'roc_trend', 'adx_trend_strength', 'aroon_oscillator',
            'vortex_trend', 'elder_ray', 'aroon_crossover', 'linear_regression_slope',
            'obv_trend', 'mfi_bands', 'vwap_deviation', 'chaikin_money_flow',
            'accumulation_distribution', 'pivot_points', 'fibonacci_retracement',
            'support_resistance_breakout', 'atr_channel_breakout', 'price_action_swing',
            'pivot_channel_breaks', 'pivot_channel_bounces', 'trendline_breaks',
            'trendline_bounces'
        ]
        
        # Find the longest matching compound strategy name
        strategy_type = None
        for compound in compound_strategies:
            if full_name.startswith(compound):
                strategy_type = compound
                break
        
        # If no compound match, use first part
        if strategy_type is None:
            strategy_type = full_name.split('_')[0]
        
        strategy_types[strategy_type] += 1
        strategy_instances[strategy_type].add(full_name)

print(f"\nStrategy types that generated signals:")
for strategy_type, count in sorted(strategy_types.items()):
    print(f"  {strategy_type}: {count} signals")

print(f"\nTotal unique strategy types generating signals: {len(strategy_types)}")

# Check specifically for our problem strategies
target_strategies = ['linear_regression_slope', 'fibonacci_retracement', 'price_action_swing', 'macd_crossover', 'ichimoku_cloud_position']
target_found = {s: strategy_types.get(s, 0) for s in target_strategies}

print(f"\nTarget strategies status:")
for strategy, count in target_found.items():
    status = "✓ WORKING" if count > 0 else "✗ NOT WORKING"
    print(f"  {strategy}: {count} signals - {status}")

# Show all working strategy types in a clean list
working_strategies = sorted([s for s, count in strategy_types.items() if count > 0])
print(f"\nAll {len(working_strategies)} working strategy types:")
for i, strategy in enumerate(working_strategies, 1):
    print(f"  {i:2d}. {strategy}")

# Show non-working from our target list
non_working = [s for s in target_strategies if strategy_types.get(s, 0) == 0]
if non_working:
    print(f"\nNon-working target strategies: {non_working}")
else:
    print(f"\n🎉 All target strategies are now working!")