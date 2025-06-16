#!/usr/bin/env python3
"""Analyze results from complete_grid_search.yaml test."""

import subprocess
import re
from collections import defaultdict, Counter
import yaml

# Load complete config to see all strategy types
with open('config/complete_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get all strategy types from config
config_strategies = set()
for strategy in config.get('strategies', []):
    strategy_type = strategy.get('type')
    if strategy_type:
        config_strategies.add(strategy_type)

print(f"=== COMPLETE CONFIG ANALYSIS ===")
print(f"Strategy types in config: {len(config_strategies)}")

# Run test to get signal output
print(f"\nRunning complete config test...")
result = subprocess.run([
    'python', 'main.py', 
    '--config', 'config/complete_grid_search.yaml',
    '--signal-generation', 
    '--bars', '100'
], capture_output=True, text=True)

all_output = result.stdout + "\n" + result.stderr
signal_lines = [line for line in all_output.split('\n') if '📡' in line]

print(f"Total signal outputs: {len(signal_lines)}")

# Extract working strategy types from signal names
working_strategies = defaultdict(int)
all_signal_names = []

for line in signal_lines:
    # Extract strategy name from signal format: SPY_{strategy_name}_grid_...
    match = re.search(r'📡 SIGNAL: SPY_(.+?)_grid_', line)
    if match:
        strategy_name = match.group(1)
        all_signal_names.append(strategy_name)
        
        # Handle compound strategy names - find longest match
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
            'trendline_bounces', 'parabolic_sar', 'supertrend'
        ]
        
        # Find longest matching compound name
        matched_strategy = None
        for compound in sorted(compound_strategies, key=len, reverse=True):
            if strategy_name.startswith(compound):
                matched_strategy = compound
                break
        
        if matched_strategy:
            working_strategies[matched_strategy] += 1
        else:
            # Handle ichimoku special case
            if 'ichimoku' in strategy_name:
                working_strategies['ichimoku_cloud_position'] += 1
            else:
                # Use strategy name as-is if no compound match
                working_strategies[strategy_name] += 1

working_strategy_types = set(working_strategies.keys())

print(f"\n=== RESULTS ===")
print(f"Config strategy types: {len(config_strategies)}")
print(f"Working strategy types: {len(working_strategy_types)}")
print(f"Success rate: {len(working_strategy_types)}/{len(config_strategies)} = {len(working_strategy_types)/len(config_strategies)*100:.1f}%")

# Find strategies not working
not_working = config_strategies - working_strategy_types
working_not_in_config = working_strategy_types - config_strategies

if not_working:
    print(f"\n❌ NOT WORKING ({len(not_working)}):")
    for strategy in sorted(not_working):
        print(f"  ❌ {strategy}")

if working_not_in_config:
    print(f"\n⚠️  WORKING BUT NOT IN CONFIG ({len(working_not_in_config)}):")
    for strategy in sorted(working_not_in_config):
        print(f"  ⚠️  {strategy}")

print(f"\n✅ WORKING STRATEGIES ({len(working_strategy_types)}) with signal counts:")
for strategy in sorted(working_strategy_types):
    count = working_strategies[strategy]
    print(f"  ✓ {strategy}: {count} signals")

# Check for errors
error_lines = [line for line in all_output.split('\n') if 'ERROR' in line or 'Exception' in line]
if error_lines:
    print(f"\n⚠️  ERRORS DETECTED ({len(error_lines)}):")
    for error in error_lines[:10]:  # Show first 10 errors
        print(f"  {error}")

print(f"\nFirst 10 signal strategy names:")
for name in Counter(all_signal_names).most_common(10):
    print(f"  {name[0]}: {name[1]} signals")