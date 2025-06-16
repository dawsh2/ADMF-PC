#!/usr/bin/env python3
"""Test the complete_grid_search.yaml config with all 50 strategies."""

import subprocess
import re
import yaml
from collections import defaultdict

# Load complete config
with open('config/complete_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get all strategy types from config (excluding classifiers)
config_strategies = set()
for strategy in config.get('strategies', []):
    strategy_type = strategy.get('type')
    if strategy_type and 'classifier' not in strategy_type:
        config_strategies.add(strategy_type.split('#')[0].strip())  # Remove comments

print(f"=== COMPLETE CONFIG ANALYSIS ===")
print(f"Total strategy types in complete config: {len(config_strategies)}")
print("\nAll strategy types in config:")
for i, strategy_type in enumerate(sorted(config_strategies), 1):
    print(f"  {i:2d}. {strategy_type}")

# Run test with complete config
print(f"\n=== RUNNING TEST ===")
print("Testing complete_grid_search.yaml with 100 bars...")

result = subprocess.run([
    'python', 'main.py', 
    '--config', 'config/complete_grid_search.yaml',
    '--signal-generation', 
    '--bars', '100'
], capture_output=True, text=True)

all_output = result.stdout + "\n" + result.stderr
signal_lines = [line for line in all_output.split('\n') if '📡' in line]

print(f"Found {len(signal_lines)} signal outputs total")

# Check for errors
error_lines = [line for line in all_output.split('\n') if 'ERROR' in line or 'Exception' in line or 'Traceback' in line]
if error_lines:
    print(f"\n⚠️  ERRORS DETECTED:")
    for error in error_lines[:5]:  # Show first 5 errors
        print(f"  {error}")

# Extract working strategy types
working_strategies = defaultdict(int)
for line in signal_lines:
    match = re.search(r'📡 SIGNAL: SPY_(.+?)_grid_', line)
    if match:
        strategy_type = match.group(1)
        # Handle compound names
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
            'trendline_bounces', 'parabolic_sar'
        ]
        
        # Find the longest matching compound strategy name
        found_match = None
        for compound in compound_strategies:
            if strategy_type.startswith(compound):
                found_match = compound
                break
        
        if found_match:
            working_strategies[found_match] += 1
        else:
            # Handle ichimoku special case
            if 'ichimoku' in strategy_type:
                working_strategies['ichimoku_cloud_position'] += 1
            else:
                # Use first part if no compound match
                base_name = strategy_type.split('_')[0]
                working_strategies[base_name] += 1

working_strategy_types = set(working_strategies.keys())

print(f"\n=== RESULTS ===")
print(f"Config strategies: {len(config_strategies)}")
print(f"Working strategies: {len(working_strategy_types)}")
print(f"Success rate: {len(working_strategy_types)}/{len(config_strategies)} = {len(working_strategy_types)/len(config_strategies)*100:.1f}%")

# Compare config vs working
not_working = config_strategies - working_strategy_types
working_but_not_in_config = working_strategy_types - config_strategies

if not_working:
    print(f"\n❌ NOT WORKING ({len(not_working)}):")
    for strategy in sorted(not_working):
        print(f"  ❌ {strategy}")

if working_but_not_in_config:
    print(f"\n⚠️  WORKING BUT NOT IN CONFIG ({len(working_but_not_in_config)}):")
    for strategy in sorted(working_but_not_in_config):
        print(f"  ⚠️  {strategy}")

print(f"\n✅ WORKING STRATEGIES ({len(working_strategy_types)}) with signal counts:")
for strategy in sorted(working_strategy_types):
    count = working_strategies[strategy]
    print(f"  ✓ {strategy}: {count} signals")

# Summary for our target strategies
target_strategies = ['fibonacci_retracement', 'price_action_swing', 'macd_crossover', 
                    'linear_regression_slope', 'ichimoku_cloud_position']

print(f"\n=== TARGET STRATEGIES STATUS ===")
for strategy in target_strategies:
    if strategy in working_strategy_types:
        count = working_strategies[strategy]
        print(f"  ✓ {strategy}: {count} signals - WORKING")
    else:
        print(f"  ❌ {strategy}: 0 signals - NOT WORKING")