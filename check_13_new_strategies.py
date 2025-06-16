#!/usr/bin/env python3
"""Check the 13 new strategy additions to the config."""

import subprocess
import re
from collections import defaultdict
import yaml

# Load config to see what strategies we have
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get all strategy types from config
config_strategies = set()
for strategy in config.get('strategies', []):
    strategy_type = strategy.get('type')
    if strategy_type:
        config_strategies.add(strategy_type)

print(f"Total strategy types in config: {len(config_strategies)}")
print("Config strategy types:")
for i, strategy_type in enumerate(sorted(config_strategies), 1):
    print(f"  {i:2d}. {strategy_type}")

# Run test to see which are working
print(f"\nRunning test to check working strategies...")
result = subprocess.run([
    'python', 'main.py', 
    '--config', 'config/expansive_grid_search.yaml',
    '--signal-generation', 
    '--bars', '100'
], capture_output=True, text=True)

all_output = result.stdout + "\n" + result.stderr
signal_lines = [line for line in all_output.split('\n') if '📡' in line]

print(f"Found {len(signal_lines)} signal outputs total")

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
            # Use first part if no compound match
            working_strategies[strategy_type.split('_')[0]] += 1

working_strategy_types = set(working_strategies.keys())

print(f"\nWorking strategy types: {len(working_strategy_types)}")
for strategy_type, count in sorted(working_strategies.items()):
    print(f"  {strategy_type}: {count} signals")

# Compare config vs working
not_working = config_strategies - working_strategy_types
working_but_not_in_config = working_strategy_types - config_strategies

print(f"\n=== ANALYSIS ===")
print(f"Config strategies: {len(config_strategies)}")
print(f"Working strategies: {len(working_strategy_types)}")
print(f"Success rate: {len(working_strategy_types)}/{len(config_strategies)} = {len(working_strategy_types)/len(config_strategies)*100:.1f}%")

if not_working:
    print(f"\nNOT WORKING ({len(not_working)}):")
    for strategy in sorted(not_working):
        print(f"  ❌ {strategy}")

if working_but_not_in_config:
    print(f"\nWORKING BUT NOT IN CONFIG ({len(working_but_not_in_config)}):")
    for strategy in sorted(working_but_not_in_config):
        print(f"  ⚠️  {strategy}")

print(f"\n✅ WORKING STRATEGIES ({len(working_strategy_types)}):")
for strategy in sorted(working_strategy_types):
    print(f"  ✓ {strategy}")

# Check if any strategies mentioned in the user's "13 new additions" are working
user_mentioned = [
    'fibonacci_retracement', 'price_action_swing', 'macd_crossover', 
    'linear_regression_slope', 'ichimoku_cloud_position'
]

print(f"\n=== USER'S TARGET STRATEGIES ===")
for strategy in user_mentioned:
    status = "✓ WORKING" if strategy in working_strategy_types else "❌ NOT WORKING"
    signals = working_strategies.get(strategy, 0)
    print(f"  {strategy}: {signals} signals - {status}")