#!/usr/bin/env python3
"""
Check if all strategies executed after fixes
"""
import json
from collections import defaultdict

workspace = "workspaces/expansive_grid_search_6417bbc4"

# Load metadata
with open(f"{workspace}/metadata.json", "r") as f:
    metadata = json.load(f)

print(f"=== RUN SUMMARY ===")
print(f"Total bars processed: {metadata.get('total_bars', 'N/A')}")

# Count components
total_strategies = 0
strategy_types = defaultdict(int)

for component_id, component_data in metadata['components'].items():
    if component_data['component_type'] == 'strategy':
        total_strategies += 1
        strategy_type = component_data.get('strategy_type', 'unknown').replace('_grid', '')
        strategy_types[strategy_type] += 1

print(f"\nTotal strategies executed: {total_strategies}")
print(f"Strategy types: {len(strategy_types)}")

# Expected strategies
expected_strategies = [
    'sma_crossover', 'ema_crossover', 'ema_sma_crossover', 'dema_crossover', 'dema_sma_crossover',
    'tema_sma_crossover', 'macd_crossover', 'stochastic_crossover', 'vortex_crossover',
    'ichimoku_cloud_position', 'rsi_threshold', 'rsi_bands', 'cci_threshold', 'cci_bands',
    'stochastic_rsi', 'williams_r', 'roc_threshold', 'ultimate_oscillator',
    'bollinger_breakout', 'donchian_breakout', 'keltner_breakout',
    'obv_trend', 'mfi_bands', 'vwap_deviation', 'chaikin_money_flow', 'accumulation_distribution',
    'adx_trend_strength', 'parabolic_sar', 'aroon_crossover', 'supertrend', 'linear_regression_slope',
    'pivot_points', 'fibonacci_retracement', 'support_resistance_breakout', 'atr_channel_breakout', 'price_action_swing'
]

executed_types = set(strategy_types.keys())
missing_types = set(expected_strategies) - executed_types

print(f"\n=== EXECUTION SUMMARY ===")
print(f"Expected strategy types: {len(expected_strategies)}")
print(f"Executed strategy types: {len(executed_types)}")
print(f"Coverage: {len(executed_types)/len(expected_strategies)*100:.1f}%")

if missing_types:
    print(f"\nStill missing {len(missing_types)} strategy types:")
    for mt in sorted(missing_types):
        print(f"  - {mt}")
else:
    print("\n✅ ALL STRATEGY TYPES EXECUTED!")

print(f"\n=== STRATEGY TYPE BREAKDOWN ===")
for st, count in sorted(strategy_types.items()):
    status = "✓" if st in expected_strategies else "?"
    print(f"{status} {st}: {count} configurations")

# Check if we hit 888
print(f"\n{'✅' if total_strategies == 888 else '⚠️ '} Total: {total_strategies}/888 strategies")