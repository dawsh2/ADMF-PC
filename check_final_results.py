#!/usr/bin/env python3
"""
Check final results from 300 bar run with all fixes
"""
import json
from collections import defaultdict
import os

workspace = "workspaces/expansive_grid_search_f164d735"

# Load metadata
with open(f"{workspace}/metadata.json", "r") as f:
    metadata = json.load(f)

print(f"=== FINAL RUN SUMMARY ===")
print(f"Total bars processed: {metadata.get('total_bars', 'N/A')}")
print(f"Total signals generated: {metadata.get('total_signals', 'N/A'):,}")
print(f"Total classifications: {metadata.get('total_classifications', 'N/A'):,}")
print(f"Compression ratio: {metadata.get('compression_ratio', 'N/A'):.1f}x")

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

print(f"\n=== EXECUTION COVERAGE ===")
print(f"Expected strategy types: {len(expected_strategies)}")
print(f"Executed strategy types: {len(executed_types)}")
print(f"Coverage: {len(executed_types)/len(expected_strategies)*100:.1f}%")

if missing_types:
    print(f"\n❌ Still missing {len(missing_types)} strategy types:")
    for mt in sorted(missing_types):
        print(f"  - {mt}")
else:
    print("\n✅ ALL STRATEGY TYPES EXECUTED SUCCESSFULLY!")

# Check if we hit 888
if total_strategies == 888:
    print(f"\n✅ ALL 888 STRATEGIES EXECUTED!")
else:
    print(f"\n⚠️  Only {total_strategies}/888 strategies executed")

# Show execution breakdown
print(f"\n=== STRATEGY EXECUTION BREAKDOWN ===")
total_configured = sum(strategy_types.values())
for st, count in sorted(strategy_types.items(), key=lambda x: x[1], reverse=True):
    pct = count / total_configured * 100
    print(f"  {st}: {count} ({pct:.1f}%)")

# Check storage efficiency
print(f"\n=== STORAGE EFFICIENCY ===")
signals_dir = f"{workspace}/traces/SPY_1m/signals"
if os.path.exists(signals_dir):
    total_files = 0
    total_size = 0
    for root, dirs, files in os.walk(signals_dir):
        for file in files:
            if file.endswith('.parquet'):
                total_files += 1
                total_size += os.path.getsize(os.path.join(root, file))
    
    print(f"Signal files created: {total_files}")
    print(f"Total storage: {total_size/1024/1024:.2f} MB")
    print(f"Average per strategy: {total_size/total_strategies/1024:.1f} KB")

# Summary
print(f"\n=== FINAL SUMMARY ===")
if total_strategies == 888 and len(missing_types) == 0:
    print("🎉 SUCCESS! All 888 strategies from 36 types executed successfully!")
    print(f"Generated {metadata.get('total_signals', 0):,} signals with {metadata.get('compression_ratio', 0):.1f}x compression")
else:
    print(f"⚠️  Partial success: {total_strategies}/888 strategies executed")
    print(f"Missing {len(missing_types)} strategy types")