#!/usr/bin/env python3
"""
Analyze the strategies table to see which strategies were registered.
"""

import duckdb
import pandas as pd

# Connect to the analytics database
db_path = 'workspaces/expansive_grid_search_8c6c181f/analytics.duckdb'
conn = duckdb.connect(db_path, read_only=True)

print("=== Strategy Analysis ===\n")

# Get all strategies
strategies_df = conn.execute("""
    SELECT strategy_type, strategy_name, parameters
    FROM strategies
    ORDER BY strategy_type, strategy_name
""").df()

print(f"Total strategies in database: {len(strategies_df)}")

# Group by strategy type
print("\n=== Strategy Types ===")
type_counts = strategies_df['strategy_type'].value_counts()
for stype, count in type_counts.items():
    print(f"  - {stype}: {count} instances")

# Check for our missing strategies
print("\n=== Checking Missing Strategies ===")
missing_types = [
    'accumulation_distribution',
    'adx_trend_strength', 
    'aroon_crossover',
    'bollinger_breakout',
    'donchian_breakout',
    'fibonacci_retracement',
    'ichimoku_cloud_position',
    'keltner_breakout',
    'linear_regression_slope',
    'macd_crossover',
    'obv_trend',
    'parabolic_sar',
    'pivot_points',
    'price_action_swing',
    'roc_threshold',
    'stochastic_crossover',
    'stochastic_rsi',
    'supertrend',
    'support_resistance_breakout',
    'ultimate_oscillator',
    'vortex_crossover',
    'vwap_deviation'
]

found_missing = []
not_found = []

for mtype in missing_types:
    count = len(strategies_df[strategies_df['strategy_type'] == mtype])
    if count > 0:
        found_missing.append((mtype, count))
    else:
        not_found.append(mtype)

print(f"\nMissing strategies that ARE in database ({len(found_missing)}):")
for stype, count in found_missing:
    print(f"  ✓ {stype}: {count} instances")
    # Show first instance parameters
    first = strategies_df[strategies_df['strategy_type'] == stype].iloc[0]
    print(f"    First params: {first['parameters']}")

print(f"\nMissing strategies NOT in database ({len(not_found)}):")
for stype in not_found:
    print(f"  ✗ {stype}")

# Check a sample of parameters to understand structure
print("\n=== Sample Strategy Parameters ===")
sample = strategies_df[strategies_df['strategy_type'].isin(['supertrend', 'aroon_crossover'])].head(3)
for _, row in sample.iterrows():
    print(f"\n{row['strategy_type']} ({row['strategy_name']}):")
    print(f"  Parameters: {row['parameters']}")

conn.close()