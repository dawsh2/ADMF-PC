#!/usr/bin/env python3
"""Analyze Keltner Bands optimization results."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

workspace_path = Path("workspaces/signal_generation_11d63547")

# Load metadata
with open(workspace_path / "metadata.json", 'r') as f:
    metadata = json.load(f)

# From config: period: [10, 20, 30], multiplier: [1.5, 2.0, 2.5]
# This creates 3x3 = 9 combinations
periods = [10, 20, 30]
multipliers = [1.5, 2.0, 2.5]

# Map strategy index to parameters
strategies = []
idx = 0
for period in periods:
    for multiplier in multipliers:
        signal_changes = metadata['components'][f'SPY_compiled_strategy_{idx}']['signal_changes']
        total_bars = metadata['components'][f'SPY_compiled_strategy_{idx}']['total_bars']
        
        strategies.append({
            'strategy_idx': idx,
            'period': period,
            'multiplier': multiplier,
            'signal_changes': signal_changes,
            'total_bars': total_bars,
            'signal_frequency': signal_changes / total_bars * 100,
            'trades_per_day': signal_changes / total_bars * 390  # Assuming 390 bars per day for 1m data
        })
        idx += 1

# Convert to DataFrame
df = pd.DataFrame(strategies)

print("=== KELTNER BANDS OPTIMIZATION RESULTS ===\n")
print("Parameter mapping (3 periods × 3 multipliers):")
print(df[['strategy_idx', 'period', 'multiplier', 'signal_frequency', 'trades_per_day']])

print("\n=== ANALYSIS BY PERIOD ===")
period_analysis = df.groupby('period').agg({
    'signal_frequency': 'mean',
    'trades_per_day': 'mean',
    'signal_changes': 'sum'
}).round(2)
print(period_analysis)

print("\n=== ANALYSIS BY MULTIPLIER ===")
multiplier_analysis = df.groupby('multiplier').agg({
    'signal_frequency': 'mean',
    'trades_per_day': 'mean', 
    'signal_changes': 'sum'
}).round(2)
print(multiplier_analysis)

print("\n=== TOP STRATEGIES BY SIGNAL FREQUENCY ===")
top_strategies = df.nlargest(5, 'signal_frequency')[['strategy_idx', 'period', 'multiplier', 'signal_frequency', 'trades_per_day']]
print(top_strategies)

print("\n=== RECOMMENDED CONFIGURATIONS ===")
print("\n1. High Frequency Trading (15+ trades/day):")
high_freq = df[df['trades_per_day'] > 15]
for _, row in high_freq.iterrows():
    print(f"   Period={row['period']}, Multiplier={row['multiplier']} → {row['trades_per_day']:.1f} trades/day")

print("\n2. Medium Frequency (5-15 trades/day):")
med_freq = df[(df['trades_per_day'] >= 5) & (df['trades_per_day'] <= 15)]
for _, row in med_freq.iterrows():
    print(f"   Period={row['period']}, Multiplier={row['multiplier']} → {row['trades_per_day']:.1f} trades/day")

print("\n3. Low Frequency (<5 trades/day):")
low_freq = df[df['trades_per_day'] < 5]
for _, row in low_freq.iterrows():
    print(f"   Period={row['period']}, Multiplier={row['multiplier']} → {row['trades_per_day']:.1f} trades/day")

# Now analyze actual signals from one strategy
print("\n=== SAMPLE SIGNAL ANALYSIS (Strategy 2 - Most Active) ===")
signal_file = workspace_path / "traces/SPY_1m/signals/keltner_bands/SPY_compiled_strategy_2.parquet"

if signal_file.exists():
    signals_df = pd.read_parquet(signal_file)
    print(f"Total signal changes: {len(signals_df)}")
    print(f"Signal values: {signals_df['val'].unique()}")
    
    # Count signal types
    signal_counts = signals_df['val'].value_counts()
    print("\nSignal distribution:")
    for val, count in signal_counts.items():
        print(f"  Signal {val}: {count} occurrences ({count/len(signals_df)*100:.1f}%)")
    
    # Calculate average holding period
    if 'idx' in signals_df.columns:
        holding_periods = signals_df['idx'].diff().dropna()
        print(f"\nAverage bars between signals: {holding_periods.mean():.1f}")
        print(f"Median bars between signals: {holding_periods.median():.1f}")

print("\n=== NEXT STEPS ===")
print("1. Run backtest on top configurations to evaluate performance")
print("2. Test with different volatility filters") 
print("3. Combine with other indicators for confirmation")
print("4. Analyze signal quality during different market conditions")