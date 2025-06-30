#!/usr/bin/env python3
"""
Check the distribution of per-trade returns to understand the data.
"""

import pandas as pd
import numpy as np

# Load the advanced analysis results
df = pd.read_csv("advanced_analysis_signal_generation_a2d31737.csv")

print("=== Per-Trade Return Distribution ===")
print(f"Total strategies: {len(df)}")
print(f"Strategies with data: {(df['num_trades'] > 0).sum()}")
print()

# Filter to strategies with trades
df_with_trades = df[df['num_trades'] > 0]

print(f"Return per trade statistics (bps):")
print(f"  Mean: {df_with_trades['avg_return_per_trade_bps'].mean():.3f}")
print(f"  Median: {df_with_trades['avg_return_per_trade_bps'].median():.3f}")
print(f"  Std Dev: {df_with_trades['avg_return_per_trade_bps'].std():.3f}")
print(f"  Min: {df_with_trades['avg_return_per_trade_bps'].min():.3f}")
print(f"  Max: {df_with_trades['avg_return_per_trade_bps'].max():.3f}")
print()

# Percentiles
print("Percentiles (bps):")
for p in [99, 95, 90, 75, 50, 25, 10, 5, 1]:
    val = df_with_trades['avg_return_per_trade_bps'].quantile(p/100)
    print(f"  {p}th: {val:.3f}")
print()

# Trades per day
print(f"Trades per day statistics:")
print(f"  Mean: {df_with_trades['trades_per_day'].mean():.2f}")
print(f"  Median: {df_with_trades['trades_per_day'].median():.2f}")
print(f"  Max: {df_with_trades['trades_per_day'].max():.2f}")
print()

# Check different thresholds
print("Strategies meeting different criteria:")
for min_bps in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for min_tpd in [0.1, 0.2, 0.3, 0.4, 0.5]:
        count = ((df_with_trades['avg_return_per_trade_bps'] >= min_bps) & 
                 (df_with_trades['trades_per_day'] >= min_tpd)).sum()
        if count > 0:
            print(f"  >={min_bps:.1f} bps & >={min_tpd:.1f} trades/day: {count} strategies")

# Look at top performers
print("\n=== Top 20 by Return per Trade ===")
top_20 = df_with_trades.nlargest(20, 'avg_return_per_trade_bps')
for idx, row in top_20.iterrows():
    print(f"{row['strategy_id']}: {row['avg_return_per_trade_bps']:.3f} bps, "
          f"{row['trades_per_day']:.2f} trades/day, "
          f"{row['win_rate']*100:.1f}% win rate, "
          f"{row['num_trades']} total trades")

# Check stop loss impact
print("\n=== Stop Loss Impact ===")
print(f"Average stop rate: {df_with_trades['stop_rate'].mean()*100:.1f}%")
print(f"Strategies with >10% stops: {(df_with_trades['stop_rate'] > 0.1).sum()}")
print(f"Strategies with >20% stops: {(df_with_trades['stop_rate'] > 0.2).sum()}")

# EOD exits
print("\n=== EOD Exit Impact ===")
print(f"Average EOD exit rate: {df_with_trades['eod_exit_rate'].mean()*100:.1f}%")
print(f"Strategies with >10% EOD exits: {(df_with_trades['eod_exit_rate'] > 0.1).sum()}")
print(f"Strategies with >20% EOD exits: {(df_with_trades['eod_exit_rate'] > 0.2).sum()}")