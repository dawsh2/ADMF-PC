#!/usr/bin/env python3
"""Quick analysis of filter combinations."""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('keltner_trades_with_conditions.csv')
print(f"Total trades: {len(df)}")
print(f"Average return: {df['gross_return_bps'].mean():.2f} bps")
print(f"Net after 2bp: {df['gross_return_bps'].mean() - 2:.2f} bps")

# Test key combinations
print("\n=== FILTER COMBINATIONS ===")

# Helper function
def analyze_filter(mask, name):
    subset = df[mask]
    if len(subset) > 0:
        avg_ret = subset['gross_return_bps'].mean()
        net_ret = avg_ret - 2
        trades_per_day = len(subset) / 209.7
        annual = net_ret * trades_per_day * 252 / 10000 * 100
        print(f"\n{name}:")
        print(f"  Trades: {len(subset)} ({trades_per_day:.2f}/day)")
        print(f"  Gross: {avg_ret:.2f} bps, Net: {net_ret:.2f} bps")
        print(f"  Annual: {annual:.2f}%")
        return net_ret > 0
    return False

# Test single filters
profitable = []

# Direction
if analyze_filter(df['direction'] == 'long', "Longs Only"):
    profitable.append("Longs")
if analyze_filter(df['direction'] == 'short', "Shorts Only"):
    profitable.append("Shorts")

# Volatility
if analyze_filter(df['volatility_rank'] > 0.9, "Vol > 90%"):
    profitable.append("Vol>90%")

# VWAP
if analyze_filter(df['price_to_vwap'] < 0, "Below VWAP"):
    profitable.append("BelowVWAP")
    
near_vwap = (df['price_to_vwap'] >= -0.2) & (df['price_to_vwap'] < 0)
if analyze_filter(near_vwap, "Near VWAP (-0.2% to 0%)"):
    profitable.append("NearVWAP")

# Volume
if analyze_filter(df['volume_ratio'] > 1.5, "Volume > 1.5x"):
    profitable.append("Vol>1.5x")

# Test best 2-filter combos
print("\n=== 2-FILTER COMBINATIONS ===")

# Longs + High Vol
combo1 = (df['direction'] == 'long') & (df['volatility_rank'] > 0.9)
analyze_filter(combo1, "Longs + Vol>90%")

# Longs + Near VWAP
combo2 = (df['direction'] == 'long') & near_vwap
analyze_filter(combo2, "Longs + Near VWAP")

# High Vol + Below VWAP
combo3 = (df['volatility_rank'] > 0.9) & (df['price_to_vwap'] < 0)
analyze_filter(combo3, "Vol>90% + Below VWAP")

# Test best 3-filter combo
print("\n=== 3-FILTER COMBINATIONS ===")

# Longs + High Vol + Below VWAP
combo4 = (df['direction'] == 'long') & (df['volatility_rank'] > 0.9) & (df['price_to_vwap'] < 0)
analyze_filter(combo4, "Longs + Vol>90% + Below VWAP")

# Longs + High Vol + High Volume
combo5 = (df['direction'] == 'long') & (df['volatility_rank'] > 0.9) & (df['volume_ratio'] > 1.5)
analyze_filter(combo5, "Longs + Vol>90% + Volume>1.5x")

# Test specific hypothesis
print("\n=== SPECIFIC HYPOTHESES ===")

# Morning high vol longs
morning_vol = (df['hour'].isin([9, 10])) & (df['volatility_rank'] > 0.8) & (df['direction'] == 'long')
analyze_filter(morning_vol, "Morning (9-10am) + Vol>80% + Longs")

# Extreme conditions
extreme = (df['volatility_rank'] > 0.95) & (df['volume_ratio'] > 2) & (df['price_to_vwap'] < -0.2)
analyze_filter(extreme, "Extreme: Vol>95% + Volume>2x + Far Below VWAP")

print("\n=== SUMMARY ===")
if profitable:
    print(f"Profitable single filters: {', '.join(profitable)}")
else:
    print("No profitable filters found after 2bp costs")
    
print("\nConclusion: Keltner Bands on 1m timeframe appears unprofitable")
print("Recommend testing on higher timeframes or different strategies")