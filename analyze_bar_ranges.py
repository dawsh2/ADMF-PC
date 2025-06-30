#!/usr/bin/env python3
"""Analyze if bars have ranges that would trigger both SL and TP."""

import pandas as pd

print("=== Analyzing Bar Ranges vs Stop Loss/Take Profit ===")

# Load the data
df = pd.read_parquet("data/spy_5m_full.parquet")

# Calculate bar ranges
df['range_pct'] = ((df['high'] - df['low']) / df['low']) * 100

# Check how many bars have range > 0.175% (SL + TP)
wide_bars = df[df['range_pct'] > 0.175]
print(f"\nBars with range > 0.175% (could hit both SL and TP): {len(wide_bars)} ({len(wide_bars)/len(df)*100:.2f}%)")

# Check distribution
print("\nBar range distribution:")
print(f"Min range: {df['range_pct'].min():.4f}%")
print(f"25th percentile: {df['range_pct'].quantile(0.25):.4f}%")
print(f"Median range: {df['range_pct'].median():.4f}%")
print(f"75th percentile: {df['range_pct'].quantile(0.75):.4f}%")
print(f"95th percentile: {df['range_pct'].quantile(0.95):.4f}%")
print(f"Max range: {df['range_pct'].max():.4f}%")

# Count bars that could trigger exits
sl_range = df[df['range_pct'] > 0.075]  # Could trigger stop loss
tp_range = df[df['range_pct'] > 0.1]    # Could trigger take profit

print(f"\nBars that could trigger stop loss (range > 0.075%): {len(sl_range)} ({len(sl_range)/len(df)*100:.2f}%)")
print(f"Bars that could trigger take profit (range > 0.1%): {len(tp_range)} ({len(tp_range)/len(df)*100:.2f}%)")

# Check average move from open
df['move_from_open'] = abs((df['close'] - df['open']) / df['open']) * 100
print(f"\nAverage absolute move from open to close: {df['move_from_open'].mean():.4f}%")

# The issue might be order of operations
print("\n\n=== Potential Issue ===")
print("If a bar has range > 0.175%, it could trigger BOTH stop loss and take profit.")
print("The current code checks stop loss FIRST, then take profit.")
print("But in reality, we don't know which was hit first within the bar!")
print("\nThis could explain why trades increased from 453 to 463:")
print("- Before: Only checking close price, missing some exits")
print("- Now: Checking high/low, but might be double-exiting on wide bars")