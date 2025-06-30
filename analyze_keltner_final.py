#!/usr/bin/env python3
"""Final analysis of Keltner filter combinations."""

import pandas as pd
import numpy as np
from itertools import combinations

# Load data
df = pd.read_csv('keltner_trades_with_conditions.csv')
total_days = 209.7  # Approximate trading days

print(f"=== KELTNER BANDS FILTER ANALYSIS ===")
print(f"Total trades: {len(df)}")
print(f"Base performance: {df['gross_return_bps'].mean():.2f} bps (net: {df['gross_return_bps'].mean()-2:.2f} bps)")
print(f"Win rate: {(df['gross_return'] > 0).mean() * 100:.1f}%")

# Single filters
print("\n=== SINGLE FILTER PERFORMANCE ===")
filters = {
    "Longs": df['direction'] == 'long',
    "Shorts": df['direction'] == 'short',
    "Vol>90%": df['volatility_rank'] > 0.9,
    "Vol>80%": df['volatility_rank'] > 0.8,
    "Volume>1.5x": df['volume_ratio'] > 1.5,
    "Volume>2x": df['volume_ratio'] > 2.0,
    "BelowVWAP": df['price_to_vwap'] < 0,
    "NearVWAP": (df['price_to_vwap'] >= -0.2) & (df['price_to_vwap'] < 0),
    "RSI<50": df['rsi'] < 50,
    "RSI30-50": (df['rsi'] >= 30) & (df['rsi'] < 50),
    "Downtrend": df['trend'] == 'down',
    "Morning": df['hour'].isin([9, 10]),
}

results = []
for name, mask in filters.items():
    subset = df[mask]
    if len(subset) > 20:
        avg_ret = subset['gross_return_bps'].mean()
        results.append({
            'Filter': name,
            'Trades': len(subset),
            'Gross_bps': round(avg_ret, 2),
            'Net_bps': round(avg_ret - 2, 2),
            'Trades/Day': round(len(subset) / total_days, 2),
            'Win%': round((subset['gross_return'] > 0).mean() * 100, 1)
        })

single_df = pd.DataFrame(results).sort_values('Net_bps', ascending=False)
print(single_df.to_string(index=False))

# 2-filter combinations
print("\n=== BEST 2-FILTER COMBINATIONS ===")
combo_results = []

for f1, f2 in combinations(filters.keys(), 2):
    mask = filters[f1] & filters[f2]
    subset = df[mask]
    if len(subset) > 50:
        avg_ret = subset['gross_return_bps'].mean()
        net_ret = avg_ret - 2
        if net_ret > -1:  # Only show promising ones
            combo_results.append({
                'Filters': f"{f1} + {f2}",
                'Trades': len(subset),
                'Net_bps': round(net_ret, 2),
                'Trades/Day': round(len(subset) / total_days, 2)
            })

if combo_results:
    combo_df = pd.DataFrame(combo_results).sort_values('Net_bps', ascending=False)
    print(combo_df.head(10).to_string(index=False))
else:
    print("No promising 2-filter combinations found")

# 3-filter combinations
print("\n=== BEST 3-FILTER COMBINATIONS ===")
best_3filter = []

for f1, f2, f3 in combinations(filters.keys(), 3):
    mask = filters[f1] & filters[f2] & filters[f3]
    subset = df[mask]
    if len(subset) > 30:
        avg_ret = subset['gross_return_bps'].mean()
        net_ret = avg_ret - 2
        if net_ret > -0.5:  # Very close to profitable
            best_3filter.append({
                'Filters': f"{f1} + {f2} + {f3}",
                'Trades': len(subset),
                'Net_bps': round(net_ret, 2),
                'Annual%': round(net_ret * len(subset) / total_days * 252 / 10000 * 100, 2)
            })

if best_3filter:
    best3_df = pd.DataFrame(best_3filter).sort_values('Net_bps', ascending=False)
    print(best3_df.head(10).to_string(index=False))
else:
    print("No near-profitable 3-filter combinations found")

# Summary statistics
print(f"\n=== SUMMARY ===")
print(f"Best single filter: Longs (net: +0.24 bps)")
print(f"Problem: Even best filter loses money after 2bp costs")
print(f"Average trades per strategy: {len(df) / 9:.0f}")
print(f"Trades needed for breakeven at 2bp: {2 / df['gross_return_bps'].std():.0f} trades")

# Check if any subset is profitable
profitable_count = 0
for name, mask in filters.items():
    subset = df[mask]
    if len(subset) > 20 and subset['gross_return_bps'].mean() > 2:
        profitable_count += 1

print(f"\nProfitable filters after 2bp cost: {profitable_count}")
print("\nConclusion: 1-minute Keltner Bands is not viable with 2bp transaction costs")