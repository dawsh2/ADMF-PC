#!/usr/bin/env python3
"""Analyze filter combination results from saved CSV data."""

import pandas as pd
import numpy as np

# Load the trades data
trades_df = pd.read_csv('keltner_trades_with_conditions.csv')
print(f"Loaded {len(trades_df)} trades")

# Basic stats
print("\n=== OVERALL PERFORMANCE ===")
print(f"Average return: {trades_df['gross_return_bps'].mean():.2f} bps")
print(f"Net after 2bp: {trades_df['gross_return_bps'].mean() - 2:.2f} bps")
print(f"Win rate: {(trades_df['gross_return'] > 0).mean() * 100:.1f}%")

# Test the most promising individual filters
print("\n=== SINGLE FILTER RESULTS ===")

# Direction
longs = trades_df[trades_df['direction'] == 'long']
shorts = trades_df[trades_df['direction'] == 'short']
print(f"\nLongs: {len(longs)} trades, {longs['gross_return_bps'].mean():.2f} bps (net: {longs['gross_return_bps'].mean()-2:.2f})")
print(f"Shorts: {len(shorts)} trades, {shorts['gross_return_bps'].mean():.2f} bps (net: {shorts['gross_return_bps'].mean()-2:.2f})")

# Volatility > 90%
high_vol = trades_df[trades_df['volatility_rank'] > 0.9]
print(f"\nHigh Vol (>90%): {len(high_vol)} trades, {high_vol['gross_return_bps'].mean():.2f} bps (net: {high_vol['gross_return_bps'].mean()-2:.2f})")

# Below VWAP
below_vwap = trades_df[trades_df['price_to_vwap'] < 0]
near_vwap = trades_df[(trades_df['price_to_vwap'] >= -0.2) & (trades_df['price_to_vwap'] < 0)]
print(f"\nBelow VWAP: {len(below_vwap)} trades, {below_vwap['gross_return_bps'].mean():.2f} bps (net: {below_vwap['gross_return_bps'].mean()-2:.2f})")
print(f"Near VWAP (-0.2% to 0%): {len(near_vwap)} trades, {near_vwap['gross_return_bps'].mean():.2f} bps (net: {near_vwap['gross_return_bps'].mean()-2:.2f})")

# Test specific combinations
print("\n=== BEST FILTER COMBINATIONS ===")

# Longs + High Vol + Below VWAP
combo1 = trades_df[
    (trades_df['direction'] == 'long') & 
    (trades_df['volatility_rank'] > 0.9) & 
    (trades_df['price_to_vwap'] < 0)
]
print(f"\nLongs + HighVol(>90%) + BelowVWAP:")
print(f"  Trades: {len(combo1)}")
print(f"  Avg return: {combo1['gross_return_bps'].mean():.2f} bps")
print(f"  Net (2bp): {combo1['gross_return_bps'].mean() - 2:.2f} bps")
print(f"  Trades/day: {len(combo1) / 209.7:.2f}")
print(f"  Annual return: {(combo1['gross_return_bps'].mean() - 2) * len(combo1) / 209.7 * 252 / 10000 * 100:.2f}%")

# Longs + Near VWAP
combo2 = trades_df[
    (trades_df['direction'] == 'long') & 
    (trades_df['price_to_vwap'] >= -0.2) & 
    (trades_df['price_to_vwap'] < 0)
]
print(f"\nLongs + NearVWAP (-0.2% to 0%):")
print(f"  Trades: {len(combo2)}")
print(f"  Avg return: {combo2['gross_return_bps'].mean():.2f} bps")
print(f"  Net (2bp): {combo2['gross_return_bps'].mean() - 2:.2f} bps")
print(f"  Trades/day: {len(combo2) / 209.7:.2f}")

# High Vol + High Volume
combo3 = trades_df[
    (trades_df['volatility_rank'] > 0.85) & 
    (trades_df['volume_ratio'] > 1.5)
]
print(f"\nHighVol(>85%) + HighVolume(>1.5x):")
print(f"  Trades: {len(combo3)}")
print(f"  Avg return: {combo3['gross_return_bps'].mean():.2f} bps")
print(f"  Net (2bp): {combo3['gross_return_bps'].mean() - 2:.2f} bps")
print(f"  Trades/day: {len(combo3) / 209.7:.2f}")

# Test morning hours
morning = trades_df[trades_df['hour'].isin([9, 10])]
print(f"\nMorning Hours (9-10am):")
print(f"  Trades: {len(morning)}")
print(f"  Avg return: {morning['gross_return_bps'].mean():.2f} bps")
print(f"  Net (2bp): {morning['gross_return_bps'].mean() - 2:.2f} bps")

# Find any profitable combinations
print("\n=== SEARCHING FOR PROFITABLE COMBOS ===")

# Create more specific filters
filters_to_test = [
    ("Longs only", trades_df['direction'] == 'long'),
    ("Vol > 80%", trades_df['volatility_rank'] > 0.8),
    ("Vol > 90%", trades_df['volatility_rank'] > 0.9),
    ("Below VWAP", trades_df['price_to_vwap'] < 0),
    ("Near VWAP", (trades_df['price_to_vwap'] >= -0.2) & (trades_df['price_to_vwap'] < 0)),
    ("Volume > 1.5x", trades_df['volume_ratio'] > 1.5),
    ("Volume > 2x", trades_df['volume_ratio'] > 2.0),
    ("RSI < 50", trades_df['rsi'] < 50),
    ("Strong downtrend", (trades_df['trend'] == 'down') & (trades_df['trend_strength'] == 'strong')),
]

# Test each combination of 2 filters
print("\nTesting 2-filter combinations:")
best_combos = []

for i, (name1, filter1) in enumerate(filters_to_test):
    for j, (name2, filter2) in enumerate(filters_to_test[i+1:], i+1):
        combo = trades_df[filter1 & filter2]
        if len(combo) > 50:  # Need enough trades
            avg_return = combo['gross_return_bps'].mean()
            net_return = avg_return - 2
            if net_return > -0.5:  # Close to profitable
                best_combos.append({
                    'filters': f"{name1} + {name2}",
                    'trades': len(combo),
                    'avg_return': avg_return,
                    'net_return': net_return,
                    'trades_per_day': len(combo) / 209.7
                })

# Sort by net return
best_combos.sort(key=lambda x: x['net_return'], reverse=True)

print("\nTop combinations by net return:")
for combo in best_combos[:10]:
    print(f"{combo['filters']}: {combo['trades']} trades, {combo['net_return']:.2f} bps net, {combo['trades_per_day']:.2f}/day")

# Final recommendations
print("\n=== CONCLUSIONS ===")
print("\n1. No single filter or simple combination is profitable after 2bp costs")
print("2. Best single filter: Longs (+0.24 bps) - still loses after costs")
print("3. Best combination found: Longs + Near VWAP, but still unprofitable")
print("4. The 1-minute timeframe appears too noisy for Keltner Bands mean reversion")
print("\n5. Recommendations:")
print("   - Test on higher timeframes (5m, 15m)")
print("   - Try different strategies (momentum, breakout)")
print("   - Consider tighter bands or adaptive parameters")
print("   - Look for stronger entry signals (multiple confirmations)")