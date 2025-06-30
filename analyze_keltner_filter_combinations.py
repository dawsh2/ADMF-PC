#!/usr/bin/env python3
"""Test combinations of filters to find profitable Keltner Bands setups."""

import pandas as pd
import numpy as np
from itertools import combinations

# Load the detailed trades data
trades_df = pd.read_csv('keltner_trades_with_conditions.csv')
print(f"Loaded {len(trades_df)} trades for analysis\n")

# Define the promising filters based on our findings
filters = {
    # Direction
    "Longs": trades_df['direction'] == 'long',  # +0.24 bps
    
    # Volatility
    "HighVol(>90%)": trades_df['volatility_rank'] > 0.9,  # +0.60 bps
    "HighVol(>80%)": trades_df['volatility_rank'] > 0.8,
    "ModerateVol(20-60%)": (trades_df['volatility_rank'] > 0.2) & (trades_df['volatility_rank'] < 0.6),
    
    # Volume
    "HighVolume(>1.5x)": trades_df['volume_ratio'] > 1.5,  # +0.21 bps
    "HighVolume(>2x)": trades_df['volume_ratio'] > 2.0,
    
    # VWAP Position
    "BelowVWAP": trades_df['price_to_vwap'] < 0,  # +0.55 to +0.83 bps
    "NearVWAP(-0.2to0%)": (trades_df['price_to_vwap'] >= -0.2) & (trades_df['price_to_vwap'] < 0),  # +0.83 bps
    "FarBelowVWAP(<-0.2%)": trades_df['price_to_vwap'] < -0.2,  # +0.55 bps
    
    # Trend
    "StrongDowntrend": (trades_df['trend'] == 'down') & (trades_df['trend_strength'] == 'strong'),  # +0.24 bps
    "Downtrend": trades_df['trend'] == 'down',
    
    # RSI
    "RSI(30-50)": (trades_df['rsi'] >= 30) & (trades_df['rsi'] < 50),  # +0.31 bps
    "LowRSI(<50)": trades_df['rsi'] < 50,
}

# Test all combinations
print("=== TESTING FILTER COMBINATIONS ===\n")

results = []

# Test single filters first
print("--- Single Filters ---")
for name, mask in filters.items():
    subset = trades_df[mask]
    if len(subset) > 20:
        avg_return = subset['gross_return_bps'].mean()
        results.append({
            'filters': name,
            'num_filters': 1,
            'trades': len(subset),
            'avg_return_bps': avg_return,
            'net_1bp': avg_return - 1,
            'net_2bp': avg_return - 2,
            'net_4bp': avg_return - 4,
            'win_rate': (subset['gross_return'] > 0).mean() * 100,
            'trades_per_day': len(subset) / 209.7,  # Approximate trading days
            'annual_return_2bp': (avg_return - 2) * len(subset) / 209.7 * 252 / 10000 * 100
        })

# Test 2-filter combinations
print("\n--- 2-Filter Combinations ---")
for combo in combinations(filters.keys(), 2):
    mask = filters[combo[0]]
    for filter_name in combo[1:]:
        mask = mask & filters[filter_name]
    
    subset = trades_df[mask]
    if len(subset) > 20:  # Minimum trades for significance
        avg_return = subset['gross_return_bps'].mean()
        results.append({
            'filters': ' + '.join(combo),
            'num_filters': 2,
            'trades': len(subset),
            'avg_return_bps': avg_return,
            'net_1bp': avg_return - 1,
            'net_2bp': avg_return - 2,
            'net_4bp': avg_return - 4,
            'win_rate': (subset['gross_return'] > 0).mean() * 100,
            'trades_per_day': len(subset) / 209.7,
            'annual_return_2bp': (avg_return - 2) * len(subset) / 209.7 * 252 / 10000 * 100
        })

# Test 3-filter combinations
print("\n--- 3-Filter Combinations ---")
for combo in combinations(filters.keys(), 3):
    mask = filters[combo[0]]
    for filter_name in combo[1:]:
        mask = mask & filters[filter_name]
    
    subset = trades_df[mask]
    if len(subset) > 20:  # Minimum trades
        avg_return = subset['gross_return_bps'].mean()
        results.append({
            'filters': ' + '.join(combo),
            'num_filters': 3,
            'trades': len(subset),
            'avg_return_bps': avg_return,
            'net_1bp': avg_return - 1,
            'net_2bp': avg_return - 2,
            'net_4bp': avg_return - 4,
            'win_rate': (subset['gross_return'] > 0).mean() * 100,
            'trades_per_day': len(subset) / 209.7,
            'annual_return_2bp': (avg_return - 2) * len(subset) / 209.7 * 252 / 10000 * 100
        })

# Convert to DataFrame and sort
results_df = pd.DataFrame(results)

print("\n=== TOP 20 COMBINATIONS BY NET RETURN (2bp cost) ===")
top_by_return = results_df.nlargest(20, 'net_2bp')
print(f"{'Filters':<70} {'Trades':<8} {'Avg(bps)':<10} {'Net(2bp)':<10} {'Win%':<8} {'Trades/Day':<12} {'Annual%':<10}")
print("-" * 130)
for _, row in top_by_return.iterrows():
    print(f"{row['filters']:<70} {row['trades']:<8} {row['avg_return_bps']:<10.2f} {row['net_2bp']:<10.2f} "
          f"{row['win_rate']:<8.1f} {row['trades_per_day']:<12.2f} {row['annual_return_2bp']:<10.2f}")

print("\n=== PROFITABLE COMBINATIONS (Net > 0 after 2bp cost) ===")
profitable = results_df[results_df['net_2bp'] > 0].sort_values('annual_return_2bp', ascending=False)
if len(profitable) > 0:
    print(f"\nFound {len(profitable)} profitable combinations!")
    print(f"{'Filters':<70} {'Trades':<8} {'Net(2bp)':<10} {'Annual%':<10}")
    print("-" * 100)
    for _, row in profitable.head(20).iterrows():
        print(f"{row['filters']:<70} {row['trades']:<8} {row['net_2bp']:<10.2f} {row['annual_return_2bp']:<10.2f}%")
else:
    print("No profitable combinations found after 2bp cost.")

print("\n=== BEST HIGH-FREQUENCY COMBINATIONS (>1 trade/day) ===")
high_freq = results_df[results_df['trades_per_day'] > 1].nlargest(10, 'net_2bp')
print(f"{'Filters':<70} {'Trades/Day':<12} {'Net(2bp)':<10} {'Annual%':<10}")
print("-" * 105)
for _, row in high_freq.iterrows():
    print(f"{row['filters']:<70} {row['trades_per_day']:<12.2f} {row['net_2bp']:<10.2f} {row['annual_return_2bp']:<10.2f}%")

# Look for sweet spots
print("\n=== ANALYSIS SUMMARY ===")
print(f"Total combinations tested: {len(results_df)}")
print(f"Profitable at 1bp cost: {len(results_df[results_df['net_1bp'] > 0])}")
print(f"Profitable at 2bp cost: {len(results_df[results_df['net_2bp'] > 0])}")
print(f"Profitable at 4bp cost: {len(results_df[results_df['net_4bp'] > 0])}")

# Save all results
results_df.to_csv('keltner_all_filter_combinations.csv', index=False)
print("\n✓ Saved all combinations to keltner_all_filter_combinations.csv")

# Create specific filter recommendations
print("\n=== RECOMMENDED FILTERS FOR IMPLEMENTATION ===")

# Find the best balance of return and frequency
viable = results_df[(results_df['trades_per_day'] > 0.5) & (results_df['net_2bp'] > -1)]
if len(viable) > 0:
    best_balanced = viable.nlargest(5, 'annual_return_2bp')
    print("\nBest balanced filters (>0.5 trades/day, net > -1bp):")
    for _, row in best_balanced.iterrows():
        print(f"\n{row['filters']}:")
        print(f"  - {row['trades_per_day']:.1f} trades/day")
        print(f"  - {row['net_2bp']:.2f} bps net return")
        print(f"  - {row['annual_return_2bp']:.1f}% annual return")
        print(f"  - {row['win_rate']:.1f}% win rate")

# Specific high-value setups
print("\n\nHigh-value setups (may be infrequent):")
high_value = results_df[results_df['avg_return_bps'] > 2].nlargest(5, 'avg_return_bps')
for _, row in high_value.iterrows():
    if row['trades'] > 50:  # Ensure statistical significance
        print(f"\n{row['filters']}:")
        print(f"  - {row['avg_return_bps']:.2f} bps average")
        print(f"  - {row['trades']} total trades")
        print(f"  - Only {row['trades_per_day']:.2f} trades/day")

# Test some specific promising combinations not in the automatic search
print("\n\n=== TESTING SPECIFIC HYPOTHESIS-DRIVEN COMBINATIONS ===")

specific_filters = {
    "Best Overall": (
        (trades_df['direction'] == 'long') & 
        (trades_df['volatility_rank'] > 0.9) & 
        (trades_df['price_to_vwap'] < 0) & 
        (trades_df['volume_ratio'] > 1.5)
    ),
    "Near VWAP Longs": (
        (trades_df['direction'] == 'long') & 
        (trades_df['price_to_vwap'] >= -0.2) & 
        (trades_df['price_to_vwap'] < 0)
    ),
    "Downtrend Bounce": (
        (trades_df['direction'] == 'long') & 
        (trades_df['trend'] == 'down') & 
        (trades_df['trend_strength'] == 'strong') & 
        (trades_df['rsi'] < 50)
    ),
    "High Vol Mean Reversion": (
        (trades_df['volatility_rank'] > 0.85) & 
        (trades_df['volume_ratio'] > 1.5) & 
        (trades_df['price_to_vwap'] < -0.1)
    ),
    "Morning Volatility": (
        (trades_df['hour'].isin([9, 10])) & 
        (trades_df['volatility_rank'] > 0.8) & 
        (trades_df['direction'] == 'long')
    ),
}

print(f"{'Filter':<30} {'Trades':<8} {'Avg(bps)':<10} {'Net(2bp)':<10} {'Trades/Day':<12}")
print("-" * 72)
for name, mask in specific_filters.items():
    subset = trades_df[mask]
    if len(subset) > 0:
        avg_return = subset['gross_return_bps'].mean()
        print(f"{name:<30} {len(subset):<8} {avg_return:<10.2f} {avg_return-2:<10.2f} {len(subset)/209.7:<12.2f}")