#!/usr/bin/env python3
"""Verify the swing pivot bounce filter results."""

import pandas as pd
import numpy as np

# Load the saved results
trades_df = pd.read_csv('swing_zones_all_trades.csv')

print("=== VERIFICATION OF RESULTS ===")
print(f"Total trades in dataset: {len(trades_df)}")
print(f"Date range: Assuming 306 trading days (April-Dec 2024)")
print(f"Overall average: {trades_df['net_return_bps'].mean():.2f} bps/trade\n")

# Define filters
filters = {
    "Vol>90 + Shorts + HighVolume": 
        (trades_df['volatility_rank'] > 0.9) & 
        (trades_df['direction'] == 'short') & 
        (trades_df['volume_ratio'] > 1.5),
    
    "Vol>90 + Downtrend + HighVolume": 
        (trades_df['volatility_rank'] > 0.9) & 
        (trades_df['trend'] == 'down') & 
        (trades_df['volume_ratio'] > 1.5),
    
    "Vol>85 + Shorts + HighVolume": 
        (trades_df['volatility_rank'] > 0.85) & 
        (trades_df['direction'] == 'short') & 
        (trades_df['volume_ratio'] > 1.5),
}

print("=== FILTER VERIFICATION ===")
for name, mask in filters.items():
    filtered = trades_df[mask]
    
    print(f"\n{name}:")
    print(f"  Total trades: {len(filtered)}")
    print(f"  Trades per day: {len(filtered) / 306:.1f}")
    
    # Check the actual returns
    avg_return = filtered['net_return'].mean()
    avg_bps = filtered['net_return_bps'].mean()
    
    print(f"  Average return: {avg_return:.6f} ({avg_bps:.2f} bps)")
    print(f"  Win rate: {(filtered['net_return'] > 0).mean()*100:.1f}%")
    
    # Sample of trades
    print(f"  Sample returns (first 10): {filtered['net_return_bps'].head(10).values}")
    
    # Calculate annualized returns
    total_trades_per_year = len(filtered) * 12 / 10  # Scale to full year
    
    print(f"\n  Annualized calculations:")
    print(f"  Trades per year: {total_trades_per_year:.0f}")
    
    # Different cost scenarios
    for cost_bps in [0.5, 1.0, 2.0]:
        net_bps = avg_bps - cost_bps
        annual_return = (net_bps / 10000) * total_trades_per_year
        print(f"  At {cost_bps} bp cost: {net_bps:.2f} bps/trade = {annual_return*100:.1f}% annually")

# Let's also check the distribution
print("\n\n=== RETURN DISTRIBUTION CHECK ===")
best_filter = (trades_df['volatility_rank'] > 0.9) & \
              (trades_df['direction'] == 'short') & \
              (trades_df['volume_ratio'] > 1.5)
best_trades = trades_df[best_filter]

if len(best_trades) > 0:
    print(f"Distribution of returns for best filter:")
    print(f"  Mean: {best_trades['net_return_bps'].mean():.2f} bps")
    print(f"  Median: {best_trades['net_return_bps'].median():.2f} bps")
    print(f"  Std Dev: {best_trades['net_return_bps'].std():.2f} bps")
    print(f"  Min: {best_trades['net_return_bps'].min():.2f} bps")
    print(f"  Max: {best_trades['net_return_bps'].max():.2f} bps")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90]
    print("\n  Percentiles:")
    for p in percentiles:
        print(f"    {p}th: {best_trades['net_return_bps'].percentile(p/100):.2f} bps")

# Double check by recalculating from scratch
print("\n\n=== MANUAL RECALCULATION ===")
# Count trades meeting each condition
vol90_count = (trades_df['volatility_rank'] > 0.9).sum()
shorts_count = (trades_df['direction'] == 'short').sum()
highvol_count = (trades_df['volume_ratio'] > 1.5).sum()

print(f"Trades with Vol>90%: {vol90_count} ({vol90_count/len(trades_df)*100:.1f}%)")
print(f"Short trades: {shorts_count} ({shorts_count/len(trades_df)*100:.1f}%)")
print(f"High volume trades: {highvol_count} ({highvol_count/len(trades_df)*100:.1f}%)")

# The intersection
all_three = best_filter.sum()
print(f"\nTrades meeting all three conditions: {all_three}")
print(f"That's {all_three/len(trades_df)*100:.2f}% of all trades")
print(f"From 600 strategies, that's {all_three/600:.1f} trades per strategy on average")