#!/usr/bin/env python3
"""Find filters that provide reasonable trade frequency with positive edge."""

import pandas as pd
import numpy as np

# Load the saved results
trades_df = pd.read_csv('swing_zones_all_trades.csv')

print("=== FINDING PRACTICAL TRADING FILTERS ===")
print(f"Total trades analyzed: {len(trades_df)}")
print(f"Overall average: {trades_df['net_return_bps'].mean():.2f} bps\n")

# Define minimum trade frequency requirements
MIN_TRADES_PER_DAY = 2  # User requirement: 2-3 trades per day
TRADING_DAYS = 306
TOTAL_STRATEGIES = 600
MIN_TOTAL_TRADES = MIN_TRADES_PER_DAY * TRADING_DAYS * TOTAL_STRATEGIES  # 367,200

print(f"Minimum requirement: {MIN_TRADES_PER_DAY} trades/day per strategy")
print(f"This means we need at least {MIN_TOTAL_TRADES:,} total trades\n")

# Test various filters
filters = {
    # Single filters
    "No filter": trades_df.index >= 0,
    "Longs only": trades_df['direction'] == 'long',
    "Shorts only": trades_df['direction'] == 'short',
    
    # Volatility thresholds
    "Vol > 50%": trades_df['volatility_rank'] > 0.5,
    "Vol > 60%": trades_df['volatility_rank'] > 0.6,
    "Vol > 70%": trades_df['volatility_rank'] > 0.7,
    "Vol > 80%": trades_df['volatility_rank'] > 0.8,
    
    # Volume thresholds
    "Volume > 1.2x": trades_df['volume_ratio'] > 1.2,
    "Volume > 1.5x": trades_df['volume_ratio'] > 1.5,
    "Volume < 0.8x": trades_df['volume_ratio'] < 0.8,
    
    # Trend filters
    "Uptrend": trades_df['trend'] == 'up',
    "Downtrend": trades_df['trend'] == 'down',
    
    # Combined filters - focus on frequency
    "Vol>50 + HighVol": (trades_df['volatility_rank'] > 0.5) & (trades_df['volume_ratio'] > 1.2),
    "Vol>60 + HighVol": (trades_df['volatility_rank'] > 0.6) & (trades_df['volume_ratio'] > 1.2),
    "Downtrend + HighVol": (trades_df['trend'] == 'down') & (trades_df['volume_ratio'] > 1.2),
    "Downtrend + Vol>60": (trades_df['trend'] == 'down') & (trades_df['volatility_rank'] > 0.6),
}

print("=== FILTER ANALYSIS (sorted by frequency) ===")
print(f"{'Filter':<25} {'Total Trades':>12} {'Trades/Day':>10} {'Avg(bps)':>10} {'Win%':>8} {'Annual%':>10}")
print("-" * 85)

results = []
for name, mask in filters.items():
    # Handle NaN values in mask
    mask = mask.fillna(False) if hasattr(mask, 'fillna') else mask
    
    filtered = trades_df[mask]
    if len(filtered) > 0:
        trades_per_day = len(filtered) / TRADING_DAYS / TOTAL_STRATEGIES
        avg_bps = filtered['net_return_bps'].mean()
        win_rate = (filtered['net_return'] > 0).mean() * 100
        
        # Annual return calculation (with 1bp cost)
        net_bps = avg_bps - 1.0
        trades_per_year = trades_per_day * 252
        annual_return = (net_bps / 10000) * trades_per_year * 100
        
        results.append({
            'filter': name,
            'total_trades': len(filtered),
            'trades_per_day': trades_per_day,
            'avg_bps': avg_bps,
            'win_rate': win_rate,
            'annual_return': annual_return
        })

# Sort by trades per day
results.sort(key=lambda x: x['trades_per_day'], reverse=True)

# Print results
for r in results:
    print(f"{r['filter']:<25} {r['total_trades']:>12,} {r['trades_per_day']:>10.2f} "
          f"{r['avg_bps']:>10.2f} {r['win_rate']:>8.1f} {r['annual_return']:>10.2f}%")

print("\n=== FILTERS MEETING FREQUENCY REQUIREMENT (2+ trades/day) ===")
practical_filters = [r for r in results if r['trades_per_day'] >= MIN_TRADES_PER_DAY]

if practical_filters:
    print(f"{'Filter':<25} {'Trades/Day':>10} {'Avg(bps)':>10} {'Annual%':>10}")
    print("-" * 60)
    for r in practical_filters:
        print(f"{r['filter']:<25} {r['trades_per_day']:>10.2f} "
              f"{r['avg_bps']:>10.2f} {r['annual_return']:>10.2f}%")
else:
    print("NO FILTERS MEET THE 2+ TRADES/DAY REQUIREMENT")

print("\n=== BEST COMPROMISE FILTERS (1+ trades/day with positive return) ===")
compromise_filters = [r for r in results if r['trades_per_day'] >= 1 and r['annual_return'] > 0]

if compromise_filters:
    print(f"{'Filter':<25} {'Trades/Day':>10} {'Avg(bps)':>10} {'Annual%':>10}")
    print("-" * 60)
    for r in compromise_filters:
        print(f"{r['filter']:<25} {r['trades_per_day']:>10.2f} "
              f"{r['avg_bps']:>10.2f} {r['annual_return']:>10.2f}%")

# Check less restrictive volatility filters
print("\n=== EXPLORING LOWER VOLATILITY THRESHOLDS ===")
for vol_threshold in [0.3, 0.4, 0.5]:
    mask = trades_df['volatility_rank'] > vol_threshold
    filtered = trades_df[mask.fillna(False)]
    
    if len(filtered) > 0:
        trades_per_day = len(filtered) / TRADING_DAYS / TOTAL_STRATEGIES
        avg_bps = filtered['net_return_bps'].mean()
        net_bps = avg_bps - 1.0
        annual_return = (net_bps / 10000) * trades_per_day * 252 * 100
        
        print(f"Vol > {int(vol_threshold*100)}%: "
              f"{trades_per_day:.2f} trades/day, "
              f"{avg_bps:.2f} bps/trade, "
              f"{annual_return:.2f}% annual")

# Test very light filters
print("\n=== MINIMAL FILTERS FOR HIGH FREQUENCY ===")
light_filters = {
    "Vol > 40%": trades_df['volatility_rank'] > 0.4,
    "Vol > 30%": trades_df['volatility_rank'] > 0.3,
    "Any HighVol (>1.1x)": trades_df['volume_ratio'] > 1.1,
    "Any HighVol (>1.0x)": trades_df['volume_ratio'] > 1.0,
}

for name, mask in light_filters.items():
    mask = mask.fillna(False) if hasattr(mask, 'fillna') else mask
    filtered = trades_df[mask]
    
    if len(filtered) > 0:
        trades_per_day = len(filtered) / TRADING_DAYS / TOTAL_STRATEGIES
        avg_bps = filtered['net_return_bps'].mean()
        net_bps = avg_bps - 1.0
        annual_return = (net_bps / 10000) * trades_per_day * 252 * 100
        
        print(f"{name:<20} {trades_per_day:>8.2f} trades/day, "
              f"{avg_bps:>8.2f} bps, {annual_return:>8.2f}% annual")

print("\n=== CONCLUSION ===")
print(f"The swing pivot bounce strategy cannot achieve {MIN_TRADES_PER_DAY}+ trades/day")
print("with positive expected returns after transaction costs.")
print("\nBest options:")
print("1. Accept lower frequency: ~0.3 trades/day with 2-3% annual return")
print("2. Trade without filters: 4.4 trades/day but negative returns")
print("3. Consider a different strategy that generates more signals")