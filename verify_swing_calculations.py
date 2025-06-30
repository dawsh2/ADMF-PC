#!/usr/bin/env python3
"""Verify swing pivot bounce calculations step by step."""

import pandas as pd
import numpy as np

# Load the saved results
trades_df = pd.read_csv('swing_zones_all_trades.csv')

print("=== DATA VERIFICATION ===")
print(f"Total rows in CSV: {len(trades_df)}")
print(f"Columns: {list(trades_df.columns)}")
print(f"\nFirst few rows:")
print(trades_df.head())

# Check data types
print("\n=== DATA TYPES ===")
print(trades_df.dtypes)

# Basic statistics
print("\n=== BASIC STATISTICS ===")
print(f"Average return (all trades): {trades_df['net_return_bps'].mean():.2f} bps")
print(f"Median return: {trades_df['net_return_bps'].median():.2f} bps")
print(f"Std deviation: {trades_df['net_return_bps'].std():.2f} bps")

# Check unique strategies
print(f"\nUnique strategies: {trades_df['strategy_idx'].nunique()}")
print(f"Strategy indices range: {trades_df['strategy_idx'].min()} to {trades_df['strategy_idx'].max()}")

# The problematic filter
print("\n=== FILTER: Vol>90 + Shorts + HighVolume ===")

# Apply filter step by step
vol_mask = trades_df['volatility_rank'] > 0.9
shorts_mask = trades_df['direction'] == 'short'
highvol_mask = trades_df['volume_ratio'] > 1.5

print(f"\nStep 1 - Volatility > 90%:")
print(f"  Trades matching: {vol_mask.sum()} ({vol_mask.sum()/len(trades_df)*100:.1f}%)")
print(f"  Average return: {trades_df[vol_mask]['net_return_bps'].mean():.2f} bps")

print(f"\nStep 2 - Short trades:")
print(f"  Trades matching: {shorts_mask.sum()} ({shorts_mask.sum()/len(trades_df)*100:.1f}%)")
print(f"  Average return: {trades_df[shorts_mask]['net_return_bps'].mean():.2f} bps")

print(f"\nStep 3 - High volume (>1.5x):")
print(f"  Trades matching: {highvol_mask.sum()} ({highvol_mask.sum()/len(trades_df)*100:.1f}%)")
print(f"  Average return: {trades_df[highvol_mask]['net_return_bps'].mean():.2f} bps")

# Combined filter
combined_mask = vol_mask & shorts_mask & highvol_mask
filtered_trades = trades_df[combined_mask]

print(f"\n=== COMBINED FILTER RESULT ===")
print(f"Total trades matching all 3 conditions: {len(filtered_trades)}")
print(f"That's {len(filtered_trades)/len(trades_df)*100:.2f}% of all trades")

if len(filtered_trades) > 0:
    print(f"\nPerformance metrics:")
    print(f"  Average return: {filtered_trades['net_return_bps'].mean():.2f} bps")
    print(f"  Median return: {filtered_trades['net_return_bps'].median():.2f} bps")
    print(f"  Win rate: {(filtered_trades['net_return'] > 0).mean()*100:.1f}%")
    print(f"  Best trade: {filtered_trades['net_return_bps'].max():.2f} bps")
    print(f"  Worst trade: {filtered_trades['net_return_bps'].min():.2f} bps")
    
    # Show sample of trades
    print(f"\nSample of filtered trades:")
    print(filtered_trades[['strategy_idx', 'net_return_bps', 'bars_held', 
                          'volatility_rank', 'volume_ratio']].head(10))
    
    # Strategy breakdown
    print(f"\n=== STRATEGY BREAKDOWN ===")
    strategy_counts = filtered_trades['strategy_idx'].value_counts()
    print(f"Number of strategies with trades: {len(strategy_counts)}")
    print(f"Average trades per strategy: {len(filtered_trades)/600:.2f}")
    print(f"\nTop 10 strategies by trade count:")
    print(strategy_counts.head(10))
    
    # Time calculations
    print(f"\n=== TIME CALCULATIONS ===")
    total_strategies = 600  # From config
    trading_days = 306  # Estimated for the period
    
    print(f"Total trades across ALL {total_strategies} strategies: {len(filtered_trades)}")
    print(f"Trades per day (all strategies): {len(filtered_trades)/trading_days:.1f}")
    print(f"Trades per strategy per day: {len(filtered_trades)/total_strategies/trading_days:.4f}")
    print(f"Days between trades (per strategy): {trading_days/(len(filtered_trades)/total_strategies):.1f}")
    
    # Annual return calculation
    avg_bps = filtered_trades['net_return_bps'].mean()
    trades_per_year = len(filtered_trades) / trading_days * 252  # Scale to full year
    trades_per_strategy_per_year = trades_per_year / total_strategies
    
    print(f"\n=== ANNUAL RETURN CALCULATION ===")
    print(f"Average return per trade: {avg_bps:.2f} bps")
    print(f"Estimated trades per year (all strategies): {trades_per_year:.0f}")
    print(f"Trades per strategy per year: {trades_per_strategy_per_year:.1f}")
    
    # Different cost scenarios
    for cost_bps in [0, 0.5, 1.0, 2.0]:
        net_bps = avg_bps - cost_bps
        annual_return = (net_bps / 10000) * trades_per_strategy_per_year
        print(f"\nWith {cost_bps} bp transaction cost:")
        print(f"  Net per trade: {net_bps:.2f} bps")
        print(f"  Annual return per strategy: {annual_return*100:.2f}%")

# Check for data issues
print("\n=== DATA QUALITY CHECK ===")
print(f"Missing volatility_rank: {trades_df['volatility_rank'].isna().sum()}")
print(f"Missing volume_ratio: {trades_df['volume_ratio'].isna().sum()}")
print(f"Missing direction: {trades_df['direction'].isna().sum()}")

# Volatility rank distribution
print(f"\nVolatility rank distribution:")
print(f"  Min: {trades_df['volatility_rank'].min():.3f}")
print(f"  25%: {trades_df['volatility_rank'].quantile(0.25):.3f}")
print(f"  50%: {trades_df['volatility_rank'].quantile(0.50):.3f}")
print(f"  75%: {trades_df['volatility_rank'].quantile(0.75):.3f}")
print(f"  Max: {trades_df['volatility_rank'].max():.3f}")

# Volume ratio distribution
print(f"\nVolume ratio distribution:")
print(f"  Min: {trades_df['volume_ratio'].min():.3f}")
print(f"  25%: {trades_df['volume_ratio'].quantile(0.25):.3f}")
print(f"  50%: {trades_df['volume_ratio'].quantile(0.50):.3f}")
print(f"  75%: {trades_df['volume_ratio'].quantile(0.75):.3f}")
print(f"  Max: {trades_df['volume_ratio'].max():.3f}")