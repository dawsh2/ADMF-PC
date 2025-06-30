#!/usr/bin/env python3
"""Explore new patterns and correlations in the swing pivot bounce data."""

import pandas as pd
import numpy as np
from itertools import combinations

# Load the data
trades_df = pd.read_csv('swing_zones_all_trades.csv')

print("=== EXPLORING NEW PATTERNS AND CORRELATIONS ===")
print(f"Total trades: {len(trades_df)}")
print(f"Trading days: 306")
print(f"Total strategies: 600\n")

# First, let's understand what parameters lead to more frequent trading
print("=== PARAMETER ANALYSIS FOR HIGH FREQUENCY ===")

# Group by parameters to see trade frequency
param_freq = trades_df.groupby(['sr_period', 'entry_zone', 'exit_zone', 'min_range']).agg({
    'net_return_bps': ['count', 'mean'],
    'direction': lambda x: (x == 'long').sum() / len(x)  # Long percentage
}).round(2)

param_freq.columns = ['trade_count', 'avg_bps', 'long_pct']
param_freq = param_freq.sort_values('trade_count', ascending=False)

print("Top 10 parameter combinations by trade frequency:")
print(param_freq.head(10))

# Analyze time-based patterns
print("\n=== TIME-BASED PATTERNS ===")
time_patterns = trades_df.groupby(['hour', 'session']).agg({
    'net_return_bps': ['count', 'mean'],
    'net_return': lambda x: (x > 0).mean()
}).round(2)

time_patterns.columns = ['trades', 'avg_bps', 'win_rate']
time_patterns = time_patterns.sort_values('avg_bps', ascending=False)

print("\nBest performing time periods:")
print(time_patterns[time_patterns['trades'] > 1000].head(10))

# Look for interaction effects
print("\n=== INTERACTION EFFECTS ===")

# RSI extremes with other conditions
trades_df['rsi_zone'] = pd.cut(trades_df['rsi'], 
                                bins=[0, 30, 40, 60, 70, 100],
                                labels=['oversold', 'low', 'neutral', 'high', 'overbought'])

# Price position relative to moving averages
trades_df['price_position'] = pd.cut(trades_df['price_to_sma20'], 
                                     bins=[-np.inf, -0.02, -0.01, 0.01, 0.02, np.inf],
                                     labels=['far_below', 'below', 'near', 'above', 'far_above'])

# Volatility regime
trades_df['vol_regime'] = pd.cut(trades_df['volatility_rank'], 
                                  bins=[0, 0.3, 0.6, 0.8, 1.0],
                                  labels=['low', 'medium', 'high', 'extreme'])

# Volume regime
trades_df['volume_regime'] = pd.cut(trades_df['volume_ratio'], 
                                     bins=[0, 0.8, 1.2, 2.0, 20],
                                     labels=['low', 'normal', 'high', 'very_high'])

# Test combinations
print("\n--- RSI Zone Performance ---")
rsi_perf = trades_df.groupby(['rsi_zone', 'direction']).agg({
    'net_return_bps': ['count', 'mean']
}).round(2)
rsi_perf.columns = ['trades', 'avg_bps']
print(rsi_perf[rsi_perf['trades'] > 1000])

print("\n--- Price Position Performance ---")
price_perf = trades_df.groupby(['price_position', 'direction']).agg({
    'net_return_bps': ['count', 'mean']
}).round(2)
price_perf.columns = ['trades', 'avg_bps']
print(price_perf[price_perf['trades'] > 1000])

# Complex filters
print("\n=== TESTING COMPLEX FILTER COMBINATIONS ===")

complex_filters = {
    # Mean reversion patterns
    "Oversold longs": (trades_df['direction'] == 'long') & (trades_df['rsi'] < 35),
    "Overbought shorts": (trades_df['direction'] == 'short') & (trades_df['rsi'] > 65),
    
    # Momentum patterns  
    "Strong trend longs": (trades_df['direction'] == 'long') & 
                         (trades_df['trend'] == 'up') & 
                         (trades_df['trend_strength'] == 'strong'),
    
    # Volatility patterns
    "Low vol mean rev": (trades_df['volatility_rank'] < 0.4) & 
                        ((trades_df['rsi'] < 35) | (trades_df['rsi'] > 65)),
    
    # Volume surge patterns
    "Volume breakout": (trades_df['volume_ratio'] > 2.0) & 
                       (trades_df['volatility_rank'] > 0.5),
    
    # Time + condition patterns
    "Morning volatility": (trades_df['hour'].isin([9, 10])) & 
                         (trades_df['volatility_rank'] > 0.6),
    
    "Afternoon mean rev": (trades_df['hour'] >= 14) & 
                         ((trades_df['rsi'] < 35) | (trades_df['rsi'] > 65)),
    
    # Inverse patterns (fade the extremes)
    "Fade vol spikes": (trades_df['volatility_rank'] > 0.9) & 
                       (trades_df['direction'] == 'short'),
    
    # Multi-condition patterns
    "Perfect storm long": (trades_df['direction'] == 'long') & 
                         (trades_df['rsi'] < 40) & 
                         (trades_df['trend'] == 'down') & 
                         (trades_df['volume_ratio'] > 1.2),
    
    "Perfect storm short": (trades_df['direction'] == 'short') & 
                          (trades_df['rsi'] > 60) & 
                          (trades_df['trend'] == 'up') & 
                          (trades_df['volume_ratio'] > 1.2),
}

print(f"\n{'Filter':<25} {'Trades':<10} {'Freq/Day':<10} {'Avg(bps)':<10} {'Win%':<8} {'Annual%':<10}")
print("-" * 80)

filter_results = []
for name, mask in complex_filters.items():
    filtered = trades_df[mask]
    if len(filtered) > 100:  # Minimum trades for significance
        trades_per_day = len(filtered) / 306 / 600
        avg_bps = filtered['net_return_bps'].mean()
        win_rate = (filtered['net_return'] > 0).mean() * 100
        annual_return = (avg_bps - 1) / 10000 * trades_per_day * 252 * 100
        
        filter_results.append({
            'name': name,
            'trades': len(filtered),
            'freq': trades_per_day,
            'avg_bps': avg_bps,
            'win_rate': win_rate,
            'annual': annual_return
        })
        
        print(f"{name:<25} {len(filtered):<10} {trades_per_day:<10.3f} "
              f"{avg_bps:<10.2f} {win_rate:<8.1f} {annual_return:<10.2f}%")

# Look for parameter sweet spots
print("\n=== PARAMETER SWEET SPOTS ===")

# Find strategies with both high frequency AND positive returns
strategy_summary = trades_df.groupby('strategy_idx').agg({
    'net_return_bps': ['count', 'mean'],
    'bars_held': 'mean'
}).round(2)

strategy_summary.columns = ['trades', 'avg_bps', 'avg_bars']
strategy_summary = strategy_summary[strategy_summary['trades'] > 1000]  # High frequency
strategy_summary = strategy_summary[strategy_summary['avg_bps'] > 0]    # Positive returns
strategy_summary = strategy_summary.sort_values('avg_bps', ascending=False)

if len(strategy_summary) > 0:
    print(f"\nFound {len(strategy_summary)} strategies with 1000+ trades AND positive returns:")
    
    # Get parameters for top strategies
    top_strategies = strategy_summary.head(5).index
    for idx in top_strategies:
        strategy_trades = trades_df[trades_df['strategy_idx'] == idx]
        params = strategy_trades.iloc[0][['sr_period', 'entry_zone', 'exit_zone', 'min_range']]
        
        print(f"\nStrategy {idx}:")
        print(f"  Parameters: SR={params['sr_period']}, Entry={params['entry_zone']}, "
              f"Exit={params['exit_zone']}, MinRange={params['min_range']}")
        print(f"  Performance: {strategy_summary.loc[idx, 'trades']} trades, "
              f"{strategy_summary.loc[idx, 'avg_bps']:.2f} bps/trade, "
              f"{strategy_summary.loc[idx, 'avg_bars']:.1f} bars held")

# Analyze holding period patterns
print("\n=== HOLDING PERIOD ANALYSIS ===")
holding_patterns = trades_df.groupby(pd.cut(trades_df['bars_held'], 
                                            bins=[0, 1, 2, 5, 10, 100])).agg({
    'net_return_bps': ['count', 'mean']
}).round(2)

holding_patterns.columns = ['trades', 'avg_bps']
print(holding_patterns)

# Look for non-linear relationships
print("\n=== NON-LINEAR PATTERNS ===")

# Volatility squared effect
trades_df['vol_squared'] = trades_df['volatility_rank'] ** 2

# Test quadratic relationship
vol_bins = pd.qcut(trades_df['volatility_rank'].dropna(), q=10)
vol_perf = trades_df.groupby(vol_bins).agg({
    'net_return_bps': ['count', 'mean']
}).round(2)

print("\nVolatility deciles performance:")
print(vol_perf)

# Final recommendations
print("\n=== RECOMMENDATIONS FOR HIGH-FREQUENCY PROFITABLE TRADING ===")
print("1. Focus on shorter holding periods (1-2 bars)")
print("2. Use tighter entry/exit zones for more signals")
print("3. Trade mean reversion in low volatility regimes")
print("4. Exploit time-of-day patterns (morning/afternoon)")
print("5. Consider inverse strategies (fade extremes)")
print("6. Combine RSI extremes with volume/volatility filters")