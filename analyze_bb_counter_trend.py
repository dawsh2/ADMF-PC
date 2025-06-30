#!/usr/bin/env python3
"""
Analyze Bollinger Bands trading against SMA slopes (counter-trend)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load signals (using the new implementation)
signal_file = Path("/Users/daws/ADMF-PC/workspaces/signal_generation_cc984d99/traces/SPY_1m/signals/bollinger_bands/SPY_compiled_strategy_0.parquet")
signals = pd.read_parquet(signal_file)

# Load and prepare price data
prices = pd.read_csv("./data/SPY_1m.csv")

# Calculate SMAs and their slopes
for period in [50, 100, 200]:
    prices[f'sma_{period}'] = prices['Close'].rolling(period).mean()
    prices[f'sma_{period}_slope'] = prices[f'sma_{period}'].diff(10) / 10  # 10-bar slope
    prices[f'sma_{period}_slope_pct'] = prices[f'sma_{period}_slope'] / prices[f'sma_{period}'] * 100

# Bollinger Bands
prices['bb_middle'] = prices['Close'].rolling(20).mean()
prices['bb_std'] = prices['Close'].rolling(20).std()
prices['bb_upper'] = prices['bb_middle'] + 2 * prices['bb_std']
prices['bb_lower'] = prices['bb_middle'] - 2 * prices['bb_std']
prices['bb_position'] = (prices['Close'] - prices['bb_lower']) / (prices['bb_upper'] - prices['bb_lower'])

# VWAP and volume
prices['vwap'] = (prices['Close'] * prices['Volume']).cumsum() / prices['Volume'].cumsum()
prices['volume_sma'] = prices['Volume'].rolling(20).mean()
prices['volume_ratio'] = prices['Volume'] / prices['volume_sma']

# Extract trades with context
trades = []
entry_idx = None
entry_signal = None

for _, row in signals.iterrows():
    signal = row['val']
    bar_idx = row['idx']
    
    if entry_idx is None and signal != 0:
        if bar_idx < len(prices):
            entry_idx = bar_idx
            entry_signal = signal
            entry_price = row['px']
            
    elif entry_idx is not None and (signal == 0 or signal != entry_signal):
        # Exit
        if entry_signal > 0:
            pnl_pct = (row['px'] - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - row['px']) / entry_price * 100
            
        if bar_idx < len(prices) and entry_idx < len(prices):
            entry_row = prices.iloc[entry_idx]
            
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': bar_idx,
                'duration': bar_idx - entry_idx,
                'pnl_pct': pnl_pct,
                'signal_type': 'long' if entry_signal > 0 else 'short',
                'entry_price': entry_price,
                'entry_bb_position': entry_row['bb_position'],
                
                # SMA slopes
                'sma_50_slope': entry_row['sma_50_slope_pct'] if pd.notna(entry_row['sma_50_slope_pct']) else None,
                'sma_100_slope': entry_row['sma_100_slope_pct'] if pd.notna(entry_row['sma_100_slope_pct']) else None,
                'sma_200_slope': entry_row['sma_200_slope_pct'] if pd.notna(entry_row['sma_200_slope_pct']) else None,
                
                # Other context
                'volume_ratio': entry_row['volume_ratio'],
                'above_vwap': entry_price > entry_row['vwap'] if pd.notna(entry_row['vwap']) else None,
            })
        
        if signal != 0:
            entry_idx = bar_idx
            entry_signal = signal
            entry_price = row['px']
        else:
            entry_idx = None

trades_df = pd.DataFrame(trades)
valid_trades = trades_df.dropna(subset=['sma_50_slope', 'sma_100_slope', 'sma_200_slope'])

print("="*60)
print("COUNTER-TREND ANALYSIS (Trading Against SMA Slopes)")
print("="*60)

print(f"\nTotal valid trades: {len(valid_trades)}")

# Define counter-trend trades
# Long when SMAs are falling (betting on bounce)
# Short when SMAs are rising (betting on pullback)

# 1. Pure counter-trend analysis
print("\n1. PURE COUNTER-TREND TRADES:")
print("-" * 60)

counter_trend_filters = [
    ("Longs against SMA50 downtrend", (valid_trades['signal_type'] == 'long') & (valid_trades['sma_50_slope'] < 0)),
    ("Shorts against SMA50 uptrend", (valid_trades['signal_type'] == 'short') & (valid_trades['sma_50_slope'] > 0)),
    ("Longs against SMA100 downtrend", (valid_trades['signal_type'] == 'long') & (valid_trades['sma_100_slope'] < 0)),
    ("Shorts against SMA100 uptrend", (valid_trades['signal_type'] == 'short') & (valid_trades['sma_100_slope'] > 0)),
    ("Longs against SMA200 downtrend", (valid_trades['signal_type'] == 'long') & (valid_trades['sma_200_slope'] < 0)),
    ("Shorts against SMA200 uptrend", (valid_trades['signal_type'] == 'short') & (valid_trades['sma_200_slope'] > 0)),
]

for name, mask in counter_trend_filters:
    filtered = valid_trades[mask]
    if len(filtered) > 10:
        print(f"\n{name}: {len(filtered)} trades")
        
        # By duration
        for dur_range in [(1, 1), (2, 5), (6, 10), (11, 20)]:
            dur_trades = filtered[filtered['duration'].between(dur_range[0], dur_range[1])]
            if len(dur_trades) > 5:
                net = dur_trades['pnl_pct'].sum() - len(dur_trades) * 0.01
                win_rate = (dur_trades['pnl_pct'] > 0).mean()
                avg_pnl = dur_trades['pnl_pct'].mean()
                print(f"  {dur_range[0]:2d}-{dur_range[1]:2d} bars: {len(dur_trades):4d} trades, "
                      f"{win_rate:.1%} win, {avg_pnl:6.3f}% avg, {net:7.2f}% net")

# 2. Strength of counter-trend
print("\n\n2. COUNTER-TREND BY SLOPE STRENGTH:")
print("-" * 60)

# Define strong slopes
strong_down = valid_trades['sma_200_slope'] < -0.01  # Strong downtrend
strong_up = valid_trades['sma_200_slope'] > 0.01     # Strong uptrend

counter_strength = [
    ("Longs in STRONG SMA200 downtrend", (valid_trades['signal_type'] == 'long') & strong_down),
    ("Shorts in STRONG SMA200 uptrend", (valid_trades['signal_type'] == 'short') & strong_up),
]

for name, mask in counter_strength:
    filtered = valid_trades[mask]
    if len(filtered) > 10:
        print(f"\n{name}: {len(filtered)} trades")
        
        # Overall performance
        total_net = filtered['pnl_pct'].sum() - len(filtered) * 0.01
        print(f"Total net return: {total_net:.2f}%")
        
        # By duration
        for d in range(1, 11):
            d_trades = filtered[filtered['duration'] == d]
            if len(d_trades) > 5:
                net = d_trades['pnl_pct'].sum() - len(d_trades) * 0.01
                win_rate = (d_trades['pnl_pct'] > 0).mean()
                print(f"  {d}-bar: {len(d_trades):3d} trades, {win_rate:.1%} win, {net:6.2f}% net")

# 3. Multiple timeframe agreement
print("\n\n3. MULTIPLE TIMEFRAME COUNTER-TREND:")
print("-" * 60)

# All SMAs pointing same direction
all_down = (valid_trades['sma_50_slope'] < 0) & (valid_trades['sma_100_slope'] < 0) & (valid_trades['sma_200_slope'] < 0)
all_up = (valid_trades['sma_50_slope'] > 0) & (valid_trades['sma_100_slope'] > 0) & (valid_trades['sma_200_slope'] > 0)

multi_tf = [
    ("Longs when ALL SMAs falling", (valid_trades['signal_type'] == 'long') & all_down),
    ("Shorts when ALL SMAs rising", (valid_trades['signal_type'] == 'short') & all_up),
]

for name, mask in multi_tf:
    filtered = valid_trades[mask]
    if len(filtered) > 10:
        print(f"\n{name}: {len(filtered)} trades")
        total_net = filtered['pnl_pct'].sum() - len(filtered) * 0.01
        win_rate = (filtered['pnl_pct'] > 0).mean()
        print(f"Overall: {win_rate:.1%} win rate, {total_net:.2f}% net return")
        
        # Show best durations
        for d in [1, 2, 3, 4, 5]:
            d_trades = filtered[filtered['duration'] == d]
            if len(d_trades) > 0:
                net = d_trades['pnl_pct'].sum() - len(d_trades) * 0.01
                print(f"  {d}-bar: {len(d_trades)} trades, {net:.2f}% net")

# 4. Counter-trend with additional filters
print("\n\n4. FILTERED COUNTER-TREND STRATEGIES:")
print("-" * 60)

# Test combinations
test_combos = [
    ("Counter-trend longs + High Volume", 
     (valid_trades['signal_type'] == 'long') & (valid_trades['sma_200_slope'] < 0) & (valid_trades['volume_ratio'] > 1.5)),
    
    ("Counter-trend shorts + Below VWAP", 
     (valid_trades['signal_type'] == 'short') & (valid_trades['sma_200_slope'] > 0) & (~valid_trades['above_vwap'])),
    
    ("1-bar counter-trend longs in downtrend",
     (valid_trades['signal_type'] == 'long') & (valid_trades['sma_200_slope'] < 0) & (valid_trades['duration'] == 1)),
    
    ("2-5 bar counter-trend with all SMAs aligned",
     (valid_trades['duration'].between(2, 5)) & 
     (((valid_trades['signal_type'] == 'long') & all_down) | ((valid_trades['signal_type'] == 'short') & all_up))),
]

best_results = []
for name, mask in test_combos:
    filtered = valid_trades[mask]
    if len(filtered) > 10:
        net = filtered['pnl_pct'].sum() - len(filtered) * 0.01
        win_rate = (filtered['pnl_pct'] > 0).mean()
        avg_duration = filtered['duration'].mean()
        
        best_results.append((net, name, len(filtered), win_rate))
        
        print(f"\n{name}:")
        print(f"  Trades: {len(filtered)}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Net Return: {net:.2f}%")
        print(f"  Avg Duration: {avg_duration:.1f} bars")

# Summary
if best_results:
    print("\n\nBEST COUNTER-TREND STRATEGIES:")
    print("-" * 60)
    best_results.sort(reverse=True)
    for i, (net, name, trades, win_rate) in enumerate(best_results[:3]):
        if net > 0:
            print(f"{i+1}. {name}")
            print(f"   {net:.2f}% net ({trades} trades, {win_rate:.1%} win)")

# Compare to trend-following
print("\n\n5. COMPARISON: TREND-FOLLOWING vs COUNTER-TREND:")
print("-" * 60)

trend_longs = (valid_trades['signal_type'] == 'long') & (valid_trades['sma_200_slope'] > 0)
counter_longs = (valid_trades['signal_type'] == 'long') & (valid_trades['sma_200_slope'] < 0)

print(f"\nLongs WITH trend: {len(valid_trades[trend_longs])} trades, "
      f"{(valid_trades[trend_longs]['pnl_pct'].sum() - len(valid_trades[trend_longs]) * 0.01):.2f}% net")
print(f"Longs AGAINST trend: {len(valid_trades[counter_longs])} trades, "
      f"{(valid_trades[counter_longs]['pnl_pct'].sum() - len(valid_trades[counter_longs]) * 0.01):.2f}% net")