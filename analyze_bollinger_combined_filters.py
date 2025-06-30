#!/usr/bin/env python3
"""
Analyze Bollinger Bands with combined filters
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load signals
signal_file = Path("/Users/daws/ADMF-PC/workspaces/signal_generation_cc984d99/traces/SPY_1m/signals/bollinger_bands/SPY_compiled_strategy_0.parquet")
signals = pd.read_parquet(signal_file)

# Load and prepare price data
prices = pd.read_csv("./data/SPY_1m.csv")

# Calculate indicators
prices['sma_200'] = prices['Close'].rolling(200).mean()
prices['sma_200_slope'] = prices['sma_200'].diff(5) / 5  # 5-bar slope
prices['sma_50'] = prices['Close'].rolling(50).mean()
prices['sma_50_slope'] = prices['sma_50'].diff(5) / 5

# VWAP
prices['vwap'] = (prices['Close'] * prices['Volume']).cumsum() / prices['Volume'].cumsum()

# Bollinger Bands
prices['bb_middle'] = prices['Close'].rolling(20).mean()
prices['bb_std'] = prices['Close'].rolling(20).std()
prices['bb_upper'] = prices['bb_middle'] + 2 * prices['bb_std']
prices['bb_lower'] = prices['bb_middle'] - 2 * prices['bb_std']
prices['bb_position'] = (prices['Close'] - prices['bb_lower']) / (prices['bb_upper'] - prices['bb_lower'])

# Extract trades with context
trades = []
entry_idx = None
entry_signal = None

for _, row in signals.iterrows():
    signal = row['val']
    bar_idx = row['idx']
    
    if entry_idx is None and signal != 0:
        if bar_idx < len(prices):
            entry_row = prices.iloc[bar_idx]
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
                'entry_above_vwap': entry_price > entry_row['vwap'],
                'entry_above_sma200': entry_price > entry_row['sma_200'] if pd.notna(entry_row['sma_200']) else None,
                'entry_sma200_slope': entry_row['sma_200_slope'] if pd.notna(entry_row['sma_200_slope']) else None,
                'entry_sma50_slope': entry_row['sma_50_slope'] if pd.notna(entry_row['sma_50_slope']) else None,
            })
        
        # Check for re-entry
        if signal != 0:
            entry_idx = bar_idx
            entry_signal = signal
            entry_price = row['px']
        else:
            entry_idx = None

trades_df = pd.DataFrame(trades)

print("="*60)
print("COMBINED FILTER ANALYSIS")
print("="*60)

# Remove trades without necessary data
valid_trades = trades_df.dropna(subset=['entry_above_vwap', 'entry_sma200_slope'])

print(f"Total valid trades for analysis: {len(valid_trades)}")

# Define filters
below_vwap = ~valid_trades['entry_above_vwap']
sma200_up = valid_trades['entry_sma200_slope'] > 0
sma200_down = valid_trades['entry_sma200_slope'] < 0
sma50_up = valid_trades['entry_sma50_slope'] > 0
sma50_down = valid_trades['entry_sma50_slope'] < 0

# Analyze different combinations
print("\n1. VWAP + SMA 200 SLOPE COMBINATIONS:")
print("-" * 60)

combinations = [
    ("Below VWAP + SMA200 Rising", below_vwap & sma200_up),
    ("Below VWAP + SMA200 Falling", below_vwap & sma200_down),
    ("Above VWAP + SMA200 Rising", ~below_vwap & sma200_up),
    ("Above VWAP + SMA200 Falling", ~below_vwap & sma200_down),
]

for name, mask in combinations:
    filtered = valid_trades[mask]
    if len(filtered) > 0:
        gross = filtered['pnl_pct'].sum()
        net = gross - len(filtered) * 0.01
        avg_pnl = filtered['pnl_pct'].mean()
        win_rate = (filtered['pnl_pct'] > 0).mean()
        print(f"\n{name}:")
        print(f"  Trades: {len(filtered)}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Avg PnL: {avg_pnl:.4f}%")
        print(f"  Gross Return: {gross:.2f}%")
        print(f"  Net Return: {net:.2f}%")
        print(f"  Net per trade: {net/len(filtered):.4f}%")

# Look at duration for best combination
print("\n2. DURATION ANALYSIS FOR BELOW VWAP + SMA200 RISING:")
print("-" * 60)

best_combo = valid_trades[below_vwap & sma200_up]
if len(best_combo) > 10:
    for duration in range(1, 15):
        dur_trades = best_combo[best_combo['duration'] == duration]
        if len(dur_trades) > 0:
            gross = dur_trades['pnl_pct'].sum()
            net = gross - len(dur_trades) * 0.01
            win_rate = (dur_trades['pnl_pct'] > 0).mean()
            print(f"Duration {duration:2d}: {len(dur_trades):3d} trades, "
                  f"{win_rate:5.1%} win, Net: {net:6.2f}%")

# Try with SMA 50 instead
print("\n3. VWAP + SMA 50 SLOPE COMBINATIONS:")
print("-" * 60)

combinations_50 = [
    ("Below VWAP + SMA50 Rising", below_vwap & sma50_up),
    ("Below VWAP + SMA50 Falling", below_vwap & sma50_down),
]

for name, mask in combinations_50:
    filtered = valid_trades[mask]
    if len(filtered) > 0:
        gross = filtered['pnl_pct'].sum()
        net = gross - len(filtered) * 0.01
        avg_pnl = filtered['pnl_pct'].mean()
        win_rate = (filtered['pnl_pct'] > 0).mean()
        print(f"\n{name}:")
        print(f"  Trades: {len(filtered)}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Gross Return: {gross:.2f}%")
        print(f"  Net Return: {net:.2f}%")

# Long only in uptrend, short only in downtrend
print("\n4. ALIGNED DIRECTIONAL TRADING:")
print("-" * 60)

long_trades = valid_trades[valid_trades['signal_type'] == 'long']
short_trades = valid_trades[valid_trades['signal_type'] == 'short']

# Longs in uptrend
long_uptrend = long_trades[long_trades['entry_sma200_slope'] > 0]
long_downtrend = long_trades[long_trades['entry_sma200_slope'] < 0]

print(f"\nLongs in SMA200 Uptrend: {len(long_uptrend)} trades")
if len(long_uptrend) > 0:
    print(f"  Net Return: {long_uptrend['pnl_pct'].sum() - len(long_uptrend) * 0.01:.2f}%")
    print(f"  Win Rate: {(long_uptrend['pnl_pct'] > 0).mean():.1%}")

print(f"\nLongs in SMA200 Downtrend: {len(long_downtrend)} trades")
if len(long_downtrend) > 0:
    print(f"  Net Return: {long_downtrend['pnl_pct'].sum() - len(long_downtrend) * 0.01:.2f}%")
    print(f"  Win Rate: {(long_downtrend['pnl_pct'] > 0).mean():.1%}")

# Shorts in downtrend
short_uptrend = short_trades[short_trades['entry_sma200_slope'] > 0]
short_downtrend = short_trades[short_trades['entry_sma200_slope'] < 0]

print(f"\nShorts in SMA200 Uptrend: {len(short_uptrend)} trades")
if len(short_uptrend) > 0:
    print(f"  Net Return: {short_uptrend['pnl_pct'].sum() - len(short_uptrend) * 0.01:.2f}%")
    print(f"  Win Rate: {(short_uptrend['pnl_pct'] > 0).mean():.1%}")

print(f"\nShorts in SMA200 Downtrend: {len(short_downtrend)} trades")
if len(short_downtrend) > 0:
    print(f"  Net Return: {short_downtrend['pnl_pct'].sum() - len(short_downtrend) * 0.01:.2f}%")
    print(f"  Win Rate: {(short_downtrend['pnl_pct'] > 0).mean():.1%}")