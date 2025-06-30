#!/usr/bin/env python3
"""
Detailed Bollinger Bands analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load signals
signal_file = Path("/Users/daws/ADMF-PC/workspaces/signal_generation_cc984d99/traces/SPY_1m/signals/bollinger_bands/SPY_compiled_strategy_0.parquet")
signals = pd.read_parquet(signal_file)

# Load and prepare price data
prices = pd.read_csv("./data/SPY_1m.csv")

# Calculate Bollinger Bands
prices['bb_middle'] = prices['Close'].rolling(20).mean()
prices['bb_std'] = prices['Close'].rolling(20).std()
prices['bb_upper'] = prices['bb_middle'] + 2 * prices['bb_std']
prices['bb_lower'] = prices['bb_middle'] - 2 * prices['bb_std']
prices['bb_position'] = (prices['Close'] - prices['bb_lower']) / (prices['bb_upper'] - prices['bb_lower'])

# Calculate VWAP
prices['vwap'] = (prices['Close'] * prices['Volume']).cumsum() / prices['Volume'].cumsum()

# Extract trades
trades = []
entry_idx = None
entry_signal = None

for _, row in signals.iterrows():
    signal = row['val']
    bar_idx = row['idx']
    
    if entry_idx is None and signal != 0:
        entry_idx = bar_idx
        entry_signal = signal
        entry_price = row['px']
    elif entry_idx is not None and (signal == 0 or signal != entry_signal):
        # Exit
        if entry_signal > 0:
            pnl_pct = (row['px'] - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - row['px']) / entry_price * 100
            
        # Get entry context
        if bar_idx < len(prices) and entry_idx < len(prices):
            entry_row = prices.iloc[entry_idx]
            
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': bar_idx,
                'duration': bar_idx - entry_idx,
                'pnl_pct': pnl_pct,
                'signal_type': 'long' if entry_signal > 0 else 'short',
                'entry_price': entry_price,
                'entry_bb_position': entry_row['bb_position'] if pd.notna(entry_row['bb_position']) else None,
                'entry_above_vwap': entry_price > entry_row['vwap'] if pd.notna(entry_row['vwap']) else None,
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
print("1. DETAILED DURATION ANALYSIS - Where performance drops off")
print("="*60)

# Fine-grained duration analysis
for duration in range(1, 15):
    duration_trades = trades_df[trades_df['duration'] == duration]
    if len(duration_trades) > 0:
        gross = duration_trades['pnl_pct'].sum()
        net = gross - len(duration_trades) * 0.01
        avg_pnl = duration_trades['pnl_pct'].mean()
        win_rate = (duration_trades['pnl_pct'] > 0).mean()
        print(f"Duration {duration:2d}: {len(duration_trades):4d} trades, "
              f"{avg_pnl:6.3f}% avg, {win_rate:5.1%} win, "
              f"Gross: {gross:7.2f}%, Net: {net:7.2f}%")

print("\n" + "="*60)
print("2. VWAP ANALYSIS - Gross returns")
print("="*60)

# Remove trades with no VWAP data
valid_vwap = trades_df.dropna(subset=['entry_above_vwap'])

above_vwap = valid_vwap[valid_vwap['entry_above_vwap']]
below_vwap = valid_vwap[~valid_vwap['entry_above_vwap']]

print(f"Above VWAP: {len(above_vwap)} trades")
print(f"  Gross return: {above_vwap['pnl_pct'].sum():.2f}%")
print(f"  Net return: {above_vwap['pnl_pct'].sum() - len(above_vwap) * 0.01:.2f}%")
print(f"  Avg per trade: {above_vwap['pnl_pct'].mean():.4f}%")

print(f"\nBelow VWAP: {len(below_vwap)} trades")
print(f"  Gross return: {below_vwap['pnl_pct'].sum():.2f}%")
print(f"  Net return: {below_vwap['pnl_pct'].sum() - len(below_vwap) * 0.01:.2f}%")
print(f"  Avg per trade: {below_vwap['pnl_pct'].mean():.4f}%")

print("\n" + "="*60)
print("3. BB POSITION ANALYSIS - Why 90% in middle?")
print("="*60)

# Remove trades with no BB position data
valid_bb = trades_df.dropna(subset=['entry_bb_position'])

# More detailed BB position buckets
bb_buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
valid_bb['bb_bucket'] = pd.cut(valid_bb['entry_bb_position'], bins=bb_buckets)

print("Entry positions by BB bucket:")
bucket_counts = valid_bb['bb_bucket'].value_counts().sort_index()
for bucket, count in bucket_counts.items():
    pct = count / len(valid_bb) * 100
    bucket_trades = valid_bb[valid_bb['bb_bucket'] == bucket]
    avg_pnl = bucket_trades['pnl_pct'].mean()
    print(f"  {bucket}: {count:4d} trades ({pct:5.1f}%), avg PnL: {avg_pnl:6.3f}%")

# Check actual signals at entry
print("\n" + "="*60)
print("4. SIGNAL ANALYSIS - What triggers entries?")
print("="*60)

# Sample some entry points
sample_entries = trades_df.head(20)
print("\nFirst 20 trade entries:")
print("Entry Idx | Signal | BB Position | Close | BB Lower | BB Upper")
print("-" * 60)

for _, trade in sample_entries.iterrows():
    entry_idx = int(trade['entry_idx'])
    if entry_idx < len(signals) and entry_idx < len(prices):
        signal_row = signals[signals['idx'] == entry_idx]
        if len(signal_row) > 0:
            signal_val = signal_row.iloc[0]['val']
            price_row = prices.iloc[entry_idx]
            
            print(f"{entry_idx:9d} | {signal_val:6d} | {trade['entry_bb_position']:11.3f} | "
                  f"{price_row['Close']:6.2f} | {price_row['bb_lower']:8.2f} | {price_row['bb_upper']:8.2f}")