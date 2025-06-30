#!/usr/bin/env python3
"""Analyze entry timing and prices between universal analysis and execution engine."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
positions_path = Path("/Users/daws/ADMF-PC/config/bollinger/results/20250628_194812/traces/portfolio/positions_open/positions_open.parquet")
signals_path = Path("/Users/daws/ADMF-PC/config/bollinger/results/20250628_194812/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")
market_data_path = Path("/Users/daws/ADMF-PC/data/SPY_5m.csv")

print("=== ENTRY TIMING ANALYSIS ===\n")

# Load position data
print("Loading position data...")
positions = pd.read_parquet(positions_path)
print(f"Total positions opened: {len(positions)}")
print(f"Columns: {list(positions.columns)}")

# Show first few positions
print("\nFirst 5 positions:")
print(positions.head())

# Load signal data
print("\n\nLoading signal data...")
signals = pd.read_parquet(signals_path)
signals['ts'] = pd.to_datetime(signals['ts'])
print(f"Total signals: {len(signals)}")
print(f"Signal columns: {list(signals.columns)}")

# Load market data
print("\n\nLoading market data...")
market_data = pd.read_csv(market_data_path)
market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
market_data = market_data.sort_values('timestamp')
print(f"Market data range: {market_data['timestamp'].min()} to {market_data['timestamp'].max()}")

# Analyze entry timing
print("\n\n=== ENTRY TIMING PATTERNS ===")

# Convert position timestamps
positions['ts'] = pd.to_datetime(positions['ts'])

# Remove timezone info for matching
if positions['ts'].dt.tz is not None:
    positions['ts'] = positions['ts'].dt.tz_localize(None)
if market_data['timestamp'].dt.tz is not None:
    market_data['timestamp'] = market_data['timestamp'].dt.tz_localize(None)

# Merge positions with market data to see bar timing
positions_with_bars = positions.merge(
    market_data,
    left_on='ts',
    right_on='timestamp',
    how='left',
    suffixes=('_pos', '_bar')
)

print("\nEntry price vs bar prices:")
for idx, row in positions_with_bars.head(10).iterrows():
    entry_price = row['entry_price']
    bar_open = row['open']
    bar_close = row['close']
    bar_high = row['high']
    bar_low = row['low']
    
    print(f"\nPosition {idx+1}:")
    print(f"  Time: {row['ts']}")
    print(f"  Entry price: ${entry_price:.2f}")
    print(f"  Bar OHLC: O=${bar_open:.2f}, H=${bar_high:.2f}, L=${bar_low:.2f}, C=${bar_close:.2f}")
    
    # Check which price matches
    if abs(entry_price - bar_open) < 0.01:
        print(f"  -> Entry at BAR OPEN")
    elif abs(entry_price - bar_close) < 0.01:
        print(f"  -> Entry at BAR CLOSE")
    elif abs(entry_price - bar_high) < 0.01:
        print(f"  -> Entry at BAR HIGH")
    elif abs(entry_price - bar_low) < 0.01:
        print(f"  -> Entry at BAR LOW")
    else:
        print(f"  -> Entry at CUSTOM PRICE (slippage?)")

# Check signal timing vs position timing
print("\n\n=== SIGNAL TO ENTRY DELAY ===")

# For each position, find the corresponding signal
for idx, pos in positions.head(10).iterrows():
    pos_time = pos['ts']
    
    # Find the most recent signal before this position
    recent_signals = signals[signals['ts'] <= pos_time].tail(5)
    
    if len(recent_signals) > 0:
        last_signal = recent_signals.iloc[-1]
        signal_time = last_signal['ts']
        delay = (pos_time - signal_time).total_seconds()
        
        print(f"\nPosition {idx+1}:")
        print(f"  Signal time: {signal_time}")
        print(f"  Entry time:  {pos_time}")
        print(f"  Delay: {delay} seconds ({delay/60:.1f} minutes)")
        print(f"  Signal value: {last_signal['val']}")
        print(f"  Signal price: ${last_signal['px']:.2f}")
        print(f"  Entry price: ${pos['entry_price']:.2f}")
        print(f"  Price diff: ${pos['entry_price'] - last_signal['px']:.2f}")

# Analyze entry price patterns
print("\n\n=== ENTRY PRICE ANALYSIS ===")

# Compare entry prices to signal prices
positions_sorted = positions.sort_values('ts')
entry_prices = []
signal_prices = []

for idx, pos in positions_sorted.iterrows():
    pos_time = pos['ts']
    
    # Find corresponding signal
    signal_mask = (signals['ts'] <= pos_time) & (signals['val'] != 0)
    if signal_mask.any():
        last_signal = signals[signal_mask].iloc[-1]
        entry_prices.append(pos['entry_price'])
        signal_prices.append(last_signal['px'])

if entry_prices:
    entry_prices = np.array(entry_prices)
    signal_prices = np.array(signal_prices)
    
    price_diffs = entry_prices - signal_prices
    
    print(f"Entry price vs signal price statistics:")
    print(f"  Mean difference: ${np.mean(price_diffs):.4f}")
    print(f"  Std deviation: ${np.std(price_diffs):.4f}")
    print(f"  Min difference: ${np.min(price_diffs):.4f}")
    print(f"  Max difference: ${np.max(price_diffs):.4f}")
    print(f"  Entries at signal price: {np.sum(np.abs(price_diffs) < 0.01)} ({np.sum(np.abs(price_diffs) < 0.01)/len(price_diffs)*100:.1f}%)")
    print(f"  Entries worse than signal: {np.sum(price_diffs > 0.01)} ({np.sum(price_diffs > 0.01)/len(price_diffs)*100:.1f}%)")

# Check for same-bar entry pattern
print("\n\n=== SAME BAR vs NEXT BAR ENTRY ===")

same_bar_entries = 0
next_bar_entries = 0
delayed_entries = 0

for idx, pos in positions.head(50).iterrows():
    pos_time = pos['ts']
    
    # Find signal
    signal_mask = (signals['ts'] <= pos_time) & (signals['val'] != 0)
    if signal_mask.any():
        last_signal = signals[signal_mask].iloc[-1]
        signal_time = last_signal['ts']
        
        # Find market bars
        signal_bar_idx = market_data[market_data['timestamp'] <= signal_time].index[-1]
        entry_bar_idx = market_data[market_data['timestamp'] <= pos_time].index[-1]
        
        bar_delay = entry_bar_idx - signal_bar_idx
        
        if bar_delay == 0:
            same_bar_entries += 1
        elif bar_delay == 1:
            next_bar_entries += 1
        else:
            delayed_entries += 1

print(f"Entry timing patterns (first 50 positions):")
print(f"  Same bar as signal: {same_bar_entries}")
print(f"  Next bar after signal: {next_bar_entries}")
print(f"  Delayed (>1 bar): {delayed_entries}")

# Check for exit memory effect
print("\n\n=== EXIT MEMORY ANALYSIS ===")

# Look for positions with exit metadata
if 'metadata' in positions.columns:
    exit_positions = positions[positions['metadata'].notna()]
    print(f"Positions with metadata: {len(exit_positions)}")
    
    if len(exit_positions) > 0:
        print("\nSample metadata:")
        for idx, row in exit_positions.head(5).iterrows():
            print(f"  {row['ts']}: {row['metadata']}")