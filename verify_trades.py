#!/usr/bin/env python3
"""Verify if signals are being converted to trades properly."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load a signal file
signal_file = Path("config/bollinger/results/20250625_185742/traces/signals/bollinger_bands/SPY_1m_strategy_81.parquet")
df = pd.read_parquet(signal_file)

print(f"Analyzing signals from: {signal_file.name}")
print(f"Total rows: {len(df)}")
print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")

# Convert timestamps to datetime
df['ts'] = pd.to_datetime(df['ts'])

# Sort by timestamp
df = df.sort_values('ts').reset_index(drop=True)

# Extract trades from signals
trades = []
current_position = 0

for idx, row in df.iterrows():
    signal = row['val']
    
    # Position change = trade
    if signal != current_position:
        if current_position != 0:  # Exit previous position
            trades.append({
                'exit_time': row['ts'],
                'exit_signal': signal,
                'exit_price': row['px'],
                'position_closed': current_position
            })
        
        if signal != 0:  # Enter new position
            trades.append({
                'entry_time': row['ts'],
                'entry_signal': signal,
                'entry_price': row['px'],
                'position_opened': signal
            })
        
        current_position = signal

print(f"\nSignal distribution:")
print(df['val'].value_counts())

print(f"\nTrades found: {len(trades)}")
print("\nFirst 10 trades:")
for i, trade in enumerate(trades[:10]):
    print(f"{i+1}: {trade}")

# Count round-trip trades
entries = [t for t in trades if 'entry_time' in t]
exits = [t for t in trades if 'exit_time' in t]
print(f"\nEntries: {len(entries)}")
print(f"Exits: {len(exits)}")
print(f"Complete round-trips: {min(len(entries), len(exits))}")