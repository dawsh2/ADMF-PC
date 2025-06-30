#!/usr/bin/env python3
"""Analyze signals from notebook results directory."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load signals from notebook results
signal_file = Path("config/bollinger/results/20250625_170201/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")
signals_df = pd.read_parquet(signal_file)

print("=== Notebook Signal Analysis ===")
print(f"Total signal records: {len(signals_df)}")
print(f"Columns: {list(signals_df.columns)}")

# Convert timestamp and sort
signals_df['ts'] = pd.to_datetime(signals_df['ts'])
signals_df = signals_df.sort_values('ts')

# Analyze signal changes
signals_df['signal_change'] = signals_df['val'].diff().fillna(0) != 0
signal_changes = signals_df[signals_df['signal_change']]

print(f"\nSignal changes: {len(signal_changes)}")
print(f"Date range: {signals_df['ts'].min()} to {signals_df['ts'].max()}")

# Count positions (entry signals)
# A new position is when signal goes from 0 to non-zero
signals_df['prev_val'] = signals_df['val'].shift(1).fillna(0)
entries = signals_df[(signals_df['prev_val'] == 0) & (signals_df['val'] != 0)]
print(f"\nEntry signals: {len(entries)}")

# Analyze signal values
print(f"\nUnique signal values: {sorted(signals_df['val'].unique())}")
value_counts = signals_df['val'].value_counts()
print("\nSignal value distribution:")
for val, count in value_counts.items():
    print(f"  {val}: {count} ({count/len(signals_df)*100:.1f}%)")

# Calculate approximate trades (entries)
# Each entry that's followed by an exit is a trade
exits = signals_df[(signals_df['prev_val'] != 0) & (signals_df['val'] == 0)]
print(f"\nExit signals: {len(exits)}")
print(f"Approximate trades: {min(len(entries), len(exits))}")

# Show sample of entries
print("\nFirst 10 entry signals:")
for idx, (_, row) in enumerate(entries.head(10).iterrows()):
    print(f"  {row['ts']}: signal = {row['val']}, price = {row.get('px', 'N/A')}")

# Compare with latest results
latest_signal_file = Path("config/bollinger/results/latest/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")
if latest_signal_file.exists():
    latest_signals = pd.read_parquet(latest_signal_file)
    latest_signals['ts'] = pd.to_datetime(latest_signals['ts'])
    latest_signals = latest_signals.sort_values('ts')
    
    # Check if same time range
    print(f"\n=== Comparison with Latest Results ===")
    print(f"Latest signals: {len(latest_signals)} records")
    print(f"Latest date range: {latest_signals['ts'].min()} to {latest_signals['ts'].max()}")
    
    # Count signal changes in latest
    latest_signals['signal_change'] = latest_signals['val'].diff().fillna(0) != 0
    latest_changes = latest_signals[latest_signals['signal_change']]
    
    latest_signals['prev_val'] = latest_signals['val'].shift(1).fillna(0)
    latest_entries = latest_signals[(latest_signals['prev_val'] == 0) & (latest_signals['val'] != 0)]
    
    print(f"\nLatest signal changes: {len(latest_changes)}")
    print(f"Latest entry signals: {len(latest_entries)}")
    
    # Check if signals are identical
    if len(signals_df) == len(latest_signals):
        # Compare values
        if signals_df['val'].equals(latest_signals['val'].iloc[:len(signals_df)]):
            print("\n✅ Signals are IDENTICAL between notebook and latest results")
        else:
            print("\n❌ Signals DIFFER between notebook and latest results")
            # Find first difference
            for i in range(min(len(signals_df), len(latest_signals))):
                if signals_df.iloc[i]['val'] != latest_signals.iloc[i]['val']:
                    print(f"First difference at index {i}:")
                    print(f"  Notebook: {signals_df.iloc[i]['ts']} = {signals_df.iloc[i]['val']}")
                    print(f"  Latest: {latest_signals.iloc[i]['ts']} = {latest_signals.iloc[i]['val']}")
                    break
    else:
        print(f"\n❌ Different number of signals: {len(signals_df)} vs {len(latest_signals)}")