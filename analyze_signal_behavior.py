"""Analyze signal behavior to understand why we have so many trades."""

import pandas as pd
from pathlib import Path

# Load signal data
results_path = Path("config/bollinger/results/latest")
signal_path = results_path / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"

signal_df = pd.read_parquet(signal_path)

print("=== SIGNAL BEHAVIOR ANALYSIS ===\n")

# Count signal changes
print(f"Total signal changes: {len(signal_df)}")

# Analyze signal values
signal_values = signal_df['val'].value_counts().sort_index()
print(f"\nSignal value distribution:")
for val, count in signal_values.items():
    print(f"  Signal {val}: {count} times")

# Check for frequent signal changes (flipping)
print(f"\n=== SIGNAL CHANGE PATTERNS ===")

# Look at first 50 signal changes
print(f"\nFirst 20 signal changes:")
for i in range(min(20, len(signal_df))):
    row = signal_df.iloc[i]
    metadata = row['metadata']
    band_pos = metadata.get('band_position', 0) if isinstance(metadata, dict) else 0
    price = metadata.get('price', 0) if isinstance(metadata, dict) else 0
    
    print(f"{i+1}. Signal: {row['val']}, Price: ${price:.2f}, Band Position: {band_pos:.2f}")

# Check exit threshold behavior
print(f"\n=== CHECKING EXIT THRESHOLD ===")

# Look for signal = 0 entries
zero_signals = signal_df[signal_df['val'] == 0]
print(f"\nTotal neutral (0) signals: {len(zero_signals)}")

if len(zero_signals) > 0:
    print(f"\nAnalyzing first 10 neutral signals:")
    for i in range(min(10, len(zero_signals))):
        row = zero_signals.iloc[i]
        metadata = row['metadata']
        if isinstance(metadata, dict):
            price = metadata.get('price', 0)
            middle = metadata.get('middle_band', 0)
            if middle > 0:
                distance_pct = abs(price - middle) / middle * 100
                print(f"  Price: ${price:.2f}, Middle: ${middle:.2f}, Distance: {distance_pct:.3f}%")

# Check strategy configuration
print(f"\n=== STRATEGY CONFIGURATION ===")
if len(signal_df) > 0:
    first_metadata = signal_df.iloc[0]['metadata']
    if isinstance(first_metadata, dict):
        params = first_metadata.get('parameters', {})
        print(f"Period: {params.get('period', 'N/A')}")
        print(f"Std Dev: {params.get('std_dev', 'N/A')}")
        
        # Check for exit_threshold
        if 'exit_threshold' in params:
            print(f"Exit Threshold: {params.get('exit_threshold')}")
        else:
            print("Exit Threshold: NOT FOUND in parameters")