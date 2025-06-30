#!/usr/bin/env python3
"""Simple analysis of entry timing and prices."""

import pandas as pd
from pathlib import Path

# Load data
positions = pd.read_parquet("/Users/daws/ADMF-PC/config/bollinger/results/20250628_194812/traces/portfolio/positions_open/positions_open.parquet")
signals = pd.read_parquet("/Users/daws/ADMF-PC/config/bollinger/results/20250628_194812/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")
fills = pd.read_parquet("/Users/daws/ADMF-PC/config/bollinger/results/20250628_194812/traces/execution/fills/execution_fills.parquet")

print("=== DATA OVERVIEW ===")
print(f"\nPositions: {len(positions)}")
print(f"Signals: {len(signals)}")
print(f"Fills: {len(fills)}")

# Check timestamps
print("\n=== TIMESTAMP ANALYSIS ===")
print("\nPosition timestamps (first 5):")
for idx, ts in enumerate(positions['ts'].head()):
    print(f"  {idx+1}: {ts}")

print("\nSignal timestamps (first 5 non-zero):")
non_zero_signals = signals[signals['val'] != 0].head()
for idx, row in non_zero_signals.iterrows():
    print(f"  {row['ts']}: val={row['val']}, px=${row['px']:.2f}")

print("\n=== ENTRY PRICES ===")
print("\nFirst 10 positions:")
print(f"{'Time':<30} {'Entry Price':>12} {'Strategy'}")
print("-" * 60)
for idx, row in positions.head(10).iterrows():
    print(f"{str(row['ts']):<30} ${row['entry_price']:>10.2f} {row['strategy_id']}")

# Compare with fills
print("\n\nFirst 10 fills:")
print(f"{'Time':<30} {'Fill Price':>12} {'Side':<6} {'Qty'}")
print("-" * 70)
for idx, row in fills.head(10).iterrows():
    print(f"{str(row['ts']):<30} ${row['price']:>10.2f} {row['side']:<6} {row['quantity']}")

# Check if all timestamps are from same run
print("\n=== TIMING PATTERNS ===")

# All positions have same timestamp pattern?
pos_ts = pd.to_datetime(positions['ts'])
print(f"\nPosition timestamp range:")
print(f"  Min: {pos_ts.min()}")
print(f"  Max: {pos_ts.max()}")
print(f"  Duration: {pos_ts.max() - pos_ts.min()}")

# Check if all positions were created at once
unique_seconds = pos_ts.dt.floor('S').nunique()
print(f"\nUnique seconds with positions: {unique_seconds}")

if unique_seconds < 10:
    print("  WARNING: All positions created in very short time!")
    print("  This suggests batch processing, not real-time signal execution")

# Analyze signal values
print("\n=== SIGNAL ANALYSIS ===")
signal_changes = signals[signals['val'].diff() != 0]
print(f"Signal changes: {len(signal_changes)}")
print(f"Non-zero signals: {(signals['val'] != 0).sum()}")

# Show signal pattern
print("\nSignal value distribution:")
print(signals['val'].value_counts().sort_index())

# Check metadata
print("\n=== METADATA ANALYSIS ===")
if 'metadata' in positions.columns and positions['metadata'].notna().any():
    print("Positions with metadata:")
    for idx, row in positions[positions['metadata'].notna()].head().iterrows():
        print(f"  {row['ts']}: {row['metadata']}")
else:
    print("No metadata found in positions")