#!/usr/bin/env python3
"""Verify we analyzed all strategies in the workspace."""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

workspace = "workspaces/signal_generation_a2d31737"
signal_pattern = str(Path(workspace) / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones/*.parquet")
signal_files = sorted(glob(signal_pattern))

print(f"Total signal files found: {len(signal_files)}")
print(f"First file: {signal_files[0] if signal_files else 'None'}")
print(f"Last file: {signal_files[-1] if signal_files else 'None'}")

# Check file sizes
total_size = 0
min_size = float('inf')
max_size = 0

for f in signal_files:
    size = Path(f).stat().st_size
    total_size += size
    min_size = min(min_size, size)
    max_size = max(max_size, size)

print(f"\nFile sizes:")
print(f"  Total: {total_size / 1024 / 1024:.2f} MB")
print(f"  Min: {min_size} bytes")
print(f"  Max: {max_size} bytes")
print(f"  Average: {total_size / len(signal_files):.0f} bytes")

# Sample analysis - check a few strategies
print("\nSampling some strategies:")
sample_indices = [0, 100, 500, 1000, 1499]

for idx in sample_indices:
    if idx < len(signal_files):
        try:
            df = pd.read_parquet(signal_files[idx])
            strategy_name = Path(signal_files[idx]).stem
            print(f"\nStrategy {idx} ({strategy_name}):")
            print(f"  Signals: {len(df)}")
            print(f"  Non-zero signals: {len(df[df['val'] != 0])}")
            
            # Quick trade count
            trades = 0
            in_position = False
            for _, row in df.iterrows():
                if row['val'] != 0 and not in_position:
                    in_position = True
                elif row['val'] == 0 and in_position:
                    trades += 1
                    in_position = False
            print(f"  Estimated trades: {trades}")
        except Exception as e:
            print(f"  Error: {e}")

# Now let's properly count how many we can analyze
print("\n\nFull analysis check:")
analyzed_count = 0
error_count = 0
low_trade_count = 0

for i, signal_file in enumerate(signal_files):
    if i % 100 == 0:
        print(f"Checking {i}/{len(signal_files)}...", end='\r')
    
    try:
        df = pd.read_parquet(signal_file)
        
        # Count trades
        trades = 0
        in_position = False
        for _, row in df.iterrows():
            if row['val'] != 0 and not in_position:
                in_position = True
            elif row['val'] == 0 and in_position:
                trades += 1
                in_position = False
        
        if trades >= 10:
            analyzed_count += 1
        else:
            low_trade_count += 1
            
    except Exception as e:
        error_count += 1

print(f"\n\nSummary:")
print(f"Total files: {len(signal_files)}")
print(f"Successfully analyzed (10+ trades): {analyzed_count}")
print(f"Low trade count (<10 trades): {low_trade_count}")
print(f"Errors: {error_count}")
print(f"Coverage: {analyzed_count / len(signal_files) * 100:.1f}%")