#!/usr/bin/env python3
"""Analyze why trailing stops aren't triggering."""

import pandas as pd
import json
from pathlib import Path
import sys

# Get results directory
if len(sys.argv) > 1:
    results_dir = Path(sys.argv[1])
else:
    results_base = Path("config/bollinger/results")
    if results_base.exists():
        results_dir = max(results_base.iterdir(), key=lambda p: p.stat().st_mtime)
    else:
        print("No results directory found")
        sys.exit(1)

print(f"Analyzing results from: {results_dir}")

# Load position data
positions_close_file = results_dir / "traces/portfolio/positions_close/positions_close.parquet"
if not positions_close_file.exists():
    print(f"No positions_close file found")
    sys.exit(1)

df = pd.read_parquet(positions_close_file)
print(f"Loaded {len(df)} position close records")

# Analyze each position
trailing_stop_eligible = 0
hit_stop_first = 0
hit_signal_first = 0
never_profitable = 0
max_gains = []

for _, row in df.iterrows():
    metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
    
    # The actual position data is at the top level
    entry_price = float(metadata.get('entry_price', 0))
    exit_type = metadata.get('exit_type', 'None')
    
    # highest_price is in the nested metadata
    nested_metadata = metadata.get('metadata', {})
    highest_price = float(nested_metadata.get('highest_price', entry_price)) if nested_metadata else entry_price
    
    if entry_price > 0 and highest_price > 0:
        max_gain = (highest_price - entry_price) / entry_price
        max_gains.append(max_gain)
        
        # Check if position ever gained enough for trailing stop
        if max_gain >= 0.0005:  # 0.05% trailing stop threshold
            trailing_stop_eligible += 1
            
            if exit_type == 'stop_loss':
                hit_stop_first += 1
            elif exit_type == 'signal':
                hit_signal_first += 1
        elif max_gain < 0:
            never_profitable += 1

print(f"\n=== Trailing Stop Analysis ===")
print(f"Total positions: {len(df)}")
print(f"Never went profitable: {never_profitable} ({never_profitable/len(df)*100:.1f}%)")
print(f"\nPositions that gained >= 0.05% (trailing stop eligible): {trailing_stop_eligible}")
if trailing_stop_eligible > 0:
    print(f"  - Exited via stop loss: {hit_stop_first} ({hit_stop_first/trailing_stop_eligible*100:.1f}%)")
    print(f"  - Exited via signal: {hit_signal_first} ({hit_signal_first/trailing_stop_eligible*100:.1f}%)")
    print(f"  - Exited via trailing stop: {df[df.apply(lambda x: json.loads(x['metadata'] if isinstance(x['metadata'], str) else x['metadata']).get('exit_type') == 'trailing_stop', axis=1)].shape[0]}")

# Always show distribution of max gains
print(f"\nFound {len(max_gains)} positions with valid price data")
if max_gains:
    import numpy as np
    print(f"\n=== Maximum Gain Distribution ===")
    print(f"Min: {min(max_gains)*100:.2f}%")
    print(f"25th percentile: {np.percentile(max_gains, 25)*100:.2f}%")
    print(f"Median: {np.percentile(max_gains, 50)*100:.2f}%")
    print(f"75th percentile: {np.percentile(max_gains, 75)*100:.2f}%")
    print(f"Max: {max(max_gains)*100:.2f}%")
    
    # How many reached different thresholds
    print(f"\nPositions reaching gain thresholds:")
    thresholds = [0.0001, 0.0003, 0.0005, 0.001, 0.002]
    for t in thresholds:
        count = sum(1 for g in max_gains if g >= t)
        print(f"  >= {t*100:.2f}%: {count} ({count/len(max_gains)*100:.1f}%)")
        
    # Show actual trailing stop hits
    trailing_hits = df[df.apply(lambda x: json.loads(x['metadata'] if isinstance(x['metadata'], str) else x['metadata']).get('exit_type') == 'trailing_stop', axis=1)]
    print(f"\nActual trailing stop exits: {len(trailing_hits)}")
    if len(trailing_hits) > 0:
        for _, hit in trailing_hits.iterrows():
            meta = json.loads(hit['metadata']) if isinstance(hit['metadata'], str) else hit['metadata']
            entry = float(meta.get('entry_price', 0))
            exit_price = float(meta.get('exit_price', 0))
            nested_meta = meta.get('metadata', {})
            highest = float(nested_meta.get('highest_price', entry)) if nested_meta else entry
            if entry > 0:
                max_gain = (highest - entry) / entry * 100
                exit_loss = (exit_price - highest) / highest * 100
                print(f"  Entry: ${entry:.2f}, High: ${highest:.2f} (+{max_gain:.3f}%), Exit: ${exit_price:.2f} ({exit_loss:.3f}% from high)")