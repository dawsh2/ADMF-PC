#!/usr/bin/env python3
"""Debug position events to understand the data structure."""

import pandas as pd
import json
from pathlib import Path

# Load position data
latest_dir = Path("config/bollinger/results/latest")
positions_open = latest_dir / "traces/portfolio/positions_open/positions_open.parquet"
positions_close = latest_dir / "traces/portfolio/positions_close/positions_close.parquet"

# Load and examine the data
df_open = pd.read_parquet(positions_open)
df_close = pd.read_parquet(positions_close)

print("=== POSITION OPEN ANALYSIS ===")
print(f"Total opens: {len(df_open)}")

# Parse first few metadata entries
print("\nFirst 5 position opens:")
for i in range(min(5, len(df_open))):
    metadata = json.loads(df_open.iloc[i]['metadata'])
    print(f"\n{i+1}. Time: {df_open.iloc[i]['ts']}, Index: {df_open.iloc[i]['idx']}")
    print(f"   Strategy: {metadata.get('strategy_id', 'unknown')}")
    print(f"   Quantity: {metadata.get('quantity')}")
    print(f"   Entry Price: {metadata.get('entry_price')}")

print("\n=== POSITION CLOSE ANALYSIS ===")
print(f"Total closes: {len(df_close)}")

# Parse first few metadata entries
print("\nFirst 5 position closes:")
for i in range(min(5, len(df_close))):
    metadata = json.loads(df_close.iloc[i]['metadata'])
    print(f"\n{i+1}. Time: {df_close.iloc[i]['ts']}, Index: {df_close.iloc[i]['idx']}")
    print(f"   Strategy: {metadata.get('strategy_id', 'unknown')}")
    print(f"   Exit Type: {metadata.get('exit_type', 'NONE')}")
    print(f"   Exit Reason: {metadata.get('exit_reason', 'NONE')}")
    print(f"   Quantity: {metadata.get('quantity')}")
    print(f"   Exit Price: {metadata.get('exit_price')}")

# Check for immediate re-entries by comparing timestamps
print("\n=== CHECKING FOR IMMEDIATE RE-ENTRIES ===")

# Combine and sort by timestamp
df_open['event_type'] = 'OPEN'
df_close['event_type'] = 'CLOSE'
all_events = pd.concat([df_open, df_close]).sort_values('ts')

# Parse metadata for all events
all_events['parsed_metadata'] = all_events['metadata'].apply(json.loads)
all_events['strategy_id'] = all_events['parsed_metadata'].apply(lambda x: x.get('strategy_id', 'strategy_0'))
all_events['exit_type'] = all_events['parsed_metadata'].apply(lambda x: x.get('exit_type'))

# Look for close followed by open for same strategy
for i in range(len(all_events) - 1):
    curr = all_events.iloc[i]
    next_evt = all_events.iloc[i + 1]
    
    if (curr['event_type'] == 'CLOSE' and 
        next_evt['event_type'] == 'OPEN' and
        curr['exit_type'] in ['stop_loss', 'trailing_stop', 'take_profit'] and
        curr['strategy_id'] == next_evt['strategy_id']):
        
        time_diff = (next_evt['ts'] - curr['ts']).total_seconds() / 60
        print(f"\nFound re-entry:")
        print(f"  Close at {curr['ts']} (idx: {curr['idx']}) - {curr['exit_type']}")
        print(f"  Open at {next_evt['ts']} (idx: {next_evt['idx']})")
        print(f"  Time difference: {time_diff:.2f} minutes")
        print(f"  Strategy: {curr['strategy_id']}")

# Also check if opens and closes are matched by index
print("\n=== INDEX MATCHING ===")
print(f"Open indices: min={df_open['idx'].min()}, max={df_open['idx'].max()}")
print(f"Close indices: min={df_close['idx'].min()}, max={df_close['idx'].max()}")

# Check for any opens that happen right after closes (by index)
close_indices = set(df_close['idx'].values)
for i, open_idx in enumerate(df_open['idx'].values):
    # Check if there's a close at index-1 or index-2
    if (open_idx - 1) in close_indices or (open_idx - 2) in close_indices:
        close_before = None
        if (open_idx - 1) in close_indices:
            close_before = df_close[df_close['idx'] == (open_idx - 1)].iloc[0]
        elif (open_idx - 2) in close_indices:
            close_before = df_close[df_close['idx'] == (open_idx - 2)].iloc[0]
        
        if close_before is not None:
            close_meta = json.loads(close_before['metadata'])
            open_meta = json.loads(df_open.iloc[i]['metadata'])
            
            if close_meta.get('exit_type') in ['stop_loss', 'trailing_stop', 'take_profit']:
                print(f"\nPotential immediate re-entry by index:")
                print(f"  Close idx {close_before['idx']} at {close_before['ts']} - {close_meta.get('exit_type')}")
                print(f"  Open idx {open_idx} at {df_open.iloc[i]['ts']}")
                print(f"  Strategies: {close_meta.get('strategy_id')} -> {open_meta.get('strategy_id')}")