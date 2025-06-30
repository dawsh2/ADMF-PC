#!/usr/bin/env python3
"""Debug why exit memory isn't blocking these 47 trades."""

import pandas as pd
import json
from pathlib import Path

results_dir = Path("config/bollinger/results/latest")

# Load data
opens = pd.read_parquet(results_dir / "traces/portfolio/positions_open/positions_open.parquet")
closes = pd.read_parquet(results_dir / "traces/portfolio/positions_close/positions_close.parquet")

# Parse metadata to get entry_signal
for i in range(len(closes)):
    if isinstance(closes.iloc[i]['metadata'], str):
        try:
            meta = json.loads(closes.iloc[i]['metadata'])
            # Extract entry_signal from parsed metadata
            if 'entry_signal' in meta:
                closes.loc[closes.index[i], 'entry_signal'] = meta['entry_signal']
            # Also check nested metadata
            if 'metadata' in meta and 'entry_signal' in meta['metadata']:
                closes.loc[closes.index[i], 'entry_signal'] = meta['metadata']['entry_signal']
        except:
            pass

print("=== Debugging Exit Memory Failure ===\n")

# Check if entry_signal is in close events
closes_with_entry = closes[closes['entry_signal'].notna()] if 'entry_signal' in closes.columns else pd.DataFrame()
print(f"Closes with entry_signal: {len(closes_with_entry)}/{len(closes)}")

if len(closes_with_entry) == 0:
    print("\n❌ PROBLEM: entry_signal is not propagated to position close events!")
    print("This means exit memory can't know what signal to store.")
    
    # Check a few close events
    print("\nFirst 5 close events:")
    for i in range(min(5, len(closes))):
        close = closes.iloc[i]
        print(f"\n{i+1}. Close at bar {close['idx']}:")
        
        # Parse metadata
        if isinstance(close['metadata'], str):
            try:
                meta = json.loads(close['metadata'])
                print(f"   Parsed metadata keys: {list(meta.keys())}")
                if 'metadata' in meta:
                    print(f"   Nested metadata keys: {list(meta['metadata'].keys())}")
            except:
                print(f"   Failed to parse metadata")
else:
    print("\n✓ entry_signal is present in close events")
    
    # Check what values are stored
    print("\nEntry signal values in closes:")
    print(closes_with_entry['entry_signal'].value_counts())