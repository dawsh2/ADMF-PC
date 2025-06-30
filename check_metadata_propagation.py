#!/usr/bin/env python3
"""Check if position metadata is being propagated correctly."""

import pandas as pd
import json
from pathlib import Path

print("=== Checking Metadata Propagation ===")

# Try both possible paths
results_dir = Path("config/bollinger/results/latest")
if not results_dir.exists():
    # If running from within a results directory
    results_dir = Path(".")

# Check position opens for entry_signal in metadata
pos_open_file = results_dir / "traces/portfolio/positions_open/positions_open.parquet"

if pos_open_file.exists():
    opens = pd.read_parquet(pos_open_file)
    
    print(f"Total position opens: {len(opens)}")
    
    # Check first few positions
    print("\nChecking first 5 positions for entry_signal in metadata:")
    
    for i in range(min(5, len(opens))):
        pos = opens.iloc[i]
        print(f"\n{i+1}. Position at bar {pos['idx']}:")
        
        # Check metadata
        metadata = pos.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                pass
        
        if isinstance(metadata, dict):
            # Check if there's a nested metadata
            actual_metadata = metadata.get('metadata', metadata)
            if isinstance(actual_metadata, dict) and 'entry_signal' in actual_metadata:
                print(f"   ✓ entry_signal: {actual_metadata['entry_signal']}")
            else:
                print(f"   ❌ NO entry_signal in metadata")
                print(f"   Available keys in parsed: {list(metadata.keys())[:10]}")
                if 'metadata' in metadata and isinstance(metadata['metadata'], dict):
                    print(f"   Available keys in nested: {list(metadata['metadata'].keys())[:10]}")
        else:
            print(f"   ❌ Metadata is not a dict: {type(metadata)}")
    
    # Count how many have entry_signal
    has_entry_signal = 0
    for i in range(len(opens)):
        metadata = opens.iloc[i].get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        if isinstance(metadata, dict):
            actual_metadata = metadata.get('metadata', metadata)
            if isinstance(actual_metadata, dict) and 'entry_signal' in actual_metadata:
                has_entry_signal += 1
    
    print(f"\n=== Summary ===")
    print(f"Positions with entry_signal: {has_entry_signal}/{len(opens)}")
    
    if has_entry_signal == 0:
        print("\n⚠️ NO positions have entry_signal in metadata!")
        print("This means our code to store entry_signal is not being executed.")
        print("The Python process needs to be restarted for changes to take effect.")

else:
    print("No position open data found")