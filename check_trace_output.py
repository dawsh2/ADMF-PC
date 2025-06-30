#!/usr/bin/env python3
"""Check if any traces were saved."""

import os
import glob

# Check for trace files
trace_files = glob.glob('traces/**/*.parquet', recursive=True)
print(f"Found {len(trace_files)} trace files")

if trace_files:
    import pandas as pd
    import pyarrow.parquet as pq
    
    # Check the first file
    first_file = trace_files[0]
    print(f"\nChecking {first_file}:")
    
    # Read the parquet file
    df = pd.read_parquet(first_file)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Check for signal_value column
    if 'signal_value' in df.columns:
        signal_counts = df['signal_value'].value_counts()
        print(f"\n  Signal values:")
        for value, count in signal_counts.items():
            print(f"    {value}: {count}")
        
        # Show a few example signals
        non_zero = df[df['signal_value'] != 0]
        if not non_zero.empty:
            print(f"\n  Example non-zero signals:")
            print(non_zero[['timestamp', 'signal_value', 'strategy_id']].head())
    else:
        print("  No signal_value column found")
else:
    print("\nNo trace files found. Checking for sparse storage...")
    
    # Check for streaming sparse storage
    sparse_files = glob.glob('results/**/*.parquet', recursive=True)
    print(f"Found {len(sparse_files)} sparse storage files")
    
    if sparse_files:
        for f in sparse_files[:5]:  # Check first 5
            print(f"\n{f}:")
            df = pd.read_parquet(f)
            print(f"  Rows: {len(df)}")
            if 'signal_value' in df.columns:
                non_zero = df[df['signal_value'] != 0]
                print(f"  Non-zero signals: {len(non_zero)}")