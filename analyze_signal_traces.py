#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def analyze_signal_traces(workspace_path):
    """Analyze signal traces to count signal changes"""
    
    workspace_path = Path(workspace_path)
    signals_dir = workspace_path / "traces/SPY_1m_1m/signals/swing_pivot_bounce_zones"
    
    if not signals_dir.exists():
        print(f"No signals directory found at {signals_dir}")
        return
    
    # Get all parquet files
    parquet_files = list(signals_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} signal files")
    
    total_signal_changes = 0
    total_non_zero_signals = 0
    
    # First check the structure
    if parquet_files:
        df_sample = pd.read_parquet(parquet_files[0])
        print(f"\nSample file structure:")
        print(f"Columns: {df_sample.columns.tolist()}")
        print(f"Shape: {df_sample.shape}")
        print(f"\nFirst few rows:")
        print(df_sample.head())
        
        # Check for signal-related columns
        signal_cols = [col for col in df_sample.columns if 'signal' in col.lower()]
        print(f"\nSignal-related columns: {signal_cols}")
        
        # If no 'signal' column, check for other likely candidates
        if 'signal' not in df_sample.columns:
            # Check if data is stored in sparse format
            if len(df_sample.columns) <= 3:  # Likely sparse format with timestamp, strategy_idx, signal
                print("\nAppears to be sparse format")
                print(f"Unique values in each column:")
                for col in df_sample.columns:
                    unique_count = df_sample[col].nunique()
                    print(f"  {col}: {unique_count} unique values")
                    if unique_count < 20:
                        print(f"    Values: {sorted(df_sample[col].unique())}")
            return
    
    # Analyze each file
    for i, file in enumerate(parquet_files[:10]):  # Sample first 10
        df = pd.read_parquet(file)
        
        # Count signal changes (transitions)
        signal_changes = (df['signal'].diff() != 0).sum() - 1  # -1 for the first NaN
        signal_changes = max(0, signal_changes)
        
        # Count non-zero signals
        non_zero = (df['signal'] != 0).sum()
        
        total_signal_changes += signal_changes
        total_non_zero_signals += non_zero
        
        if i < 5:  # Show details for first 5
            print(f"\nFile: {file.name}")
            print(f"  Shape: {df.shape}")
            print(f"  Signal changes: {signal_changes}")
            print(f"  Non-zero signals: {non_zero}")
            print(f"  Signal value counts:")
            print(df['signal'].value_counts().to_dict())
    
    print(f"\n--- Summary (first 10 files) ---")
    print(f"Total signal changes: {total_signal_changes}")
    print(f"Total non-zero signals: {total_non_zero_signals}")
    print(f"Average changes per file: {total_signal_changes / min(10, len(parquet_files)):.1f}")
    
    # Now check all files to get total count
    if len(parquet_files) > 10:
        print("\nAnalyzing all files...")
        all_changes = 0
        for file in parquet_files:
            df = pd.read_parquet(file)
            signal_changes = (df['signal'].diff() != 0).sum() - 1
            all_changes += max(0, signal_changes)
        
        print(f"\n--- Total across all {len(parquet_files)} files ---")
        print(f"Total signal changes: {all_changes}")
        print(f"Expected orders (signal changes - 1): {all_changes - 1}")

if __name__ == "__main__":
    workspace = sys.argv[1] if len(sys.argv) > 1 else "workspaces/signal_generation_20250622_095959"
    analyze_signal_traces(workspace)