#!/usr/bin/env python3
"""
Examine the structure of the ensemble signal files to understand the data format.
"""

import pandas as pd
from pathlib import Path

def examine_file(file_path: Path):
    """Examine a parquet file in detail."""
    print(f"\n{'='*60}")
    print(f"EXAMINING: {file_path.name}")
    print(f"{'='*60}")
    
    try:
        df = pd.read_parquet(file_path)
        
        print(f"Shape: {df.shape}")
        print(f"Index type: {type(df.index)}")
        print(f"Index name: {df.index.name}")
        
        print(f"\nColumns and types:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        
        print(f"\nFirst 10 rows:")
        print(df.head(10))
        
        print(f"\nLast 5 rows:")
        print(df.tail(5))
        
        # Look at unique values in key columns
        for col in ['val', 'strat']:
            if col in df.columns:
                unique_vals = df[col].unique()
                print(f"\nUnique values in '{col}' ({len(unique_vals)} total):")
                if len(unique_vals) <= 20:
                    print(f"  {unique_vals}")
                else:
                    print(f"  First 10: {unique_vals[:10]}")
                    print(f"  Last 10: {unique_vals[-10:]}")
        
        # Check timestamp column
        if 'ts' in df.columns:
            print(f"\nTimestamp info:")
            print(f"  Type: {df['ts'].dtype}")
            print(f"  Range: {df['ts'].min()} to {df['ts'].max()}")
            print(f"  Sample values: {df['ts'].head(3).tolist()}")
        
        # Check price column (px)
        if 'px' in df.columns:
            print(f"\nPrice info:")
            print(f"  Type: {df['px'].dtype}")
            print(f"  Range: ${df['px'].min():.2f} to ${df['px'].max():.2f}")
            print(f"  Mean: ${df['px'].mean():.2f}")
            print(f"  Sample values: {df['px'].head(5).tolist()}")
        
        return df
        
    except Exception as e:
        print(f"Error examining {file_path}: {e}")
        return None

def main():
    """Main examination function."""
    workspace_path = Path("/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_v1_9c2c22c9")
    signals_path = workspace_path / "traces" / "SPY_1m" / "signals" / "unknown"
    
    # File paths
    default_file = signals_path / "SPY_adaptive_ensemble_default.parquet"
    custom_file = signals_path / "SPY_adaptive_ensemble_custom.parquet"
    
    print("🔍 Examining signal trace file structures...")
    
    # Examine both files
    default_data = examine_file(default_file)
    custom_data = examine_file(custom_file)
    
    # Compare structures
    if default_data is not None and custom_data is not None:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Default file: {len(default_data)} rows")
        print(f"Custom file: {len(custom_data)} rows")
        print(f"Difference: {len(default_data) - len(custom_data)} rows")
        
        # Check if they have same columns
        default_cols = set(default_data.columns)
        custom_cols = set(custom_data.columns)
        
        if default_cols == custom_cols:
            print("✅ Both files have identical column structure")
        else:
            print("❌ Column structures differ:")
            print(f"  Default only: {default_cols - custom_cols}")
            print(f"  Custom only: {custom_cols - default_cols}")

if __name__ == "__main__":
    main()