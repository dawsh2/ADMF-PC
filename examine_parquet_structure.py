#!/usr/bin/env python3
"""
Examine the structure of the parquet files to understand the data format.
"""

import pandas as pd
from pathlib import Path

def examine_parquet_files():
    workspace_path = Path("workspaces/two_layer_regime_ensemble_v1_4f71d9e1")
    traces_path = workspace_path / "traces" / "SPY_1m"
    
    print("=== EXAMINING PARQUET FILE STRUCTURE ===\n")
    
    # Examine regime data
    regime_file = traces_path / "classifiers" / "regime" / "SPY_market_regime_detector.parquet"
    if regime_file.exists():
        print("REGIME DATA:")
        regime_data = pd.read_parquet(regime_file)
        print(f"Shape: {regime_data.shape}")
        print(f"Columns: {list(regime_data.columns)}")
        print(f"Data types: {dict(regime_data.dtypes)}")
        print("\nFirst 5 rows:")
        print(regime_data.head())
        print("\nSample of unique values in key columns:")
        for col in regime_data.columns:
            if regime_data[col].dtype == 'object':
                unique_vals = regime_data[col].unique()
                if len(unique_vals) <= 10:
                    print(f"  {col}: {list(unique_vals)}")
                else:
                    print(f"  {col}: {len(unique_vals)} unique values, sample: {list(unique_vals[:5])}")
        print("\n" + "="*60 + "\n")
    
    # Examine signal data files
    signal_dirs = [
        traces_path / "signals" / "ma_crossover",
        traces_path / "signals" / "regime"
    ]
    
    for signal_dir in signal_dirs:
        if signal_dir.exists():
            print(f"SIGNAL DATA FROM: {signal_dir.name}")
            
            for file_path in signal_dir.glob("*.parquet"):
                print(f"\nFile: {file_path.name}")
                try:
                    data = pd.read_parquet(file_path)
                    print(f"Shape: {data.shape}")
                    print(f"Columns: {list(data.columns)}")
                    print(f"Data types: {dict(data.dtypes)}")
                    print("\nFirst 3 rows:")
                    print(data.head(3))
                    
                    print("\nSample of unique values in key columns:")
                    for col in data.columns:
                        if data[col].dtype == 'object':
                            unique_vals = data[col].unique()
                            if len(unique_vals) <= 10:
                                print(f"  {col}: {list(unique_vals)}")
                            else:
                                print(f"  {col}: {len(unique_vals)} unique values, sample: {list(unique_vals[:5])}")
                        elif col in ['val', 'px']:  # Likely numeric columns
                            print(f"  {col}: min={data[col].min():.4f}, max={data[col].max():.4f}, unique={data[col].nunique()}")
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                
                print("\n" + "-"*50)
            
            print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    examine_parquet_files()