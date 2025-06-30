#!/usr/bin/env python3
"""Check the strategy index contents"""
import pandas as pd
from pathlib import Path

# Load the strategy index
index_path = Path("config/bollinger/results/latest/strategy_index.parquet")
if index_path.exists():
    df = pd.read_parquet(index_path)
    print(f"Strategy index shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Check unique values
    print(f"\nUnique strategy hashes: {df['strategy_hash'].nunique()}")
    print(f"Strategy hash values: {df['strategy_hash'].unique()[:5]}")
    
    # Check parameters
    param_cols = [col for col in df.columns if col.startswith('param_')]
    print(f"\nParameter columns: {param_cols}")
    
    if param_cols:
        print("\nParameter value samples:")
        for col in param_cols[:3]:
            if df[col].notna().any():
                print(f"  {col}: {df[col].dropna().unique()[:5]}")
else:
    print(f"No strategy index found at {index_path}")