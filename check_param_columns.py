#!/usr/bin/env python3
"""Check parameter columns in strategy index"""
import pandas as pd
from pathlib import Path

index_path = Path("config/bollinger/results/latest/strategy_index.parquet")
if index_path.exists():
    df = pd.read_parquet(index_path)
    
    # Find parameter columns
    param_cols = [col for col in df.columns if col.startswith('param_')]
    print(f"Parameter columns found: {param_cols}")
    
    # Check data types
    print("\nParameter column data types:")
    for col in param_cols:
        print(f"  {col}: {df[col].dtype}")
        
    # Check for non-null values
    print("\nNon-null counts:")
    for col in param_cols:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(df)} values")
        
    # Show sample values
    print("\nSample values:")
    for col in param_cols:
        if df[col].notna().any():
            unique_vals = df[col].dropna().unique()
            print(f"  {col}: {unique_vals[:5]} (showing first 5)")
else:
    print(f"No strategy index found at {index_path}")

# Also check if there's a performance_metrics.parquet
perf_path = Path("config/bollinger/results/latest/performance_metrics.parquet")
if perf_path.exists():
    print("\n\nChecking performance_metrics.parquet:")
    perf_df = pd.read_parquet(perf_path)
    perf_param_cols = [col for col in perf_df.columns if col.startswith('param_')]
    print(f"Parameter columns in performance data: {perf_param_cols}")
    for col in perf_param_cols:
        print(f"  {col}: {perf_df[col].dtype}, {perf_df[col].notna().sum()} non-null values")