#!/usr/bin/env python3
"""
Check the structure of the parquet files to understand what columns are available.
"""

import pandas as pd
from pathlib import Path

# Find the latest results directory
results_base = Path("config/keltner/results")
if results_base.exists():
    # Get most recent directory
    result_dirs = [d for d in results_base.iterdir() if d.is_dir() and d.name != 'latest']
    if result_dirs:
        latest_dir = max(result_dirs, key=lambda d: d.stat().st_mtime)
        print(f"Using results from: {latest_dir}")
    else:
        print("No results directories found")
        exit(1)
else:
    print(f"Results directory not found: {results_base}")
    exit(1)

traces_dir = latest_dir / "traces" / "keltner_bands"
if not traces_dir.exists():
    print(f"No traces found in {traces_dir}")
    exit(1)

# Load first strategy file to check structure
first_file = traces_dir / "SPY_5m_compiled_strategy_0.parquet"
if first_file.exists():
    df = pd.read_parquet(first_file)
    print(f"\n📋 Parquet File Structure:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Index: {df.index.name}")
    print(f"   Data types:")
    for col in df.columns:
        print(f"     {col}: {df[col].dtype}")
    
    # Show first few rows
    print(f"\n📊 First 5 rows:")
    print(df.head())
    
    # Check for signal data
    print(f"\n🔍 Data Summary:")
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            unique_vals = df[col].nunique()
            non_zero = (df[col] != 0).sum()
            print(f"   {col}: {unique_vals} unique values, {non_zero} non-zero values")
    
    # Show value distribution for 'val' column
    if 'val' in df.columns:
        print(f"\n📊 Signal Value Distribution:")
        value_counts = df['val'].value_counts().sort_index()
        for val, count in value_counts.items():
            print(f"   Signal {val}: {count} occurrences")
    
    # Show some actual signal changes
    print(f"\n📈 First 10 Signal Changes:")
    for i, row in df.head(10).iterrows():
        if 'val' in df.columns and 'idx' in df.columns:
            print(f"   Bar {row['idx']}: signal={row['val']}, price={row.get('px', 'N/A')}")
else:
    print(f"❌ File not found: {first_file}")