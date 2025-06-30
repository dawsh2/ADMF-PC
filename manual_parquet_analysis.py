#!/usr/bin/env python3
"""Manual analysis of the ensemble parquet file without complex imports"""

import struct
import sys

# Try to use pyarrow directly
try:
    import pyarrow.parquet as pq
    
    file_path = 'config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet'
    
    # Read parquet file metadata
    parquet_file = pq.ParquetFile(file_path)
    
    print("=== PARQUET FILE METADATA ===")
    print(f"Number of row groups: {parquet_file.num_row_groups}")
    
    # Get schema
    schema = parquet_file.schema
    print(f"\nSchema:")
    for i in range(len(schema)):
        field = schema[i]
        print(f"  - {field.name}: {field.type}")
    
    # Read first row group
    first_row_group = parquet_file.read_row_group(0)
    df = first_row_group.to_pandas()
    
    print(f"\nFirst row group shape: {df.shape}")
    print("\nFirst 10 rows:")
    print(df.head(10))
    
    # Read all data
    table = parquet_file.read()
    full_df = table.to_pandas()
    
    print(f"\n=== FULL DATA ANALYSIS ===")
    print(f"Total rows: {len(full_df)}")
    print(f"Columns: {list(full_df.columns)}")
    
    if 'val' in full_df.columns:
        print("\nSignal value distribution:")
        print(full_df['val'].value_counts().sort_index())
        
        # Count signal changes
        signal_changes = (full_df['val'] != full_df['val'].shift()).sum() - 1
        print(f"\nTotal signal changes: {signal_changes}")
        
        # Show transitions
        print("\nFirst 20 signal transitions:")
        transitions = full_df[full_df['val'] != full_df['val'].shift()].head(20)
        for idx, row in transitions.iterrows():
            print(f"  Row {idx}: idx={row['idx']}, val={row['val']}, px={row.get('px', 'N/A')}")
    
    # Check for sub-strategy information
    print("\n=== CHECKING FOR SUB-STRATEGY INFO ===")
    for col in full_df.columns:
        if col not in ['idx', 'val', 'px']:
            unique_vals = full_df[col].nunique()
            print(f"{col}: {unique_vals} unique values")
            if unique_vals < 20:
                print(f"  Values: {full_df[col].unique()}")
    
except ImportError:
    print("PyArrow not available")
    
# Try pandas as fallback
try:
    import pandas as pd
    
    file_path = 'config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet'
    df = pd.read_parquet(file_path)
    
    print("\n=== PANDAS ANALYSIS ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head())
    
except Exception as e:
    print(f"Error with pandas: {e}")

print("\nAnalysis complete!")