#!/usr/bin/env python3
"""Inspect the structure of Bollinger Band parquet files."""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

# Path to a sample parquet file
SAMPLE_FILE = Path("/Users/daws/ADMF-PC/config/bollinger/results/20250623_062931/traces/bollinger_bands/SPY_5m_compiled_strategy_0.parquet")

def inspect_parquet():
    """Inspect the structure and content of a parquet file."""
    print(f"Inspecting: {SAMPLE_FILE}")
    print("-" * 60)
    
    try:
        # Read the parquet file
        df = pd.read_parquet(SAMPLE_FILE)
        
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nData types:\n{df.dtypes}")
        
        print(f"\nFirst 10 rows:")
        print(df.head(10))
        
        print(f"\nLast 10 rows:")
        print(df.tail(10))
        
        # Check for any non-null values
        print(f"\nNon-null counts:")
        print(df.count())
        
        # Check unique values in each column
        print(f"\nUnique value counts:")
        for col in df.columns:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            if unique_count < 20:  # Show values if not too many
                print(f"    Values: {sorted(df[col].unique())}")
        
        # Check if there's any data that's not 0
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            non_zero = (df[col] != 0).sum()
            if non_zero > 0:
                print(f"\n{col} has {non_zero} non-zero values ({non_zero/len(df)*100:.2f}%)")
                
    except Exception as e:
        print(f"Error reading file: {e}")
        
        # Try to read the parquet schema
        try:
            parquet_file = pq.ParquetFile(SAMPLE_FILE)
            print(f"\nParquet schema:")
            print(parquet_file.schema)
            
            print(f"\nMetadata:")
            print(parquet_file.metadata)
        except Exception as e2:
            print(f"Error reading schema: {e2}")

if __name__ == "__main__":
    inspect_parquet()