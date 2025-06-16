#!/usr/bin/env python3
"""
Simple parquet file reader
Usage: python parquet-reader.py <parquet_file>
"""

import sys
import pandas as pd
from pathlib import Path

def read_parquet_file(filepath):
    """Read and display parquet file contents"""
    try:
        # Read the parquet file
        df = pd.read_parquet(filepath)
        
        print(f"File: {filepath}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 10 rows:")
        print(df.head(10))
        print("\nLast 10 rows:")
        print(df.tail(10))
        print(f"\nData types:")
        print(df.dtypes)
        
        # If it's a sparse signal file, show some statistics
        if 'ts' in df.columns and 'val' in df.columns:
            print(f"\nSignal Statistics:")
            print(f"Total signals: {len(df)}")
            print(f"Unique values: {df['val'].unique()}")
            print(f"Value counts:\n{df['val'].value_counts()}")
            print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parquet-reader.py <parquet_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        sys.exit(1)
    
    read_parquet_file(filepath)