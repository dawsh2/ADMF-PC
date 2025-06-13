#!/usr/bin/env python3
"""
Simple script to read and display Parquet files from signal traces.
Usage: python read_parquet.py <file_path>
"""

import sys
import pandas as pd
import json

def read_parquet_file(file_path):
    """Read and display a Parquet file with metadata."""
    print(f"Reading: {file_path}")
    print("=" * 50)
    
    # Read the data
    df = pd.read_parquet(file_path)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Display the data
    print("Data:")
    print(df.to_string(index=False))
    print()
    
    # Try to read metadata
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(file_path)
        if table.schema.metadata:
            print("Metadata:")
            for key, value in table.schema.metadata.items():
                key_str = key.decode() if isinstance(key, bytes) else str(key)
                value_str = value.decode() if isinstance(value, bytes) else str(value)
                
                # Pretty print JSON metadata
                if key_str in ['strategies', 'signal_statistics']:
                    try:
                        parsed = json.loads(value_str)
                        print(f"  {key_str}:")
                        print(f"    {json.dumps(parsed, indent=2)}")
                    except:
                        print(f"  {key_str}: {value_str}")
                else:
                    print(f"  {key_str}: {value_str}")
        else:
            print("No metadata found")
    except Exception as e:
        print(f"Could not read metadata: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_parquet.py <file_path>")
        print("\nExample files:")
        import os
        workspace = "workspaces/expansive_grid_search_bc73ecec/traces"
        if os.path.exists(workspace):
            for root, dirs, files in os.walk(workspace):
                for file in files[:3]:  # Show first 3 files
                    if file.endswith('.parquet'):
                        print(f"  {os.path.join(root, file)}")
        sys.exit(1)
    
    read_parquet_file(sys.argv[1])