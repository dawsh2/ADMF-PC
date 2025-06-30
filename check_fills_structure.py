"""Check fills structure."""

import pandas as pd
import pyarrow.parquet as pq

# Load the fill data
fills_path = 'config/bollinger/results/latest/traces/execution/fills/execution_fills.parquet'

try:
    fills_df = pd.read_parquet(fills_path)
    print(f"Found {len(fills_df)} fills")
    print(f"\nColumns: {list(fills_df.columns)}")
    print(f"\nFirst fill:")
    print(fills_df.iloc[0])
    
    # Check metadata structure
    if 'metadata' in fills_df.columns:
        first_metadata = fills_df.iloc[0]['metadata']
        print(f"\nFirst metadata type: {type(first_metadata)}")
        print(f"First metadata: {first_metadata}")
        
        # Look for fills with exit_type
        for idx in range(min(10, len(fills_df))):
            metadata = fills_df.iloc[idx]['metadata']
            if isinstance(metadata, dict) and 'exit_type' in metadata:
                print(f"\nFound exit fill at index {idx}:")
                print(f"  Metadata: {metadata}")
                break
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()