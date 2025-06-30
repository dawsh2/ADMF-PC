# Inspect the structure of fills data
import pandas as pd
from pathlib import Path

# Load a sample fill file to inspect structure
results_dir = Path('/Users/daws/ADMF-PC/config/bollinger/results/latest')
fills_path = results_dir / 'traces/execution/fills'

print("Inspecting fills structure...")
fills_files = list(fills_path.glob('*.parquet'))

if fills_files:
    # Load first file to inspect
    sample_df = pd.read_parquet(fills_files[0])
    print(f"\nColumns in fills parquet: {sample_df.columns.tolist()}")
    print(f"\nData types:")
    print(sample_df.dtypes)
    print(f"\nFirst row as dict:")
    if len(sample_df) > 0:
        first_row = sample_df.iloc[0].to_dict()
        for key, value in first_row.items():
            print(f"  {key}: {value} (type: {type(value).__name__})")
    
    # Check if metadata is nested
    if 'metadata' in sample_df.columns and len(sample_df) > 0:
        print("\nMetadata structure:")
        meta = sample_df.iloc[0]['metadata']
        if isinstance(meta, dict):
            for key, value in meta.items():
                print(f"  metadata.{key}: {value}")
        else:
            print(f"  metadata type: {type(meta)}")
            
    # Show first few rows
    print("\nFirst 3 rows:")
    print(sample_df.head(3))
else:
    print("No fills files found!")