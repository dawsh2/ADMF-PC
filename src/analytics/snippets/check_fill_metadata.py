# Check the actual timestamps in fill metadata
import pandas as pd
from pathlib import Path
import json

# Load a sample fill to inspect metadata
results_dir = Path('/Users/daws/ADMF-PC/config/bollinger/results/latest')
fills_path = results_dir / 'traces/execution/fills'

fills_files = list(fills_path.glob('*.parquet'))
if fills_files:
    df = pd.read_parquet(fills_files[0])
    print("First fill metadata:")
    if len(df) > 0:
        meta = df.iloc[0]['metadata']
        if isinstance(meta, str):
            meta = json.loads(meta)
        
        print(json.dumps(meta, indent=2))
        
        # Extract executed_at timestamp
        if 'executed_at' in meta:
            print(f"\nActual execution time: {meta['executed_at']}")
            print(f"Trace timestamp (ts): {df.iloc[0]['ts']}")
            
    # Show first 5 fills with extracted timestamps
    print("\nFirst 5 fills with actual execution times:")
    for i in range(min(5, len(df))):
        meta = df.iloc[i]['metadata']
        if isinstance(meta, str):
            meta = json.loads(meta)
        executed_at = meta.get('executed_at', 'N/A')
        print(f"Fill {i+1}: executed_at={executed_at}, ts={df.iloc[i]['ts']}")