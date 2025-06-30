#!/usr/bin/env python3
"""Check raw metadata content."""

import pandas as pd
import json
from pathlib import Path

results_dir = Path("config/bollinger/results/latest")
pos_open_file = results_dir / "traces/portfolio/positions_open/positions_open.parquet"

if pos_open_file.exists():
    opens = pd.read_parquet(pos_open_file)
    
    print(f"DataFrame columns: {list(opens.columns)}")
    print(f"\nFirst position raw data:")
    
    first_pos = opens.iloc[0]
    for col in opens.columns:
        val = first_pos[col]
        print(f"  {col}: {val} (type: {type(val).__name__})")
    
    # Look at metadata specifically
    print(f"\n\nMetadata field analysis:")
    metadata = first_pos.get('metadata', {})
    print(f"  Raw metadata: {repr(metadata)}")
    print(f"  Type: {type(metadata)}")
    
    if isinstance(metadata, str):
        print(f"  Length: {len(metadata)}")
        print(f"  First 200 chars: {metadata[:200]}")
        try:
            parsed = json.loads(metadata)
            print(f"  Parsed successfully!")
            print(f"  Parsed type: {type(parsed)}")
            print(f"  Parsed keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'Not a dict'}")
        except Exception as e:
            print(f"  Failed to parse: {e}")
    
    # Check if metadata is stored in a nested way
    if isinstance(metadata, dict):
        print(f"  Dict keys: {list(metadata.keys())}")
        if 'metadata' in metadata:
            print(f"  Has nested metadata key!")
            inner = metadata['metadata']
            print(f"  Inner type: {type(inner)}")
            if isinstance(inner, str):
                try:
                    inner_parsed = json.loads(inner)
                    print(f"  Inner parsed keys: {list(inner_parsed.keys()) if isinstance(inner_parsed, dict) else 'Not a dict'}")
                except:
                    pass