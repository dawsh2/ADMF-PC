#!/usr/bin/env python3
"""Quick fix to create strategy index"""
import json
import pandas as pd
import hashlib
from pathlib import Path
import re

run_dir = Path("config/bollinger/results/20250624_214106")
metadata_path = run_dir / 'metadata.json'

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

index_data = []
components = metadata.get('components', {})

print(f"Found {len(components)} components")

for component_id, component_info in components.items():
    if component_info.get('type') != 'strategy':
        continue
    
    strategy_type = component_info.get('strategy_type', 'unknown')
    
    # Extract from component_id
    clean_params = {}
    if 'bollinger_bands' in component_id:
        match = re.search(r'bollinger_bands_(\d+)_(\d+)$', component_id)
        if match:
            clean_params = {
                'period': int(match.group(1)),
                'std_dev': int(match.group(2)) / 10.0
            }
    
    # Compute unique hash
    hash_config = {
        'type': strategy_type,
        'parameters': clean_params
    }
    config_str = json.dumps(hash_config, sort_keys=True, separators=(',', ':'))
    strategy_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
    
    # Build entry
    entry = {
        'strategy_id': component_id,
        'strategy_hash': strategy_hash,
        'strategy_type': strategy_type,
        'symbol': 'SPY',
        'timeframe': '5m',
        'trace_path': f"traces/signals/{strategy_type}/{component_id}.parquet",
    }
    
    # Add parameters
    for k, v in clean_params.items():
        entry[k] = v
    
    index_data.append(entry)

if index_data:
    df = pd.DataFrame(index_data)
    print(f"\nGenerated {len(df)} strategy entries")
    print(f"Unique hashes: {df['strategy_hash'].nunique()}")
    
    # Save
    index_path = run_dir / 'strategy_index.parquet'
    df.to_parquet(index_path, engine='pyarrow', index=False)
    print(f"Saved to: {index_path}")
    
    # Show sample
    print("\nFirst 5 entries:")
    print(df[['strategy_id', 'strategy_hash', 'period', 'std_dev']].head())
else:
    print("No strategies found")