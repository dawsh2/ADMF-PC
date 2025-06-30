#!/usr/bin/env python3
"""
Generate missing strategy_index.parquet from metadata.json
"""
import json
import pandas as pd
import hashlib
from pathlib import Path
import sys

def generate_strategy_index(run_dir):
    """Generate strategy index from metadata.json"""
    run_dir = Path(run_dir)
    metadata_path = run_dir / 'metadata.json'
    
    if not metadata_path.exists():
        print(f"No metadata.json found at {metadata_path}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    index_data = []
    components = metadata.get('components', {})
    
    for component_id, component_info in components.items():
        if component_info.get('component_type') != 'strategy':
            continue
        
        # Extract parameters
        params = component_info.get('parameters', {})
        strategy_type = component_info.get('strategy_type', 'unknown')
        
        # Clean parameters (remove internal ones)
        clean_params = {k: v for k, v in params.items() if not k.startswith('_')}
        
        # If no parameters found, try to extract from component_id
        if not clean_params and 'bollinger_bands' in component_id:
            import re
            # Pattern: symbol_timeframe_bollinger_bands_period_stddev
            match = re.search(r'bollinger_bands_(\d+)_(\d+)$', component_id)
            if match:
                clean_params = {
                    'period': int(match.group(1)),
                    'std_dev': int(match.group(2)) / 10.0  # Encoded as std_dev * 10
                }
        
        # Compute unique hash
        hash_config = {
            'type': strategy_type,
            'parameters': clean_params
        }
        config_str = json.dumps(hash_config, sort_keys=True, separators=(',', ':'))
        strategy_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
        
        # Build index entry
        entry = {
            'strategy_id': component_id,
            'strategy_hash': strategy_hash,
            'strategy_type': strategy_type,
            'symbol': 'SPY',  # Extract from component_id if needed
            'timeframe': '5m',
            'trace_path': component_info.get('signal_file_path', ''),
        }
        
        # Add parameters as direct columns
        for param_name, param_value in clean_params.items():
            entry[param_name] = param_value
        
        index_data.append(entry)
    
    if index_data:
        # Create DataFrame
        df = pd.DataFrame(index_data)
        
        print(f"Generated strategy index with {len(df)} strategies")
        print(f"Columns: {list(df.columns)}")
        print(f"Unique hashes: {df['strategy_hash'].nunique()}")
        
        # Save it
        index_path = run_dir / 'strategy_index.parquet'
        df.to_parquet(index_path, engine='pyarrow', index=False)
        print(f"Saved to: {index_path}")
        
        # Show sample
        print("\nSample entries:")
        print(df[['strategy_id', 'strategy_hash', 'period', 'std_dev']].head(10))
    else:
        print("No strategies found in metadata")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = "config/bollinger/results/20250624_214106"
    
    print(f"Generating strategy index for: {run_dir}")
    generate_strategy_index(run_dir)