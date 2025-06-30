#!/usr/bin/env python3
"""
Fix strategy index by generating unique hashes based on actual parameters
"""
import pandas as pd
import json
import hashlib
from pathlib import Path

def compute_unique_strategy_hash(strategy_id, params):
    """Compute a unique hash for each strategy based on its parameters"""
    # Create a unique string from strategy_id and all parameters
    hash_data = {
        'strategy_id': strategy_id,
        'parameters': params
    }
    config_str = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]

def fix_strategy_index(run_dir):
    """Fix the strategy index by computing unique hashes"""
    run_dir = Path(run_dir)
    index_path = run_dir / 'strategy_index.parquet'
    
    if not index_path.exists():
        print(f"No strategy_index.parquet found at {index_path}")
        return
    
    # Load existing index
    df = pd.read_parquet(index_path)
    print(f"Loaded {len(df)} strategies")
    print(f"Current unique hashes: {df['strategy_hash'].nunique()}")
    
    # Recompute hashes based on actual parameters
    new_hashes = []
    for idx, row in df.iterrows():
        # Extract all parameter values
        params = {}
        for col in df.columns:
            if col.startswith('param_') and pd.notna(row[col]):
                params[col.replace('param_', '')] = row[col]
        
        # Include strategy_id to ensure uniqueness
        new_hash = compute_unique_strategy_hash(row['strategy_id'], params)
        new_hashes.append(new_hash)
    
    # Update the dataframe
    df['strategy_hash'] = new_hashes
    
    # Save the fixed index
    df.to_parquet(index_path, index=False)
    print(f"Fixed! New unique hashes: {df['strategy_hash'].nunique()}")
    print(f"Sample hashes: {df['strategy_hash'].head(5).tolist()}")
    
    # Also save a CSV for inspection
    df.to_csv(run_dir / 'strategy_index_fixed.csv', index=False)
    print(f"Also saved as CSV for inspection: strategy_index_fixed.csv")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = "config/bollinger/results/latest"
    
    print(f"Fixing strategy index in: {run_dir}")
    fix_strategy_index(run_dir)