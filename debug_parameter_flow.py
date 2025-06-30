#!/usr/bin/env python3
"""
Debug script to trace parameter flow through the system
"""
import json
from pathlib import Path

def check_metadata(run_dir):
    """Check metadata.json for parameters"""
    metadata_path = Path(run_dir) / 'metadata.json'
    
    if not metadata_path.exists():
        print(f"❌ No metadata.json found at {metadata_path}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Found metadata.json")
    print(f"   Total components: {len(metadata.get('components', {}))}")
    
    # Check each component
    components = metadata.get('components', {})
    for comp_id, comp_info in components.items():
        if comp_info.get('type') == 'strategy':
            params = comp_info.get('parameters', {})
            print(f"\n📊 Strategy: {comp_id}")
            print(f"   Strategy type: {comp_info.get('strategy_type', 'unknown')}")
            print(f"   Parameters: {params}")
            
            # Check if parameters only have _strategy_type
            if len(params) == 1 and '_strategy_type' in params:
                print(f"   ⚠️  WARNING: Only _strategy_type found, missing actual parameters!")
            elif not params:
                print(f"   ⚠️  WARNING: No parameters found!")
            else:
                # Show actual parameters
                actual_params = {k: v for k, v in params.items() if not k.startswith('_')}
                print(f"   Actual parameters: {actual_params}")

def check_strategy_index(run_dir):
    """Check strategy_index.parquet if it exists"""
    index_path = Path(run_dir) / 'strategy_index.parquet'
    
    if not index_path.exists():
        print(f"\n❌ No strategy_index.parquet found")
        return
    
    try:
        import pandas as pd
        df = pd.read_parquet(index_path)
        print(f"\n✅ Found strategy_index.parquet")
        print(f"   Strategies: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Check for parameter columns
        param_cols = [c for c in df.columns if c not in ['strategy_id', 'strategy_hash', 'strategy_type', 'symbol', 'timeframe', 'trace_path', 'constraints']]
        if param_cols:
            print(f"   Parameter columns: {param_cols}")
            
            # Show unique hashes
            print(f"\n   Unique strategy hashes: {df['strategy_hash'].nunique()}")
            if df['strategy_hash'].nunique() == 1:
                print(f"   ⚠️  WARNING: All strategies have the same hash: {df['strategy_hash'].iloc[0]}")
        else:
            print(f"   ⚠️  WARNING: No parameter columns found!")
            
        # Show sample
        print("\n   Sample entries:")
        print(df[['strategy_id', 'strategy_hash'] + param_cols[:3]].head())
        
    except Exception as e:
        print(f"   Error reading parquet: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        # Default to the problematic run
        run_dir = "config/bollinger/results/20250624_214106"
    
    print(f"Checking run directory: {run_dir}")
    check_metadata(run_dir)
    check_strategy_index(run_dir)