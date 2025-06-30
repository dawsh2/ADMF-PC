#!/usr/bin/env python3
"""
Quick fix to extract parameters from strategy names and update strategy index
"""
import pandas as pd
from pathlib import Path
import re

def extract_params_from_name(strategy_id):
    """Extract parameters from strategy name like SPY_5m_bollinger_bands_10_15"""
    # Pattern: symbol_timeframe_strategy_type_period_stddev
    match = re.search(r'bollinger_bands_(\d+)_(\d+)$', strategy_id)
    if match:
        period = int(match.group(1))
        std_dev_encoded = int(match.group(2))
        # Decode std_dev (seems to be encoded as std_dev * 10)
        std_dev = std_dev_encoded / 10.0
        return {'period': period, 'std_dev': std_dev}
    return {}

def fix_strategy_index_params(run_dir):
    """Fix strategy index by extracting parameters from strategy names"""
    run_dir = Path(run_dir)
    index_path = run_dir / 'strategy_index.parquet'
    
    if not index_path.exists():
        print(f"No strategy_index.parquet found at {index_path}")
        return
    
    # Load index
    df = pd.read_parquet(index_path)
    print(f"Loaded {len(df)} strategies")
    
    # Extract parameters from strategy_id
    for idx, row in df.iterrows():
        params = extract_params_from_name(row['strategy_id'])
        if params:
            df.at[idx, 'param_period'] = params.get('period')
            df.at[idx, 'param_std_dev'] = params.get('std_dev')
    
    # Save updated index
    df.to_parquet(index_path, index=False)
    print(f"Updated strategy index with extracted parameters")
    
    # Show sample
    print("\nSample of updated parameters:")
    print(df[['strategy_id', 'param_period', 'param_std_dev']].head(10))
    
    # Also check performance_metrics.parquet if it exists
    perf_path = run_dir / 'performance_metrics.parquet'
    if perf_path.exists():
        perf_df = pd.read_parquet(perf_path)
        print(f"\nUpdating performance_metrics.parquet...")
        
        # Update performance metrics too
        for idx, row in perf_df.iterrows():
            if 'strategy_id' in row:
                params = extract_params_from_name(row['strategy_id'])
                if params:
                    perf_df.at[idx, 'param_period'] = params.get('period')
                    perf_df.at[idx, 'param_std_dev'] = params.get('std_dev')
        
        perf_df.to_parquet(perf_path, index=False)
        print("Updated performance metrics with parameters")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = "config/bollinger/results/latest"
    
    print(f"Fixing parameters in: {run_dir}")
    fix_strategy_index_params(run_dir)