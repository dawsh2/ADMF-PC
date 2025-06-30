#!/usr/bin/env python3
"""
Fix for creating strategy index from metadata.json when strategy_index.parquet is missing.
This extracts parameters properly and creates the index needed for analysis.
"""

import json
import pandas as pd
from pathlib import Path
import hashlib

def create_strategy_index_from_metadata(run_dir):
    """Create strategy index from metadata.json and trace files."""
    run_dir = Path(run_dir)
    metadata_path = run_dir / 'metadata.json'
    
    if not metadata_path.exists():
        print(f"No metadata.json found at {metadata_path}")
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract strategy information from components
    strategies = []
    components = metadata.get('components', {})
    
    for component_id, component_info in components.items():
        if component_info.get('component_type') == 'strategy':
            # Get parameters - they might be in different places
            params = component_info.get('parameters', {})
            
            # Extract actual parameters (remove internal fields)
            actual_params = {k: v for k, v in params.items() 
                           if not k.startswith('_') and k not in ['composite_strategies']}
            
            # If we have composite_strategies, extract params from there
            if 'composite_strategies' in params and params['composite_strategies']:
                # For ensemble with single strategy, use those params
                if len(params['composite_strategies']) == 1:
                    actual_params.update(params['composite_strategies'][0].get('params', {}))
            
            # Create a unique hash for this configuration
            param_str = json.dumps(actual_params, sort_keys=True)
            strategy_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            
            # Extract strategy type
            strategy_type = component_info.get('strategy_type', 'unknown')
            if strategy_type == 'ensemble' and 'composite_strategies' in params:
                # For ensembles, use the actual strategy type
                if params['composite_strategies']:
                    strategy_type = params['composite_strategies'][0].get('type', strategy_type)
            
            # Build strategy record
            strategy_record = {
                'strategy_hash': strategy_hash,
                'strategy_type': strategy_type,
                'component_id': component_id,
                'trace_path': component_info.get('signal_file_path', ''),
                'signal_changes': component_info.get('signal_changes', 0),
                'compression_ratio': component_info.get('compression_ratio', 0),
            }
            
            # Add parameters with param_ prefix for consistency
            for param_name, param_value in actual_params.items():
                strategy_record[f'param_{param_name}'] = param_value
            
            strategies.append(strategy_record)
    
    if strategies:
        df = pd.DataFrame(strategies)
        print(f"Created strategy index with {len(df)} strategies")
        print(f"Strategy types: {df['strategy_type'].value_counts().to_dict()}")
        print(f"Parameters found: {[col for col in df.columns if col.startswith('param_')]}")
        return df
    else:
        print("No strategies found in metadata")
        return None


def save_strategy_index(run_dir, strategy_index_df):
    """Save strategy index to parquet file."""
    if strategy_index_df is not None and not strategy_index_df.empty:
        output_path = Path(run_dir) / 'strategy_index.parquet'
        strategy_index_df.to_parquet(output_path)
        print(f"Saved strategy index to {output_path}")
        return output_path
    return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        # Default to latest bollinger results
        run_dir = "config/bollinger/results/20250623_062931"
    
    print(f"Creating strategy index for: {run_dir}")
    
    # Create the index
    strategy_index = create_strategy_index_from_metadata(run_dir)
    
    if strategy_index is not None:
        # Save it
        save_strategy_index(run_dir, strategy_index)
        
        # Show sample
        print("\nSample of strategy index:")
        print(strategy_index.head())
        
        # Show parameter distributions
        print("\nParameter distributions:")
        for col in strategy_index.columns:
            if col.startswith('param_'):
                param_name = col.replace('param_', '')
                unique_values = strategy_index[col].nunique()
                print(f"  {param_name}: {unique_values} unique values")
                if unique_values <= 10:
                    print(f"    Values: {sorted(strategy_index[col].unique())}")