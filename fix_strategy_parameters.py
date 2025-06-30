#!/usr/bin/env python3
"""
Fix strategy parameters in existing runs by extracting from strategy_id
"""
import json
import pandas as pd
import hashlib
from pathlib import Path
import re

def extract_parameters_from_id(strategy_id, strategy_type):
    """Extract parameters from strategy_id string"""
    params = {}
    
    if strategy_type == 'bollinger_bands':
        # Pattern: bollinger_bands_PERIOD_STDDEV
        match = re.search(r'bollinger_bands_(\d+)_(\d+)$', strategy_id)
        if match:
            params['period'] = int(match.group(1))
            params['std_dev'] = int(match.group(2)) / 10.0  # Encoded as std_dev * 10
    
    elif 'ma_crossover' in strategy_type:
        # Pattern: ma_crossover_FAST_SLOW
        match = re.search(r'ma_crossover_(\d+)_(\d+)', strategy_id)
        if match:
            params['fast_period'] = int(match.group(1))
            params['slow_period'] = int(match.group(2))
    
    elif 'rsi' in strategy_type:
        # Pattern: rsi_PERIOD_OVERSOLD_OVERBOUGHT
        match = re.search(r'rsi_(\d+)_(\d+)_(\d+)', strategy_id)
        if match:
            params['period'] = int(match.group(1))
            params['oversold'] = int(match.group(2))
            params['overbought'] = int(match.group(3))
    
    # Add more patterns as needed
    
    return params

def fix_metadata_json(run_dir):
    """Fix parameters in metadata.json"""
    metadata_path = Path(run_dir) / 'metadata.json'
    
    if not metadata_path.exists():
        print(f"❌ No metadata.json found at {metadata_path}")
        return False
    
    # Backup original
    backup_path = Path(run_dir) / 'metadata.json.backup'
    if not backup_path.exists():
        import shutil
        shutil.copy(metadata_path, backup_path)
        print(f"✅ Created backup at {backup_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    fixed_count = 0
    components = metadata.get('components', {})
    
    for comp_id, comp_info in components.items():
        if comp_info.get('type') == 'strategy':
            current_params = comp_info.get('parameters', {})
            
            # Check if parameters need fixing
            if len(current_params) <= 1 or (len(current_params) == 1 and '_strategy_type' in current_params):
                # Extract parameters from ID
                strategy_type = comp_info.get('strategy_type', '')
                extracted_params = extract_parameters_from_id(comp_id, strategy_type)
                
                if extracted_params:
                    # Preserve _strategy_type if it exists
                    if '_strategy_type' in current_params:
                        extracted_params['_strategy_type'] = current_params['_strategy_type']
                    
                    comp_info['parameters'] = extracted_params
                    fixed_count += 1
                    print(f"✅ Fixed {comp_id}: {extracted_params}")
    
    if fixed_count > 0:
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\n✅ Updated metadata.json - fixed {fixed_count} strategies")
    else:
        print(f"\n✅ No fixes needed in metadata.json")
    
    return True

def regenerate_strategy_index(run_dir):
    """Regenerate strategy_index.parquet with proper parameters"""
    metadata_path = Path(run_dir) / 'metadata.json'
    
    if not metadata_path.exists():
        print(f"❌ No metadata.json found")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    index_data = []
    components = metadata.get('components', {})
    
    for component_id, component_info in components.items():
        if component_info.get('type') != 'strategy':
            continue
        
        # Get parameters
        params = component_info.get('parameters', {})
        strategy_type = component_info.get('strategy_type', 'unknown')
        
        # Clean parameters (remove internal ones)
        clean_params = {k: v for k, v in params.items() if not k.startswith('_')}
        
        # Compute unique hash based on actual parameters
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
            'symbol': component_info.get('symbol', 'SPY'),
            'timeframe': component_info.get('timeframe', '5m'),
        }
        
        # Add trace path if available
        signal_file = component_info.get('signal_file_path', '')
        if signal_file:
            entry['trace_path'] = signal_file
        else:
            # Construct expected path
            entry['trace_path'] = f"traces/signals/{strategy_type}/{component_id}.parquet"
        
        # Add parameters as direct columns
        entry.update(clean_params)
        
        index_data.append(entry)
    
    if index_data:
        # Create DataFrame
        df = pd.DataFrame(index_data)
        
        print(f"\n📊 Strategy index summary:")
        print(f"   Total strategies: {len(df)}")
        print(f"   Unique hashes: {df['strategy_hash'].nunique()}")
        print(f"   Columns: {list(df.columns)}")
        
        # Save
        index_path = Path(run_dir) / 'strategy_index.parquet'
        df.to_parquet(index_path, engine='pyarrow', index=False)
        print(f"\n✅ Saved strategy index to {index_path}")
        
        # Show sample
        param_cols = [c for c in df.columns if c not in ['strategy_id', 'strategy_hash', 'strategy_type', 'symbol', 'timeframe', 'trace_path']]
        if param_cols:
            print(f"\n📊 Sample with parameters:")
            print(df[['strategy_id', 'strategy_hash'] + param_cols].head())
    else:
        print("❌ No strategies found")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = "config/bollinger/results/20250624_214106"
    
    print(f"Fixing run directory: {run_dir}\n")
    
    # First fix metadata.json
    if fix_metadata_json(run_dir):
        # Then regenerate strategy index
        regenerate_strategy_index(run_dir)