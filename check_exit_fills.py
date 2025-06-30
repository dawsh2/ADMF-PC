"""Check for exit fills in the data."""

import pandas as pd

# Load the fill data
fills_path = 'config/bollinger/results/latest/traces/execution/fills/execution_fills.parquet'

try:
    fills_df = pd.read_parquet(fills_path)
    print(f"Found {len(fills_df)} fills\n")
    
    # Look for exit fills
    exit_count = 0
    stop_loss_count = 0
    take_profit_count = 0
    
    for idx, row in fills_df.iterrows():
        metadata = row['metadata']
        if isinstance(metadata, dict):
            # Check nested metadata
            nested_metadata = metadata.get('metadata', {})
            if isinstance(nested_metadata, dict):
                exit_type = nested_metadata.get('exit_type')
                if exit_type:
                    exit_count += 1
                    if exit_type == 'stop_loss':
                        stop_loss_count += 1
                        if stop_loss_count <= 3:
                            print(f"Stop loss example {stop_loss_count}:")
                            print(f"  Full metadata: {metadata}")
                            print()
                    elif exit_type == 'take_profit':
                        take_profit_count += 1
                        if take_profit_count <= 3:
                            print(f"Take profit example {take_profit_count}:")
                            print(f"  Full metadata: {metadata}")
                            print()
    
    print(f"\nSummary:")
    print(f"Total fills: {len(fills_df)}")
    print(f"Exit fills: {exit_count}")
    print(f"Stop loss exits: {stop_loss_count}")
    print(f"Take profit exits: {take_profit_count}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()