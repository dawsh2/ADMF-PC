"""Check fills to understand exit prices."""

import pandas as pd
import pyarrow.parquet as pq

# Load the fill data
fills_path = 'config/bollinger/results/latest/traces/execution/fills/execution_fills.parquet'

try:
    fills_df = pd.read_parquet(fills_path)
    print(f"Found {len(fills_df)} fills")
    print(f"\nColumns: {list(fills_df.columns)}")
    
    # Check fills with exit metadata
    if 'metadata' in fills_df.columns:
        exit_fills = fills_df[fills_df['metadata'].apply(lambda x: 'exit_type' in str(x))]
        print(f"\nFound {len(exit_fills)} exit fills")
        
        # Check stop loss fills
        stop_loss_fills = exit_fills[exit_fills['metadata'].apply(lambda x: 'stop_loss' in str(x))]
        take_profit_fills = exit_fills[exit_fills['metadata'].apply(lambda x: 'take_profit' in str(x))]
        
        print(f"\nStop loss fills: {len(stop_loss_fills)}")
        print(f"Take profit fills: {len(take_profit_fills)}")
        
        if len(stop_loss_fills) > 0:
            # Analyze stop loss exits
            print("\n=== STOP LOSS ANALYSIS ===")
            for idx in range(min(5, len(stop_loss_fills))):
                row = stop_loss_fills.iloc[idx]
                metadata = row['metadata'] if isinstance(row['metadata'], dict) else {}
                entry_price = metadata.get('entry_price', 0)
                exit_price = float(row['price'])
                
                if entry_price and entry_price > 0:
                    pct_change = (exit_price - entry_price) / entry_price
                    print(f"\nFill {idx+1}:")
                    print(f"  Entry: ${entry_price:.4f}")
                    print(f"  Exit:  ${exit_price:.4f}")
                    print(f"  Change: {pct_change:.4%}")
                    print(f"  Expected: -0.075%")
                    print(f"  Metadata: {metadata}")
        
        if len(take_profit_fills) > 0:
            # Analyze take profit exits
            print("\n=== TAKE PROFIT ANALYSIS ===")
            for idx in range(min(5, len(take_profit_fills))):
                row = take_profit_fills.iloc[idx]
                metadata = row['metadata'] if isinstance(row['metadata'], dict) else {}
                entry_price = metadata.get('entry_price', 0)
                exit_price = float(row['price'])
                
                if entry_price and entry_price > 0:
                    pct_change = (exit_price - entry_price) / entry_price
                    print(f"\nFill {idx+1}:")
                    print(f"  Entry: ${entry_price:.4f}")
                    print(f"  Exit:  ${exit_price:.4f}")
                    print(f"  Change: {pct_change:.4%}")
                    print(f"  Expected: +0.15%")
                    print(f"  Metadata: {metadata}")
    
    # Show first few fills
    print("\n=== SAMPLE FILLS ===")
    print(fills_df.head())
    
except Exception as e:
    print(f"Error loading fills: {e}")
    import traceback
    traceback.print_exc()