"""Analyze exit prices to see if stops are working correctly."""

import pandas as pd
from collections import defaultdict

# Load the fill data
fills_path = 'config/bollinger/results/latest/traces/execution/fills/execution_fills.parquet'

try:
    fills_df = pd.read_parquet(fills_path)
    print(f"Found {len(fills_df)} fills")
    
    # Extract relevant data from metadata
    stop_loss_pcts = []
    take_profit_pcts = []
    
    # Track positions and their exits
    positions = {}  # order_id -> entry fill
    exits = []  # exit fills with calculated returns
    
    for idx, row in fills_df.iterrows():
        metadata = row['metadata']
        if not isinstance(metadata, dict):
            continue
            
        order_id = metadata.get('order_id', '')
        fill_metadata = metadata.get('metadata', {})
        exit_type = fill_metadata.get('exit_type')
        side = metadata.get('side', '').lower()
        price = float(metadata.get('price', 0))
        
        if exit_type in ['stop_loss', 'take_profit']:
            # This is an exit fill
            # Find the corresponding entry
            entry_order_id = order_id.replace('_exit', '').replace('_stop', '').replace('_take', '')
            
            # Look for entry in previous fills
            for prev_idx in range(idx):
                prev_row = fills_df.iloc[prev_idx]
                prev_metadata = prev_row['metadata']
                if isinstance(prev_metadata, dict):
                    prev_order_id = prev_metadata.get('order_id', '')
                    if prev_order_id == entry_order_id or prev_order_id in order_id:
                        # Found potential entry
                        entry_price = float(prev_metadata.get('price', 0))
                        if entry_price > 0:
                            # Calculate return
                            if side == 'sell':  # Exit sell means we were long
                                pct_return = (price - entry_price) / entry_price
                            else:  # Exit buy means we were short
                                pct_return = (entry_price - price) / entry_price
                            
                            exits.append({
                                'exit_type': exit_type,
                                'entry_price': entry_price,
                                'exit_price': price,
                                'pct_return': pct_return,
                                'side': side
                            })
                            
                            if exit_type == 'stop_loss':
                                stop_loss_pcts.append(pct_return)
                            elif exit_type == 'take_profit':
                                take_profit_pcts.append(pct_return)
                            break
    
    # Analyze results
    print(f"\n=== STOP LOSS ANALYSIS ===")
    print(f"Total stop loss exits: {len(stop_loss_pcts)}")
    if stop_loss_pcts:
        print(f"Average stop loss %: {sum(stop_loss_pcts)/len(stop_loss_pcts):.4%}")
        print(f"Min stop loss %: {min(stop_loss_pcts):.4%}")
        print(f"Max stop loss %: {max(stop_loss_pcts):.4%}")
        
        # Count how many are exactly -0.075%
        exact_stops = sum(1 for pct in stop_loss_pcts if abs(pct - (-0.00075)) < 0.0001)
        print(f"Stops at exactly -0.075%: {exact_stops} ({exact_stops/len(stop_loss_pcts)*100:.1f}%)")
        
        # Show distribution
        print("\nStop loss distribution:")
        ranges = [(-1, -0.01), (-0.01, -0.001), (-0.001, -0.0005), (-0.0005, 0), (0, 1)]
        for low, high in ranges:
            count = sum(1 for pct in stop_loss_pcts if low <= pct < high)
            if count > 0:
                print(f"  {low:.3%} to {high:.3%}: {count} ({count/len(stop_loss_pcts)*100:.1f}%)")
    
    print(f"\n=== TAKE PROFIT ANALYSIS ===")
    print(f"Total take profit exits: {len(take_profit_pcts)}")
    if take_profit_pcts:
        print(f"Average take profit %: {sum(take_profit_pcts)/len(take_profit_pcts):.4%}")
        print(f"Min take profit %: {min(take_profit_pcts):.4%}")
        print(f"Max take profit %: {max(take_profit_pcts):.4%}")
        
        # Count how many are exactly 0.15%
        exact_takes = sum(1 for pct in take_profit_pcts if abs(pct - 0.0015) < 0.0001)
        print(f"Takes at exactly 0.15%: {exact_takes} ({exact_takes/len(take_profit_pcts)*100:.1f}%)")
        
        # Show distribution
        print("\nTake profit distribution:")
        ranges = [(-1, 0), (0, 0.001), (0.001, 0.002), (0.002, 0.01), (0.01, 1)]
        for low, high in ranges:
            count = sum(1 for pct in take_profit_pcts if low <= pct < high)
            if count > 0:
                print(f"  {low:.3%} to {high:.3%}: {count} ({count/len(take_profit_pcts)*100:.1f}%)")
    
    # Show some example exits
    print("\n=== EXAMPLE EXITS ===")
    for i, exit in enumerate(exits[:5]):
        print(f"\nExit {i+1} ({exit['exit_type']}):")
        print(f"  Entry: ${exit['entry_price']:.4f}")
        print(f"  Exit:  ${exit['exit_price']:.4f}")
        print(f"  Return: {exit['pct_return']:.4%}")
        print(f"  Expected: {'-0.075%' if exit['exit_type'] == 'stop_loss' else '0.15%'}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()