"""Trace the order flow to see where price is lost."""

import pandas as pd

# Load the fill data
fills_path = 'config/bollinger/results/latest/traces/execution/fills/execution_fills.parquet'

try:
    fills_df = pd.read_parquet(fills_path)
    
    # Find a stop loss fill with its entry
    stop_fills = []
    
    for idx, row in fills_df.iterrows():
        metadata = row['metadata']
        if isinstance(metadata, dict):
            nested = metadata.get('metadata', {})
            if isinstance(nested, dict) and nested.get('exit_type') == 'stop_loss':
                # Found a stop loss
                order_id = metadata.get('order_id', '')
                price = float(metadata.get('price', 0))
                
                # Find the entry order
                entry_price = None
                for prev_idx in range(idx):
                    prev_row = fills_df.iloc[prev_idx]
                    prev_meta = prev_row['metadata']
                    if isinstance(prev_meta, dict):
                        prev_nested = prev_meta.get('metadata', {})
                        if prev_nested.get('strategy_id') == nested.get('strategy_id').replace('_exit', ''):
                            # Check if this could be the entry
                            if prev_meta.get('side') != metadata.get('side'):
                                entry_price = float(prev_meta.get('price', 0))
                                break
                
                if entry_price:
                    actual_return = (price - entry_price) / entry_price
                    expected_return = -0.00075  # -0.075%
                    expected_price = entry_price * (1 + expected_return)
                    
                    stop_fills.append({
                        'order_id': order_id,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'expected_price': expected_price,
                        'actual_return': actual_return,
                        'expected_return': expected_return,
                        'price_diff': abs(price - expected_price),
                        'exit_reason': nested.get('exit_reason', '')
                    })
    
    # Analyze results
    print(f"Found {len(stop_fills)} stop loss fills with matching entries\n")
    
    if stop_fills:
        # Check how many exited at exact stop price
        exact_stops = sum(1 for s in stop_fills if s['price_diff'] < 0.01)
        print(f"Stops at exact price (within $0.01): {exact_stops} ({exact_stops/len(stop_fills)*100:.1f}%)")
        
        # Show examples
        print("\nExample stop losses:")
        for i, stop in enumerate(stop_fills[:5]):
            print(f"\nStop {i+1}:")
            print(f"  Entry price:    ${stop['entry_price']:.4f}")
            print(f"  Expected exit:  ${stop['expected_price']:.4f} (-0.075%)")
            print(f"  Actual exit:    ${stop['exit_price']:.4f}")
            print(f"  Actual return:  {stop['actual_return']:.4%}")
            print(f"  Price diff:     ${stop['price_diff']:.4f}")
            print(f"  Exit reason:    {stop['exit_reason']}")
    
    # Check order metadata for price field
    print("\n\nChecking if orders have price in metadata...")
    price_count = 0
    for idx, row in fills_df.iterrows()[:10]:
        metadata = row['metadata']
        if isinstance(metadata, dict):
            # The 'price' at this level is the fill price
            # We need to check if there's an order_price field
            nested = metadata.get('metadata', {})
            print(f"\nFill {idx}: {nested.get('exit_type', 'entry')}")
            print(f"  Fill price: {metadata.get('price')}")
            print(f"  Order metadata: {nested}")
            
            # Check if we see any price-related fields in the order
            if 'order_price' in nested or 'target_price' in nested or 'stop_price' in nested:
                price_count += 1
                print("  ✓ Found price-related field in order metadata")
    
    if price_count == 0:
        print("\n⚠️  No price fields found in order metadata!")
        print("This suggests the order price is not being passed through the event system.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()