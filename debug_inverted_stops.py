"""Debug why some stop losses have inverted returns."""

import pandas as pd
from pathlib import Path

# Load both orders and fills data
results_path = Path("config/bollinger/results/latest")
fills_path = results_path / "traces/execution/fills/execution_fills.parquet"
orders_path = results_path / "traces/execution/orders/execution_orders.parquet"

fills_df = pd.read_parquet(fills_path)
orders_df = pd.read_parquet(orders_path)

print("=== DEBUGGING INVERTED STOP LOSSES ===\n")

# Find stop loss orders with inverted returns
inverted_cases = []

for idx, row in fills_df.iterrows():
    metadata = row['metadata']
    if isinstance(metadata, dict):
        nested = metadata.get('metadata', {})
        if isinstance(nested, dict) and nested.get('exit_type') == 'stop_loss':
            order_id = metadata.get('order_id')
            
            # Find the previous entry
            for j in range(idx-1, -1, -1):
                prev_row = fills_df.iloc[j]
                prev_meta = prev_row['metadata']
                if isinstance(prev_meta, dict):
                    prev_nested = prev_meta.get('metadata', {})
                    if not prev_nested.get('exit_type') and prev_meta.get('side') != metadata.get('side'):
                        entry_price = float(prev_meta.get('price', 0))
                        exit_price = float(metadata.get('price', 0))
                        if entry_price > 0:
                            return_pct = (exit_price - entry_price) / entry_price * 100
                            
                            # Check if it's inverted (positive instead of negative)
                            if abs(return_pct - 0.075) < 0.01:  # ~0.075% positive
                                # Find the corresponding order
                                order_row = orders_df[orders_df['order_id'] == order_id].iloc[0] if order_id else None
                                
                                inverted_cases.append({
                                    'order_id': order_id,
                                    'entry_price': entry_price,
                                    'exit_fill_price': exit_price,
                                    'order_price': float(order_row['price']) if order_row is not None else None,
                                    'return_pct': return_pct,
                                    'exit_reason': nested.get('exit_reason', ''),
                                    'fill_metadata': metadata,
                                    'order_metadata': order_row.get('metadata') if order_row is not None else None
                                })
                            break

print(f"Found {len(inverted_cases)} inverted stop losses\n")

# Analyze the first few cases
for i, case in enumerate(inverted_cases[:3]):
    print(f"\n=== CASE {i+1}: {case['order_id']} ===")
    print(f"Entry price: ${case['entry_price']:.4f}")
    print(f"Exit fill price: ${case['exit_fill_price']:.4f}")
    print(f"Return: {case['return_pct']:.4f}% (POSITIVE instead of negative!)")
    
    if case['order_price']:
        print(f"\nOrder price: ${case['order_price']:.4f}")
        # Calculate what the return should have been
        expected_return = (case['order_price'] - case['entry_price']) / case['entry_price'] * 100
        print(f"Expected return if order price was used: {expected_return:.4f}%")
        
        # Check if order price would give correct -0.075%
        if abs(expected_return - (-0.075)) < 0.001:
            print("✅ Order price would have given correct -0.075% return")
        else:
            print("❌ Order price is also incorrect")
    
    print(f"\nExit reason: {case['exit_reason']}")
    
    # Check if order had exit_type in metadata
    if case['order_metadata'] and isinstance(case['order_metadata'], dict):
        order_exit_type = case['order_metadata'].get('exit_type')
        print(f"Order exit_type: {order_exit_type}")
        
        # Check nested metadata
        nested_order_meta = case['order_metadata'].get('metadata', {})
        if isinstance(nested_order_meta, dict):
            print(f"Order nested exit_type: {nested_order_meta.get('exit_type')}")

# Summary
print("\n=== SUMMARY ===")
correct_order_prices = sum(1 for c in inverted_cases if c['order_price'] and abs(((c['order_price'] - c['entry_price']) / c['entry_price'] * 100) - (-0.075)) < 0.001)
print(f"Cases where order price is correct: {correct_order_prices}/{len(inverted_cases)}")
print("\nThis suggests the order has the correct stop price, but the fill is using a different price!")