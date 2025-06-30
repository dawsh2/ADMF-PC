#!/usr/bin/env python3
"""Debug the latest order IDs to see if unique IDs are being generated."""

import pandas as pd
import json
from pathlib import Path

# Find the latest results directory
results_base = Path('config/bollinger/results')
latest_dir = max(results_base.glob('20250627_*'), key=lambda p: p.name)
traces_dir = latest_dir / 'traces'

print(f"Checking order IDs in: {latest_dir}")
print("=" * 80)

# Load orders
order_files = list(traces_dir.glob('portfolio/orders/*.parquet'))
if order_files:
    orders = pd.read_parquet(order_files[0])
    print(f"Total orders: {len(orders)}")
    
    # Check strategy IDs
    print(f"\nUnique strategy IDs: {orders['strat'].nunique()}")
    print("\nAll strategy IDs:")
    for i, strat_id in enumerate(orders['strat']):
        print(f"  Order {i}: {strat_id}")
        if i > 10:  # Show first 10
            print("  ...")
            break
    
    # Check if they're all the same
    if orders['strat'].nunique() == 1:
        print("\n❌ PROBLEM: All orders have the same strategy ID!")
        print("This means sparse storage is treating them as one source")
        print("and only storing changes in direction.")
    else:
        print("\n✅ Good: Orders have unique strategy IDs")
    
    # Check metadata
    print("\nChecking metadata:")
    for i in range(min(5, len(orders))):
        metadata = orders.iloc[i]['metadata']
        if metadata:
            try:
                data = json.loads(metadata) if isinstance(metadata, str) else metadata
                order_id = data.get('order_id', 'NO_ORDER_ID')
                print(f"  Order {i}: order_id = {order_id}")
            except:
                print(f"  Order {i}: Unable to parse metadata")
        else:
            print(f"  Order {i}: No metadata")

# Check MultiStrategyTracer code
print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("The issue is that all orders still have the same strategy ID.")
print("This suggests the fix in MultiStrategyTracer isn't being applied.")
print("\nPossible causes:")
print("1. The code wasn't saved before running")
print("2. Python cached the old version") 
print("3. The order_id in payload is the same for all orders")
print("\nNext step: Check the MultiStrategyTracer code to ensure the fix is there")