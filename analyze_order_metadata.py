#!/usr/bin/env python3
"""Analyze order metadata to understand trading behavior."""

import pandas as pd
import json
from pathlib import Path

# Find the latest results directory
results_base = Path('config/bollinger/results')
latest_dir = max(results_base.glob('20250627_*'), key=lambda p: p.name)
traces_dir = latest_dir / 'traces'

print(f"Analyzing order metadata from: {latest_dir}")
print("=" * 80)

# Load orders
order_files = list(traces_dir.glob('portfolio/orders/*.parquet'))
if order_files:
    orders = pd.read_parquet(order_files[0])
    print(f"Total orders: {len(orders)}")
    
    # Parse metadata
    order_data = []
    for i, row in orders.iterrows():
        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
        order_data.append({
            'idx': row['idx'],
            'side': metadata.get('side'),
            'quantity': metadata.get('quantity'),
            'order_id': metadata.get('order_id'),
            'status': metadata.get('status'),
            'strategy_id': metadata.get('metadata', {}).get('strategy_id')
        })
    
    orders_df = pd.DataFrame(order_data)
    
    # Analyze order patterns
    print(f"\nOrder sides:")
    print(orders_df['side'].value_counts())
    
    print(f"\nOrder status:")
    print(orders_df['status'].value_counts())
    
    # Check for alternating buy/sell
    sides = orders_df['side'].tolist()
    print(f"\nFirst 20 order sides: {sides[:20]}")
    
    # Check if fills happened
    fill_files = list(traces_dir.glob('execution/fills/*.parquet'))
    if fill_files:
        fills = pd.read_parquet(fill_files[0])
        print(f"\nTotal fills: {len(fills)}")
        
        # Compare orders to fills
        print(f"Orders-to-fills ratio: {len(fills)/len(orders)*100:.1f}%")
    
    # Check for unique order IDs
    unique_order_ids = orders_df['order_id'].nunique()
    print(f"\nUnique order IDs: {unique_order_ids}")
    if unique_order_ids == 1:
        print("❌ All orders have the same ID - this may prevent proper position tracking")
    
    # Check bar gaps
    gaps = orders_df['idx'].diff().dropna()
    print(f"\nBar gaps between orders:")
    print(f"  Mean: {gaps.mean():.1f}")
    print(f"  Min: {gaps.min():.0f}")
    print(f"  Max: {gaps.max():.0f}")

# Check if we should see position events based on this
print("\n" + "=" * 80)
print("ANALYSIS:")
print("If all orders are 'buy' with no sells, the portfolio would stay in one position")
print("and we'd only see one POSITION_OPEN event (no closes).")
print("If orders alternate buy/sell, we should see many position open/close events.")