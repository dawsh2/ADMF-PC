#!/usr/bin/env python3
"""Debug script to analyze position tracking issues."""

import pandas as pd
import json
from pathlib import Path

# Load traces
traces_dir = Path('config/bollinger/results/20250627_095819/traces')

# Load orders
orders = pd.read_parquet(traces_dir / 'portfolio/orders/portfolio_orders.parquet')
print(f"Total orders: {len(orders)}")

# Load fills
fills = pd.read_parquet(traces_dir / 'execution/fills/execution_fills.parquet')
print(f"Total fills: {len(fills)}")

# Parse metadata
orders['order_data'] = orders['metadata'].apply(lambda x: json.loads(x) if x else {})
fills['fill_data'] = fills['metadata'].apply(lambda x: json.loads(x) if x else {})

# Check metadata directly
print("\nRaw metadata sample:")
for i in range(5):
    print(f"Order {i}: metadata = {orders.iloc[i]['metadata']}")

# Extract order details - check if metadata is dict already
if len(orders) > 0 and isinstance(orders.iloc[0]['metadata'], dict):
    # Metadata is already a dict
    orders['order_id'] = orders['metadata'].apply(lambda x: x.get('order_id') if x else None)
    orders['side'] = orders['metadata'].apply(lambda x: x.get('side') if x else None)
    orders['quantity'] = orders['metadata'].apply(lambda x: x.get('quantity') if x else None)
    orders['strategy_id'] = orders['metadata'].apply(lambda x: x.get('strategy_id') if x else None)
else:
    # Use parsed data
    orders['order_id'] = orders['order_data'].apply(lambda x: x.get('order_id'))
    orders['side'] = orders['order_data'].apply(lambda x: x.get('side'))
    orders['quantity'] = orders['order_data'].apply(lambda x: x.get('quantity'))
    orders['strategy_id'] = orders['order_data'].apply(lambda x: x.get('strategy_id'))

# Extract fill details
fills['order_id'] = fills['fill_data'].apply(lambda x: x.get('order_id'))
fills['side'] = fills['fill_data'].apply(lambda x: x.get('side'))
fills['quantity'] = fills['fill_data'].apply(lambda x: x.get('quantity'))

print("\nFirst 10 orders:")
print(orders[['idx', 'side', 'quantity', 'strategy_id']].head(10))

print("\nFirst 10 fills:")
print(fills[['idx', 'side', 'quantity']].head(10))

# Check for alternating buy/sell pattern
print("\nOrder side distribution:")
print(orders['side'].value_counts())

# Check if orders are immediately followed by opposite orders
order_sides = orders.sort_values('idx')[['idx', 'side']].reset_index(drop=True)
consecutive_opposites = 0
for i in range(1, len(order_sides)):
    current_side = order_sides.iloc[i]['side']
    prev_side = order_sides.iloc[i-1]['side']
    if (current_side == 'BUY' and prev_side == 'SELL') or (current_side == 'SELL' and prev_side == 'BUY'):
        consecutive_opposites += 1

print(f"\nConsecutive opposite orders: {consecutive_opposites} out of {len(orders)-1} pairs")
print(f"Percentage: {consecutive_opposites / (len(orders)-1) * 100:.1f}%")

# Check time between orders
order_gaps = orders.sort_values('idx')['idx'].diff()
print(f"\nTime between orders:")
print(f"  Mean: {order_gaps.mean():.1f} bars")
print(f"  Median: {order_gaps.median():.1f} bars")
print(f"  Min: {order_gaps.min():.0f} bars")

# Check for flattening pattern
flat_orders = orders[orders['side'].str.contains('FLAT', na=False)]
print(f"\nFlat orders: {len(flat_orders)}")

# Load signals to understand the pattern
signals = pd.read_parquet(list(traces_dir.glob('signals/*/*.parquet'))[0])
print(f"\nTotal signals: {len(signals)}")
print(f"Signal value distribution:")
print(signals['val'].value_counts())

# Check signal changes
signal_changes = signals['val'].diff() != 0
print(f"\nSignal changes: {signal_changes.sum()} out of {len(signals)} signals")
print(f"Percentage: {signal_changes.sum() / len(signals) * 100:.1f}%")