#!/usr/bin/env python3
"""Debug order tracing to understand why most orders have no metadata."""

import pandas as pd
import json
from pathlib import Path

# Load traces
traces_dir = Path('config/bollinger/results/20250627_095819/traces')

# Load signals first to understand the pattern
signals = pd.read_parquet(list(traces_dir.glob('signals/*/*.parquet'))[0])
print(f"Total signals: {len(signals)}")

# Check signal metadata
print("\nFirst 5 signal records:")
for i in range(min(5, len(signals))):
    print(f"Signal {i}: idx={signals.iloc[i]['idx']}, val={signals.iloc[i]['val']}, metadata='{signals.iloc[i].get('metadata', 'N/A')}'")

# Load orders
orders = pd.read_parquet(traces_dir / 'portfolio/orders/portfolio_orders.parquet')
print(f"\nTotal orders: {len(orders)}")

# Check what columns orders have
print(f"\nOrder columns: {list(orders.columns)}")

# Show first few orders with all columns
print("\nFirst 5 orders (all columns):")
print(orders.head())

# Check if orders are stored sparsely (only changes)
print("\nOrder indices (first 20):")
print(orders['idx'].head(20).tolist())

# Compare with signal changes
signal_changes = signals[signals['val'] != 0]
print(f"\nNon-zero signals: {len(signal_changes)}")
print("First 20 non-zero signal indices:")
print(signal_changes['idx'].head(20).tolist())

# Check if the issue is sparse storage
print("\nChecking sparse storage pattern...")
print(f"Average gap between orders: {orders['idx'].diff().mean():.1f} bars")
print(f"Orders per 100 bars: {len(orders) / (orders['idx'].max() - orders['idx'].min()) * 100:.1f}")