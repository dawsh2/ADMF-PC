#!/usr/bin/env python3
"""Check the format of dense storage files."""

import pandas as pd
from pathlib import Path

# Find the latest results directory
results_base = Path('config/bollinger/results')
latest_dir = max(results_base.glob('20250627_*'), key=lambda p: p.name)
traces_dir = latest_dir / 'traces'

print(f"Checking dense storage format in: {latest_dir}")
print("=" * 80)

# Check orders
order_files = list(traces_dir.glob('portfolio/orders/*.parquet'))
if order_files:
    orders = pd.read_parquet(order_files[0])
    print(f"\nOrders columns: {list(orders.columns)}")
    print(f"Total orders: {len(orders)}")
    
    # Show first few records
    print("\nFirst 3 orders:")
    for i in range(min(3, len(orders))):
        print(f"\nOrder {i}:")
        for col in orders.columns:
            print(f"  {col}: {orders.iloc[i][col]}")

# Check fills
fill_files = list(traces_dir.glob('execution/fills/*.parquet'))
if fill_files:
    fills = pd.read_parquet(fill_files[0])
    print(f"\n\nFills columns: {list(fills.columns)}")
    print(f"Total fills: {len(fills)}")

# Check position files
print("\n\nLooking for position files:")
print(f"Position open files: {list(traces_dir.glob('portfolio/positions_open/*.parquet'))}")
print(f"Position close files: {list(traces_dir.glob('portfolio/positions_close/*.parquet'))}")

# Check all portfolio subdirectories
print("\nAll portfolio subdirectories:")
for p in (traces_dir / 'portfolio').iterdir():
    if p.is_dir():
        print(f"  {p.name}/: {list(p.glob('*.parquet'))}")