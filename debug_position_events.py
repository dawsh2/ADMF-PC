#!/usr/bin/env python3
"""Debug why position events are missing."""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

# Find the latest results directory
results_base = Path('config/bollinger/results')
latest_dir = max(results_base.glob('20250627_*'), key=lambda p: p.name)
traces_dir = latest_dir / 'traces'

print(f"Debugging position events in: {latest_dir}")
print("=" * 80)

# Load orders
order_files = list(traces_dir.glob('portfolio/orders/*.parquet'))
if order_files:
    orders = pd.read_parquet(order_files[0])
    print(f"Total orders: {len(orders)}")
    
    # Analyze order pattern
    order_sides = []
    for i, row in orders.iterrows():
        metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
        side = metadata.get('side')
        order_sides.append(side)
    
    # Count consecutive orders of same side
    consecutive_counts = defaultdict(int)
    current_side = None
    current_count = 0
    
    for side in order_sides:
        if side == current_side:
            current_count += 1
        else:
            if current_side:
                consecutive_counts[current_count] += 1
            current_side = side
            current_count = 1
    
    if current_side:
        consecutive_counts[current_count] += 1
    
    print("\nConsecutive orders of same side:")
    for count, occurrences in sorted(consecutive_counts.items()):
        print(f"  {count} consecutive: {occurrences} times")
    
    # Check for position-closing pattern
    position_count = 0
    current_position = 0
    
    for side in order_sides:
        if side == 'buy':
            if current_position <= 0:  # Opening new long or closing short
                position_count += 1
            current_position = 100  # Assume fixed size
        elif side == 'sell':
            if current_position >= 0:  # Opening new short or closing long
                position_count += 1
            current_position = -100
    
    print(f"\nExpected position events (approx): {position_count}")
    print(f"Actual position events in metadata: 17")
    print(f"Ratio: {17/position_count*100:.1f}%")

# Check position files
print("\n" + "=" * 80)
print("Position file analysis:")

position_open_dir = traces_dir / 'portfolio' / 'positions_open'
position_close_dir = traces_dir / 'portfolio' / 'positions_close'

print(f"\nPositions open directory exists: {position_open_dir.exists()}")
print(f"Positions close directory exists: {position_close_dir.exists()}")

if position_open_dir.exists():
    open_files = list(position_open_dir.glob('*.parquet'))
    print(f"Position open files: {len(open_files)}")
    for f in open_files:
        print(f"  - {f.name}")

if position_close_dir.exists():
    close_files = list(position_close_dir.glob('*.parquet'))
    print(f"Position close files: {len(close_files)}")
    for f in close_files:
        print(f"  - {f.name}")

# Analysis
print("\n" + "=" * 80)
print("ANALYSIS:")
print("The portfolio is likely keeping positions open for long periods,")
print("resulting in fewer position events than expected.")
print("This could be due to:")
print("1. Risk management not closing positions")
print("2. Strategy signals staying in same direction")
print("3. Portfolio logic for position management")