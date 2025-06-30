#!/usr/bin/env python3
"""Debug why dense storage isn't working."""

import pandas as pd
from pathlib import Path

# Find the latest results directory
results_base = Path('config/bollinger/results')
latest_dir = max(results_base.glob('20250627_*'), key=lambda p: p.name)
traces_dir = latest_dir / 'traces'

print(f"Checking traces in: {latest_dir}")
print("=" * 80)

# Check orders
orders_path = traces_dir / 'portfolio' / 'orders' / 'portfolio_orders.parquet'
if orders_path.exists():
    orders = pd.read_parquet(orders_path)
    print(f"\nOrders file analysis:")
    print(f"  Total records: {len(orders)}")
    print(f"  Columns: {list(orders.columns)}")
    
    # Check if this is sparse or dense
    if 'strat' in orders.columns:
        print(f"  Unique strategy IDs: {orders['strat'].nunique()}")
        print("  ❌ This appears to be sparse storage (has 'strat' column)")
    
    if 'idx' in orders.columns and len(orders) > 1:
        gaps = orders['idx'].diff().dropna()
        avg_gap = gaps.mean()
        print(f"  Average gap between records: {avg_gap:.1f} bars")
        if avg_gap > 5:
            print("  ❌ Large gaps suggest sparse storage")
        else:
            print("  ✅ Small gaps suggest dense storage")

# Check execution fills
fills_path = traces_dir / 'execution' / 'fills' / 'execution_fills.parquet'
if fills_path.exists():
    fills = pd.read_parquet(fills_path)
    print(f"\nFills file analysis:")
    print(f"  Total records: {len(fills)}")
    print(f"  Columns: {list(fills.columns)}")
    
    if 'strat' in fills.columns:
        print(f"  Unique strategy IDs: {fills['strat'].nunique()}")
        print("  ❌ This appears to be sparse storage (has 'strat' column)")

# The problem is clear: we're still using sparse storage
print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("The MultiStrategyTracer is still using the old sparse storage code path.")
print("Even though we created DenseEventStorage, it's not being used.")
print("\nThe fix:")
print("1. The _get_or_create_storage method is still being called")
print("2. This creates TemporalSparseStorage instances")
print("3. We need to ensure orders/fills/positions use DenseEventStorage")