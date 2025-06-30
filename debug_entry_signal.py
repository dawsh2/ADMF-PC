#!/usr/bin/env python3
"""Debug why entry_signal is not being stored."""

import pandas as pd
import json
from pathlib import Path

print("=== Debugging Entry Signal Storage ===\n")

# Let's trace through the order events to see what's happening
results_dir = Path("config/bollinger/results/latest")

# Check order events
order_file = results_dir / "traces/portfolio/orders/orders.parquet"

if order_file.exists():
    orders = pd.read_parquet(order_file)
    print(f"Total orders: {len(orders)}")
    
    # Check first few orders
    print("\nFirst 5 orders:")
    for i in range(min(5, len(orders))):
        order = orders.iloc[i]
        print(f"\n{i+1}. Order at bar {order['idx']}:")
        
        # Check metadata
        metadata = order.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                pass
        
        print(f"   Order type: {order.get('order_type', 'unknown')}")
        print(f"   Symbol: {order.get('symbol', 'unknown')}")
        print(f"   Quantity: {order.get('quantity', 'unknown')}")
        
        if isinstance(metadata, dict):
            print(f"   Metadata keys: {list(metadata.keys())}")
            if 'entry_signal' in metadata:
                print(f"   ✓ entry_signal: {metadata['entry_signal']}")
            else:
                print(f"   ❌ NO entry_signal")
                # Check if we have any decision info
                if 'decision' in metadata:
                    print(f"   Decision: {metadata['decision']}")
        else:
            print(f"   Metadata type: {type(metadata)}")
else:
    print("No order data found")

# Also check the signals to understand the flow
print("\n\n=== Checking Signal Flow ===")
signal_file = results_dir / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"

if signal_file.exists():
    signals = pd.read_parquet(signal_file)
    print(f"Total signal changes: {len(signals)}")
    print(f"Signal coverage: {len(signals) / 4154 * 100:.1f}%")
    
    # Show first few signals
    print("\nFirst 10 signal changes:")
    for i in range(min(10, len(signals))):
        sig = signals.iloc[i]
        print(f"   Bar {sig['idx']}: signal = {sig['val']}")