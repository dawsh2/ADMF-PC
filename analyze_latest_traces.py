#!/usr/bin/env python3
"""Analyze the latest trace files to verify position tracking fixes."""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Find the latest results directory
results_base = Path('config/bollinger/results')
latest_dir = max(results_base.glob('20250627_*'), key=lambda p: p.name)
traces_dir = latest_dir / 'traces'

print(f"Analyzing traces from: {latest_dir}")
print(f"Timestamp: {latest_dir.name}")
print("=" * 80)

# Load all trace files
try:
    # Load signals
    signal_files = list(traces_dir.glob('signals/*/*.parquet'))
    if signal_files:
        signals = pd.read_parquet(signal_files[0])
        print(f"\n1. Signals loaded: {len(signals)} records")
    else:
        print("\n1. No signal files found")
        signals = None

    # Load orders
    order_files = list(traces_dir.glob('portfolio/orders/*.parquet'))
    if order_files:
        orders = pd.read_parquet(order_files[0])
        print(f"\n2. Orders loaded: {len(orders)} records")
        
        # Check metadata
        orders_with_metadata = orders[orders['metadata'].notna()]
        print(f"   - Orders with metadata: {len(orders_with_metadata)} ({len(orders_with_metadata)/len(orders)*100:.1f}%)")
    else:
        print("\n2. No order files found")
        orders = None

    # Load fills
    fill_files = list(traces_dir.glob('execution/fills/*.parquet'))
    if fill_files:
        fills = pd.read_parquet(fill_files[0])
        print(f"\n3. Fills loaded: {len(fills)} records")
        
        # Check metadata
        fills_with_metadata = fills[fills['metadata'].notna()]
        print(f"   - Fills with metadata: {len(fills_with_metadata)} ({len(fills_with_metadata)/len(fills)*100:.1f}%)")
    else:
        print("\n3. No fill files found")
        fills = None

    # Load position opens
    position_open_files = list(traces_dir.glob('portfolio/positions_open/*.parquet'))
    position_opens = None
    if position_open_files:
        position_opens = pd.read_parquet(position_open_files[0])
        print(f"\n4. Position opens loaded: {len(position_opens)} records")
        
        # Check metadata
        opens_with_metadata = position_opens[position_opens['metadata'].notna()]
        print(f"   - Position opens with metadata: {len(opens_with_metadata)} ({len(opens_with_metadata)/len(position_opens)*100:.1f}%)")
    else:
        print("\n4. No position open files found")

    # Load position closes
    position_close_files = list(traces_dir.glob('portfolio/positions_close/*.parquet'))
    if position_close_files:
        position_closes = pd.read_parquet(position_close_files[0])
        print(f"\n5. Position closes loaded: {len(position_closes)} records")
        
        # Check metadata
        closes_with_metadata = position_closes[position_closes['metadata'].notna()]
        print(f"   - Position closes with metadata: {len(closes_with_metadata)} ({len(closes_with_metadata)/len(position_closes)*100:.1f}%)")
    else:
        print("\n5. No position close files found")
        position_closes = None

    # Analyze patterns
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    
    # Check if we have proper position tracking now
    if orders is not None and position_opens is not None:
        print(f"\nPosition tracking ratio:")
        print(f"  - Orders: {len(orders)}")
        print(f"  - Position opens: {len(position_opens)}")
        print(f"  - Ratio: {len(position_opens)/len(orders)*100:.1f}%")
        
        # Expected: ~50% of orders should create positions (buy orders)
        # The other ~50% are sell orders that close positions
        if len(position_opens) > len(orders) * 0.4:
            print("  ✅ Position tracking appears to be working correctly!")
        else:
            print("  ❌ Position tracking still seems incomplete")
    
    # Sample position metadata
    if position_opens is not None and len(position_opens) > 0:
        print(f"\nSample position open metadata:")
        for i in range(min(3, len(position_opens))):
            metadata = position_opens.iloc[i]['metadata']
            if metadata:
                try:
                    data = json.loads(metadata) if isinstance(metadata, str) else metadata
                    print(f"\nPosition {i+1}:")
                    print(f"  - Symbol: {data.get('symbol')}")
                    print(f"  - Side: {data.get('side')}")
                    print(f"  - Quantity: {data.get('quantity')}")
                    print(f"  - Entry price: {data.get('entry_price')}")
                    print(f"  - Strategy: {data.get('strategy_id')}")
                except:
                    print(f"\nPosition {i+1}: Unable to parse metadata")
    
    # Check order pattern
    if orders is not None and len(orders) > 0:
        print(f"\nOrder pattern analysis:")
        
        # Check if we have alternating buy/sell
        buy_orders = orders[orders['val'] == 1]  # long direction
        sell_orders = orders[orders['val'] == -1]  # short direction
        
        print(f"  - Buy orders: {len(buy_orders)}")
        print(f"  - Sell orders: {len(sell_orders)}")
        
        # Check bar gaps between orders
        order_gaps = orders['idx'].diff()
        print(f"\nOrder frequency:")
        print(f"  - Mean gap: {order_gaps.mean():.1f} bars")
        print(f"  - Median gap: {order_gaps.median():.1f} bars")
        print(f"  - Min gap: {order_gaps.min():.0f} bars")
        print(f"  - Max gap: {order_gaps.max():.0f} bars")

except Exception as e:
    print(f"Error analyzing traces: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("SUMMARY:")
if position_opens is not None and orders is not None:
    if len(position_opens) > len(orders) * 0.4:
        print("✅ The sparse storage fix appears to be working!")
        print(f"✅ We now have {len(position_opens)} position events for {len(orders)} orders")
        print("✅ This should enable proper trade reconstruction and performance analysis")
    else:
        print("❌ Position tracking still needs investigation")
        print(f"❌ Only {len(position_opens)} position events for {len(orders)} orders")
else:
    print("❌ Unable to verify - missing required trace files")