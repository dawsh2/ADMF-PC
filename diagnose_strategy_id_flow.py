#!/usr/bin/env python3
"""Diagnose the flow of strategy_id through the system."""

import pandas as pd
from pathlib import Path

print("=== Strategy ID Flow Diagnosis ===")

results_dir = Path("config/bollinger/results/latest")

# 1. Check SIGNAL events for strategy_id
signals_events_file = results_dir / "traces/events/strategy/signal_events.parquet"
if signals_events_file.exists():
    signal_events = pd.read_parquet(signals_events_file)
    print(f"\n1. SIGNAL Events: {len(signal_events)}")
    has_strategy_id = signal_events['strategy_id'].notna().sum()
    print(f"   With strategy_id: {has_strategy_id}/{len(signal_events)}")
    if has_strategy_id < len(signal_events):
        print("   ⚠️ Some signals missing strategy_id!")
else:
    print("\n1. SIGNAL events file not found")

# 2. Check ORDER events for strategy_id in metadata
order_events_file = results_dir / "traces/events/portfolio/order_events.parquet"
if order_events_file.exists():
    order_events = pd.read_parquet(order_events_file)
    print(f"\n2. ORDER Events: {len(order_events)}")
    
    # Check metadata column
    if 'metadata' in order_events.columns:
        orders_with_metadata = order_events['metadata'].notna().sum()
        print(f"   With metadata: {orders_with_metadata}/{len(order_events)}")
        
        # Check for strategy_id in metadata
        def has_strategy_id(metadata):
            if pd.isna(metadata):
                return False
            if isinstance(metadata, dict):
                return 'strategy_id' in metadata
            return False
        
        orders_with_strategy_id = order_events['metadata'].apply(has_strategy_id).sum()
        print(f"   With strategy_id in metadata: {orders_with_strategy_id}/{len(order_events)}")
    else:
        print("   ⚠️ No metadata column in order events!")
else:
    print("\n2. ORDER events file not found")

# 3. Check FILL events for strategy_id in metadata
fill_events_file = results_dir / "traces/events/execution/fill_events.parquet"
if fill_events_file.exists():
    fill_events = pd.read_parquet(fill_events_file)
    print(f"\n3. FILL Events: {len(fill_events)}")
    
    if 'metadata' in fill_events.columns:
        fills_with_metadata = fill_events['metadata'].notna().sum()
        print(f"   With metadata: {fills_with_metadata}/{len(fill_events)}")
        
        fills_with_strategy_id = fill_events['metadata'].apply(has_strategy_id).sum()
        print(f"   With strategy_id in metadata: {fills_with_strategy_id}/{len(fill_events)}")
    else:
        print("   ⚠️ No metadata column in fill events!")
else:
    print("\n3. FILL events file not found")

# 4. Check POSITION events for strategy_id
position_events_file = results_dir / "traces/events/portfolio/position_events.parquet"
if position_events_file.exists():
    position_events = pd.read_parquet(position_events_file)
    open_events = position_events[position_events['event_type'] == 'POSITION_OPEN']
    print(f"\n4. POSITION_OPEN Events: {len(open_events)}")
    
    # Direct strategy_id column
    if 'strategy_id' in open_events.columns:
        pos_with_strategy_id = open_events['strategy_id'].notna().sum()
        print(f"   With strategy_id: {pos_with_strategy_id}/{len(open_events)}")
        
        if pos_with_strategy_id < len(open_events):
            missing = open_events[open_events['strategy_id'].isna()]
            print(f"\n   Missing strategy_id timestamps:")
            for _, row in missing.head(3).iterrows():
                print(f"     - {row['timestamp']}: {row['symbol']}")
    
    # Also check metadata column
    if 'metadata' in open_events.columns:
        pos_with_metadata = open_events['metadata'].notna().sum()
        print(f"   With metadata: {pos_with_metadata}/{len(open_events)}")
else:
    print("\n4. POSITION events file not found")

# 5. Check actual positions in trades
trades_file = results_dir / "traces/events/portfolio/trades.parquet"
if trades_file.exists():
    trades = pd.read_parquet(trades_file)
    print(f"\n5. Trades: {len(trades)}")
    
    if 'strategy_id' in trades.columns:
        trades_with_strategy_id = trades['strategy_id'].notna().sum()
        print(f"   With strategy_id: {trades_with_strategy_id}/{len(trades)}")
        
        # Check specifically for risk exits
        risk_exits = trades[trades['exit_type'].isin(['stop_loss', 'take_profit', 'trailing_stop'])]
        if len(risk_exits) > 0:
            risk_exits_with_id = risk_exits['strategy_id'].notna().sum()
            print(f"\n   Risk exits: {len(risk_exits)}")
            print(f"   Risk exits with strategy_id: {risk_exits_with_id}/{len(risk_exits)}")
else:
    print("\n5. Trades file not found")

print("\n\n=== Summary ===")
print("Strategy ID should flow through the system as follows:")
print("1. SIGNAL event has strategy_id")
print("2. ORDER event has strategy_id in metadata")
print("3. FILL event preserves metadata with strategy_id")
print("4. POSITION_OPEN event has strategy_id")
print("5. Position object has strategy_id in metadata")
print("\nIf any step is missing strategy_id, exit memory won't work!")