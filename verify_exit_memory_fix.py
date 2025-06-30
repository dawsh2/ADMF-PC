#!/usr/bin/env python3
"""Verify that the exit memory fix is working correctly."""

import pandas as pd
import numpy as np
from pathlib import Path

print("=== Exit Memory Fix Verification ===")

# Find the latest results directory
results_dir = Path("config/bollinger/results/latest")
if not results_dir.exists():
    print("Error: No latest results directory found")
    exit(1)

# Load position events
position_events_file = results_dir / "traces/events/portfolio/position_events.parquet"
if position_events_file.exists():
    positions = pd.read_parquet(position_events_file)
    print(f"\nFound {len(positions)} position events")
    
    # Check for strategy_id in POSITION_OPEN events
    open_events = positions[positions['event_type'] == 'POSITION_OPEN']
    print(f"POSITION_OPEN events: {len(open_events)}")
    
    # Count how many have strategy_id
    has_strategy_id = open_events['strategy_id'].notna().sum()
    print(f"POSITION_OPEN events with strategy_id: {has_strategy_id}/{len(open_events)}")
    
    if has_strategy_id < len(open_events):
        print("\n⚠️ WARNING: Not all POSITION_OPEN events have strategy_id!")
        print("This means the exit memory fix may not be fully working.")
        
        # Show sample of events without strategy_id
        missing_strategy = open_events[open_events['strategy_id'].isna()]
        print(f"\nEvents missing strategy_id: {len(missing_strategy)}")
        if len(missing_strategy) > 0:
            print("\nFirst few events without strategy_id:")
            print(missing_strategy[['timestamp', 'symbol', 'quantity', 'entry_price']].head())
    else:
        print("\n✓ All POSITION_OPEN events have strategy_id - fix appears to be working!")
        
else:
    print(f"Error: Position events file not found at {position_events_file}")

# Also check trades to see if we still have immediate re-entries
trades_file = results_dir / "traces/events/portfolio/trades.parquet"
if trades_file.exists():
    trades = pd.read_parquet(trades_file)
    print(f"\n\nTotal trades: {len(trades)}")
    
    # Check for immediate re-entries
    trades['timestamp'] = pd.to_datetime(trades['timestamp'])
    trades = trades.sort_values(['symbol', 'timestamp'])
    
    # Group by symbol and look for immediate re-entries
    immediate_reentries = 0
    for symbol in trades['symbol'].unique():
        symbol_trades = trades[trades['symbol'] == symbol].copy()
        
        # For each exit, check if there's an immediate re-entry
        for i in range(len(symbol_trades) - 1):
            curr = symbol_trades.iloc[i]
            next = symbol_trades.iloc[i + 1]
            
            # If current is an exit and next is an entry
            if curr.get('exit_type') in ['stop_loss', 'take_profit', 'trailing_stop']:
                time_diff = (next['timestamp'] - curr['timestamp']).total_seconds() / 60
                if time_diff == 0:  # Same bar
                    immediate_reentries += 1
                    
    print(f"Immediate re-entries after risk exits: {immediate_reentries}")
    
    if immediate_reentries > 0:
        print("\n❌ Still seeing immediate re-entries - fix may not be working correctly")
    else:
        print("\n✓ No immediate re-entries found - fix is working!")
else:
    print(f"\nTrades file not found at {trades_file}")

print("\n=== Summary ===")
print("Run this script after your backtest to verify if the exit memory fix is working.")
print("If you still see 453 trades and immediate re-entries, the fix needs debugging.")