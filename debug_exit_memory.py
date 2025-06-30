#!/usr/bin/env python3
"""Debug exit memory behavior in detail."""

import pandas as pd
import numpy as np
from pathlib import Path

print("=== Exit Memory Debug Analysis ===")

# Load the trades
results_dir = Path("config/bollinger/results/latest")
trades_file = results_dir / "traces/events/portfolio/trades.parquet"

if not trades_file.exists():
    print(f"Error: Trades file not found at {trades_file}")
    exit(1)

trades = pd.read_parquet(trades_file)
trades['timestamp'] = pd.to_datetime(trades['timestamp'])
trades = trades.sort_values(['symbol', 'timestamp'])

print(f"Total trades: {len(trades)}")

# Look for immediate re-entries after risk exits
risk_exit_types = ['stop_loss', 'take_profit', 'trailing_stop']
immediate_reentries = []

for symbol in trades['symbol'].unique():
    symbol_trades = trades[trades['symbol'] == symbol].reset_index(drop=True)
    
    for i in range(len(symbol_trades) - 1):
        curr = symbol_trades.iloc[i]
        next = symbol_trades.iloc[i + 1]
        
        # Check if current is a risk exit
        if curr.get('exit_type') in risk_exit_types:
            time_diff = (next['timestamp'] - curr['timestamp']).total_seconds() / 60
            
            if time_diff == 0:  # Same bar
                immediate_reentries.append({
                    'exit_trade': i,
                    'exit_time': curr['timestamp'],
                    'exit_type': curr['exit_type'],
                    'exit_strategy_id': curr.get('strategy_id', 'MISSING'),
                    'entry_trade': i + 1,
                    'entry_time': next['timestamp'],
                    'entry_strategy_id': next.get('strategy_id', 'MISSING'),
                    'symbol': symbol
                })

print(f"\nImmediate re-entries found: {len(immediate_reentries)}")

if len(immediate_reentries) > 0:
    # Show details of first few
    print("\nFirst 5 immediate re-entries:")
    for j, reentry in enumerate(immediate_reentries[:5]):
        print(f"\n{j+1}. {reentry['symbol']} at {reentry['exit_time']}")
        print(f"   Exit: {reentry['exit_type']} (strategy_id: {reentry['exit_strategy_id']})")
        print(f"   Entry: Same bar (strategy_id: {reentry['entry_strategy_id']})")
        
    # Check if strategy_id is missing
    missing_strategy_id = [r for r in immediate_reentries if r['exit_strategy_id'] == 'MISSING' or r['entry_strategy_id'] == 'MISSING']
    if missing_strategy_id:
        print(f"\n⚠️ WARNING: {len(missing_strategy_id)} re-entries have missing strategy_id!")
        print("This explains why exit memory is not working.")
    
    # Group by exit type
    print("\nRe-entries by exit type:")
    exit_type_counts = {}
    for r in immediate_reentries:
        exit_type = r['exit_type']
        exit_type_counts[exit_type] = exit_type_counts.get(exit_type, 0) + 1
    
    for exit_type, count in exit_type_counts.items():
        print(f"  {exit_type}: {count}")
        
    # Check if it's always the same strategy_id
    unique_strategies = set()
    for r in immediate_reentries:
        unique_strategies.add(r['exit_strategy_id'])
        unique_strategies.add(r['entry_strategy_id'])
    
    print(f"\nUnique strategy IDs involved: {unique_strategies}")

# Also check the signals to understand the pattern
signals_file = results_dir / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"
if signals_file.exists():
    signals = pd.read_parquet(signals_file)
    signals['ts'] = pd.to_datetime(signals['ts'])
    
    # Check signal values at re-entry times
    print("\n\n=== Signal Analysis at Re-entry Times ===")
    
    for i, reentry in enumerate(immediate_reentries[:3]):  # First 3
        exit_time = reentry['exit_time']
        
        # Find signal at this time
        signal_at_time = signals[signals['ts'] == exit_time]
        if not signal_at_time.empty:
            signal_val = signal_at_time.iloc[0]['val']
            print(f"\n{i+1}. Time: {exit_time}")
            print(f"   Signal value: {signal_val}")
            print(f"   Exit type: {reentry['exit_type']}")
            
            # Check if signal changed from previous bar
            idx = signals[signals['ts'] == exit_time].index[0]
            if idx > 0:
                prev_signal = signals.iloc[idx-1]['val']
                print(f"   Previous signal: {prev_signal}")
                print(f"   Signal changed: {prev_signal != signal_val}")

print("\n\n=== Diagnosis ===")
print("If you see immediate re-entries with 'MISSING' strategy_id:")
print("  - The fix hasn't been applied or isn't working")
print("  - Need to ensure metadata is passed through the entire chain")
print("\nIf strategy_id is present but re-entries still happen:")
print("  - Exit memory logic may have other issues")
print("  - Check risk manager's exit memory implementation")