#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Read signals and positions data
signals = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/20250627_173309/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet')
positions_close = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/20250627_173309/traces/portfolio/positions_close/positions_close.parquet')

print("=== Signal Analysis ===")
print(f"Total signal changes: {len(signals)}")
print(f"Signal values: {signals['val'].unique()}")

# Count signal transitions
signal_changes = []
for i in range(1, len(signals)):
    prev_val = signals.iloc[i-1]['val']
    curr_val = signals.iloc[i]['val']
    if prev_val != 0 and curr_val == 0:  # Exit signal
        signal_changes.append('exit')
    elif prev_val == 0 and curr_val != 0:  # Entry signal
        signal_changes.append('entry')
    elif prev_val * curr_val < 0:  # Direction change (e.g., 1 to -1)
        signal_changes.append('reversal')

print(f"\nSignal transitions:")
print(f"  Entries: {signal_changes.count('entry')}")
print(f"  Exits: {signal_changes.count('exit')}")
print(f"  Reversals: {signal_changes.count('reversal')}")

print("\n=== Trade Analysis ===")
print(f"Total trades closed: {len(positions_close)}")

# Analyze exit types
exit_types = {}
win_count = 0
loss_count = 0

for i in range(len(positions_close)):
    metadata = positions_close.iloc[i]['metadata']
    if isinstance(metadata, dict):
        exit_type = metadata.get('metadata', {}).get('exit_type', 'unknown')
        exit_types[exit_type] = exit_types.get(exit_type, 0) + 1
        
        pnl = metadata.get('realized_pnl', 0)
        if pnl > 0:
            win_count += 1
        elif pnl < 0:
            loss_count += 1

print("\nExit type breakdown:")
for exit_type, count in sorted(exit_types.items()):
    print(f"  {exit_type}: {count} ({count/len(positions_close)*100:.1f}%)")

print(f"\nWin rate: {win_count/(win_count+loss_count)*100:.1f}%")
print(f"  Wins: {win_count}")
print(f"  Losses: {loss_count}")

# Check if we're exiting too early
print("\n=== Potential Issues ===")

# 1. Check for immediate exits after entry
print("\n1. Checking for premature exits...")
positions_open = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/20250627_173309/traces/portfolio/positions_open/positions_open.parquet')

# Match opens and closes
premature_exits = 0
for i in range(min(len(positions_open), len(positions_close))):
    open_bar = positions_open.iloc[i]['idx']
    close_bar = positions_close.iloc[i]['idx']
    bars_held = close_bar - open_bar
    
    if bars_held <= 1:
        premature_exits += 1

print(f"   Trades exited within 1 bar: {premature_exits}")

# 2. Check stop loss frequency
stop_losses = exit_types.get('stop_loss', 0)
take_profits = exit_types.get('take_profit', 0)
signal_exits = exit_types.get('signal', 0)

print(f"\n2. Exit distribution analysis:")
print(f"   Stop losses: {stop_losses} ({stop_losses/len(positions_close)*100:.1f}%)")
print(f"   Take profits: {take_profits} ({take_profits/len(positions_close)*100:.1f}%)")
print(f"   Signal exits: {signal_exits} ({signal_exits/len(positions_close)*100:.1f}%)")

if stop_losses > take_profits * 2:
    print("   ⚠️ WARNING: Too many stop losses relative to take profits!")

# 3. Check if signals are being followed
print(f"\n3. Signal following check:")
print(f"   Total signals that changed: {len(signals)}")
print(f"   Total trades executed: {len(positions_close)}")
print(f"   Ratio: {len(positions_close)/len(signals):.2f}")

if len(positions_close) < len(signals) * 0.3:
    print("   ⚠️ WARNING: Many signals not resulting in trades!")

# 4. Sample some losing trades to see what happened
print("\n4. Sample losing trades:")
loss_count = 0
for i in range(len(positions_close)):
    if loss_count >= 5:
        break
    
    metadata = positions_close.iloc[i]['metadata']
    if isinstance(metadata, dict):
        pnl = metadata.get('realized_pnl', 0)
        if pnl < 0:
            qty = metadata.get('quantity', 0)
            entry = metadata.get('entry_price', 0)
            exit = metadata.get('exit_price', 0)
            exit_type = metadata.get('metadata', {}).get('exit_type', 'unknown')
            
            print(f"\n   Trade {i+1}:")
            print(f"     Type: {'LONG' if qty > 0 else 'SHORT'}")
            print(f"     Entry: ${entry:.2f}")
            print(f"     Exit: ${exit:.2f}")
            print(f"     Loss: ${pnl:.2f}")
            print(f"     Exit type: {exit_type}")
            
            loss_count += 1

print("\n=== Hypothesis ===")
print("Possible reasons for low win rate:")
print("1. Exit memory preventing re-entry after stops")
print("2. Intraday constraint forcing exits")
print("3. Risk management too aggressive")
print("4. Signal interpretation differences")