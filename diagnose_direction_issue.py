#!/usr/bin/env python3
"""
Diagnose why losers go to -0.1% and winners stop at 0.075%
This suggests stop/target logic might be inverted for one direction
"""
import pandas as pd
import numpy as np

def diagnose_direction_issue(trades_df):
    """Analyze trade outcomes by direction"""
    
    print("=== Direction Diagnosis ===\n")
    
    # Group by return percentage (rounded to identify clusters)
    trades_df['return_bucket'] = (trades_df['return_pct'] * 1000).round() / 1000
    
    print("Return distribution:")
    print(trades_df['return_bucket'].value_counts().sort_index())
    print()
    
    # Identify trades near key levels
    stop_loss_trades = trades_df[abs(trades_df['return_pct'] + 0.00075) < 0.0001]
    take_profit_trades = trades_df[abs(trades_df['return_pct'] - 0.001) < 0.0001]
    
    # Also check inverted levels
    inverted_stop_trades = trades_df[abs(trades_df['return_pct'] - 0.00075) < 0.0001]
    inverted_tp_trades = trades_df[abs(trades_df['return_pct'] + 0.001) < 0.0001]
    
    print(f"Trades at -0.075% (stop loss): {len(stop_loss_trades)}")
    print(f"Trades at +0.1% (take profit): {len(take_profit_trades)}")
    print(f"Trades at +0.075% (inverted stop?): {len(inverted_stop_trades)}")
    print(f"Trades at -0.1% (inverted TP?): {len(inverted_tp_trades)}")
    print()
    
    # Check if we have direction info
    if 'direction' in trades_df.columns:
        print("By direction:")
        for direction in trades_df['direction'].unique():
            dir_trades = trades_df[trades_df['direction'] == direction]
            print(f"\n{direction} trades:")
            print(f"  Total: {len(dir_trades)}")
            print(f"  At -0.075%: {len(dir_trades[abs(dir_trades['return_pct'] + 0.00075) < 0.0001])}")
            print(f"  At +0.1%: {len(dir_trades[abs(dir_trades['return_pct'] - 0.001) < 0.0001])}")
            print(f"  At +0.075%: {len(dir_trades[abs(dir_trades['return_pct'] - 0.00075) < 0.0001])}")
            print(f"  At -0.1%: {len(dir_trades[abs(dir_trades['return_pct'] + 0.001) < 0.0001])}")
    
    # Check exit reasons if available
    if 'exit_reason' in trades_df.columns:
        print("\nExit reasons by return level:")
        for level, label in [(-0.001, "-0.1%"), (-0.00075, "-0.075%"), 
                           (0.00075, "+0.075%"), (0.001, "+0.1%")]:
            level_trades = trades_df[abs(trades_df['return_pct'] - level) < 0.0001]
            if len(level_trades) > 0:
                print(f"\n{label}:")
                print(level_trades['exit_reason'].value_counts())
    
    # Analyze the pattern
    print("\n=== DIAGNOSIS ===")
    if len(inverted_tp_trades) > len(take_profit_trades):
        print("⚠️  More trades hitting -0.1% than +0.1%")
        print("   This suggests LOSING trades are running to the take profit level")
    
    if len(inverted_stop_trades) > len(stop_loss_trades):
        print("⚠️  More trades hitting +0.075% than -0.075%")
        print("   This suggests WINNING trades are being stopped out")
    
    print("\nPossible issues:")
    print("1. Signal directions might be inverted (LONG/SHORT swapped)")
    print("2. Stop/target calculations might use wrong sign for one direction")
    print("3. Position quantity signs might be inverted")
    
    return trades_df

# If you have a trades DataFrame, run:
# diagnose_direction_issue(trades_df)

# Or analyze from parquet files:
def analyze_workspace(workspace_path):
    """Analyze trades from a workspace"""
    import os
    import glob
    
    # Find trace files
    pattern = os.path.join(workspace_path, "traces", "*", "trace_*.parquet")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No trace files found in {workspace_path}")
        return
    
    # Load and combine
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    
    traces = pd.concat(dfs, ignore_index=True)
    
    # Calculate returns from entry/exit events
    entries = traces[traces['event_type'] == 'POSITION_OPENED'].copy()
    exits = traces[traces['event_type'] == 'POSITION_CLOSED'].copy()
    
    # Match entries and exits
    trades = []
    for _, exit in exits.iterrows():
        # Find matching entry
        entry_mask = (
            (entries['timestamp'] < exit['timestamp']) &
            (entries['strategy_id'] == exit['strategy_id'])
        )
        if entry_mask.any():
            entry = entries[entry_mask].iloc[-1]
            
            # Calculate return
            entry_price = entry['price']
            exit_price = exit['price']
            quantity = entry.get('quantity', entry.get('size', 1))
            
            if quantity > 0:  # Long
                return_pct = (exit_price - entry_price) / entry_price
            else:  # Short
                return_pct = (entry_price - exit_price) / entry_price
            
            trades.append({
                'entry_time': entry['timestamp'],
                'exit_time': exit['timestamp'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return_pct': return_pct,
                'direction': 'LONG' if quantity > 0 else 'SHORT',
                'exit_reason': exit.get('metadata', {}).get('exit_reason', 'unknown')
            })
    
    trades_df = pd.DataFrame(trades)
    return diagnose_direction_issue(trades_df)

print("Usage:")
print("1. If you have trades_df: diagnose_direction_issue(trades_df)")
print("2. To analyze a workspace: analyze_workspace('path/to/workspace')")