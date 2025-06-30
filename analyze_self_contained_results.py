#!/usr/bin/env python3
"""
Analyze Bollinger RSI Self-Contained results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

if len(sys.argv) < 2:
    print("Usage: python3 analyze_self_contained_results.py <workspace_id>")
    sys.exit(1)

workspace = sys.argv[1]

# Load signals - look for self_contained strategy
signal_file = Path(f"/Users/daws/ADMF-PC/workspaces/{workspace}/traces/SPY_1m/signals/bollinger_rsi_self_contained/SPY_compiled_strategy_0.parquet")

if not signal_file.exists():
    print(f"Signal file not found, checking directory...")
    traces_dir = Path(f"/Users/daws/ADMF-PC/workspaces/{workspace}/traces/")
    if traces_dir.exists():
        print(f"Contents of traces directory:")
        for item in traces_dir.rglob("*.parquet"):
            print(f"  {item}")
    sys.exit(1)

signals = pd.read_parquet(signal_file)

print("="*60)
print("BOLLINGER RSI SELF-CONTAINED - RESULTS")
print(f"Workspace: {workspace}")
print("="*60)

# Extract trades
trades = []
entry_idx = None
entry_signal = None
entry_price = None

for _, row in signals.iterrows():
    signal = row['val']
    bar_idx = row['idx']
    price = row['px']
    
    if entry_idx is None and signal != 0:
        entry_idx = bar_idx
        entry_signal = signal
        entry_price = price
            
    elif entry_idx is not None and (signal == 0 or signal != entry_signal):
        # Exit - calculate PnL
        if entry_signal > 0:  # Long
            pnl_pct = (price - entry_price) / entry_price * 100
        else:  # Short
            pnl_pct = (entry_price - price) / entry_price * 100
            
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': bar_idx,
            'duration': bar_idx - entry_idx,
            'entry_price': entry_price,
            'exit_price': price,
            'pnl_pct': pnl_pct,
            'signal_type': 'long' if entry_signal > 0 else 'short'
        })
        
        # Check for re-entry
        if signal != 0:
            entry_idx = bar_idx
            entry_signal = signal
            entry_price = price
        else:
            entry_idx = None

trades_df = pd.DataFrame(trades)

print(f"\nTotal trades: {len(trades_df)}")
print(f"Signal changes: {len(signals)}")

if len(trades_df) > 0:
    # Overall statistics
    print("\nPERFORMANCE METRICS:")
    print(f"Gross return: {trades_df['pnl_pct'].sum():.2f}%")
    print(f"Win rate: {(trades_df['pnl_pct'] > 0).mean():.1%}")
    print(f"Average PnL per trade: {trades_df['pnl_pct'].mean():.3f}%")
    
    # Transaction costs
    print("\nNET RETURNS:")
    net_1bp = trades_df['pnl_pct'].sum() - len(trades_df) * 0.01
    print(f"After 1bp costs: {net_1bp:.2f}%")
    
    # Duration
    print(f"\nAverage duration: {trades_df['duration'].mean():.1f} bars")
    print(f"Median duration: {trades_df['duration'].median():.0f} bars")
    
    # Duration breakdown
    print("\nDuration breakdown:")
    for d_range in [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]:
        d_trades = trades_df[trades_df['duration'].between(d_range[0], d_range[1])]
        if len(d_trades) > 0:
            win_rate = (d_trades['pnl_pct'] > 0).mean()
            net = d_trades['pnl_pct'].sum() - len(d_trades) * 0.01
            print(f"  {d_range[0]:2d}-{d_range[1]:2d} bars: {len(d_trades):3d} trades, "
                  f"{win_rate:.1%} win, {net:6.2f}% net")
    
    # Signal type breakdown
    print("\nBY SIGNAL TYPE:")
    for signal_type in ['long', 'short']:
        type_trades = trades_df[trades_df['signal_type'] == signal_type]
        if len(type_trades) > 0:
            net = type_trades['pnl_pct'].sum() - len(type_trades) * 0.01
            win_rate = (type_trades['pnl_pct'] > 0).mean()
            print(f"{signal_type.capitalize()}: {len(type_trades)} trades, "
                  f"{win_rate:.1%} win, {net:.2f}% net")
    
    # Compare to expected
    print("\n" + "="*60)
    print("COMPARISON TO EXPECTED RESULTS:")
    print(f"Trades:    {len(trades_df)} (expected: ~494)")
    print(f"Win rate:  {(trades_df['pnl_pct'] > 0).mean():.1%} (expected: 71.9%)")
    print(f"Net return: {net_1bp:.2f}% (expected: 11.82%)")
    print(f"Avg duration: {trades_df['duration'].mean():.1f} bars (expected: ~12)")
    
    # Sample trades
    print("\nSAMPLE TRADES (first 10):")
    for i, trade in trades_df.head(10).iterrows():
        print(f"  {trade['signal_type']}: {trade['duration']} bars, {trade['pnl_pct']:.3f}%")