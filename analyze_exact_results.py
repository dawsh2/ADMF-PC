#!/usr/bin/env python3
"""
Analyze Bollinger RSI Divergence Exact results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

if len(sys.argv) < 2:
    print("Usage: python3 analyze_exact_results.py <workspace_id>")
    print("Example: python3 analyze_exact_results.py signal_generation_abc123")
    sys.exit(1)

workspace = sys.argv[1]

# Load signals
signal_file = Path(f"/Users/daws/ADMF-PC/workspaces/{workspace}/traces/SPY_1m/signals/bollinger_rsi_divergence_exact/SPY_compiled_strategy_0.parquet")

if not signal_file.exists():
    print(f"Error: Signal file not found at {signal_file}")
    sys.exit(1)

signals = pd.read_parquet(signal_file)

print("="*60)
print("BOLLINGER RSI DIVERGENCE EXACT - RESULTS")
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
    
    # Compare to expected
    print("\n" + "="*60)
    print("COMPARISON TO EXPECTED RESULTS:")
    print(f"Trades:    {len(trades_df)} (expected: ~494)")
    print(f"Win rate:  {(trades_df['pnl_pct'] > 0).mean():.1%} (expected: 71.9%)")
    print(f"Net return: {net_1bp:.2f}% (expected: 11.82%)")
    print(f"Avg duration: {trades_df['duration'].mean():.1f} bars (expected: ~12)")
    
    # Sample trades
    print("\nFIRST 5 TRADES:")
    for i, trade in trades_df.head(5).iterrows():
        print(f"  {trade['signal_type']}: {trade['duration']} bars, {trade['pnl_pct']:.3f}%")

else:
    print("\nNo trades found - check if strategy is generating signals correctly")