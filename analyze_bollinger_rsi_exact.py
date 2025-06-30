#!/usr/bin/env python3
"""
Analyze Bollinger RSI Exact strategy performance
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load signals
signal_file = Path("/Users/daws/ADMF-PC/workspaces/signal_generation_9812a86b/traces/SPY_1m/signals/bollinger_rsi_exact/SPY_compiled_strategy_0.parquet")
signals = pd.read_parquet(signal_file)

print("="*60)
print("BOLLINGER RSI EXACT STRATEGY ANALYSIS")
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
print(f"Signal changes in data: {len(signals)}")

if len(trades_df) > 0:
    # Overall statistics
    print("\nOVERALL PERFORMANCE:")
    print(f"Gross return: {trades_df['pnl_pct'].sum():.2f}%")
    print(f"Win rate: {(trades_df['pnl_pct'] > 0).mean():.1%}")
    print(f"Average PnL per trade: {trades_df['pnl_pct'].mean():.3f}%")
    print(f"Median PnL per trade: {trades_df['pnl_pct'].median():.3f}%")
    
    # Transaction costs
    print("\nNET RETURNS AFTER COSTS:")
    for cost_bp in [0.5, 1.0, 2.0, 5.0]:
        net = trades_df['pnl_pct'].sum() - len(trades_df) * cost_bp / 100
        print(f"  {cost_bp} bps: {net:.2f}% net return")
    
    # Duration analysis
    print("\nDURATION ANALYSIS:")
    print(f"Average duration: {trades_df['duration'].mean():.1f} bars")
    print(f"Median duration: {trades_df['duration'].median():.0f} bars")
    
    print("\nBy duration bucket:")
    for d_range in [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100)]:
        d_trades = trades_df[trades_df['duration'].between(d_range[0], d_range[1])]
        if len(d_trades) > 0:
            gross = d_trades['pnl_pct'].sum()
            net = gross - len(d_trades) * 0.01
            win_rate = (d_trades['pnl_pct'] > 0).mean()
            avg_pnl = d_trades['pnl_pct'].mean()
            print(f"  {d_range[0]:3d}-{d_range[1]:3d} bars: {len(d_trades):4d} trades, "
                  f"{win_rate:.1%} win, {avg_pnl:6.3f}% avg, {net:7.2f}% net")
    
    # Signal type breakdown
    print("\nBY SIGNAL TYPE:")
    for signal_type in ['long', 'short']:
        type_trades = trades_df[trades_df['signal_type'] == signal_type]
        if len(type_trades) > 0:
            net = type_trades['pnl_pct'].sum() - len(type_trades) * 0.01
            win_rate = (type_trades['pnl_pct'] > 0).mean()
            avg_pnl = type_trades['pnl_pct'].mean()
            print(f"{signal_type.capitalize()}: {len(type_trades)} trades, "
                  f"{win_rate:.1%} win, {avg_pnl:.3f}% avg, {net:.2f}% net")
    
    # Trade size distribution
    print("\nTRADE SIZE DISTRIBUTION:")
    print(f"Largest win: {trades_df['pnl_pct'].max():.2f}%")
    print(f"Largest loss: {trades_df['pnl_pct'].min():.2f}%")
    print(f"Std deviation: {trades_df['pnl_pct'].std():.3f}%")
    
    # Win/Loss analysis
    winners = trades_df[trades_df['pnl_pct'] > 0]
    losers = trades_df[trades_df['pnl_pct'] < 0]
    
    if len(winners) > 0 and len(losers) > 0:
        print(f"\nAverage win: {winners['pnl_pct'].mean():.3f}%")
        print(f"Average loss: {losers['pnl_pct'].mean():.3f}%")
        print(f"Win/Loss ratio: {abs(winners['pnl_pct'].mean() / losers['pnl_pct'].mean()):.2f}")
    
    # Monthly estimate
    total_bars = trades_df['exit_idx'].max() - trades_df['entry_idx'].min() if len(trades_df) > 0 else 1
    trades_per_month = len(trades_df) / (total_bars / (390 * 20))
    print(f"\nApproximate trades per month: {trades_per_month:.1f}")
    
    # Compare to our backtested RSI divergence
    print("\n" + "="*60)
    print("COMPARISON TO ORIGINAL RSI DIVERGENCE BACKTEST:")
    print(f"This implementation: {len(trades_df)} trades")
    print(f"Original backtest: 494 trades")
    print(f"Ratio: {len(trades_df) / 494:.1f}x")
    print("")
    print(f"This win rate: {(trades_df['pnl_pct'] > 0).mean():.1%}")
    print(f"Original win rate: 71.9%")
    print("")
    print(f"This avg duration: {trades_df['duration'].mean():.1f} bars")
    print(f"Original avg duration: ~12 bars")
    print("")
    print(f"This net return (1bp): {trades_df['pnl_pct'].sum() - len(trades_df) * 0.01:.2f}%")
    print(f"Original net return: 11.82%")
    
    # Sample trades
    print("\nSAMPLE TRADES (first 10):")
    print("-" * 80)
    for i, trade in trades_df.head(10).iterrows():
        print(f"Trade {i+1}: Entry {trade['entry_idx']} → Exit {trade['exit_idx']} "
              f"({trade['duration']} bars), {trade['signal_type']}, "
              f"PnL: {trade['pnl_pct']:.3f}%")

else:
    print("\nNo trades found!")
    print(f"Total signal changes: {len(signals)}")
    print("\nFirst 10 signal changes:")
    print(signals.head(10))

# Check metadata
metadata_file = Path("/Users/daws/ADMF-PC/workspaces/signal_generation_9812a86b/metadata.json")
if metadata_file.exists():
    import json
    with open(metadata_file) as f:
        metadata = json.load(f)
    print("\n" + "="*60)
    print("WORKSPACE METADATA:")
    print(f"Total bars: {metadata.get('total_bars', 'N/A')}")
    print(f"Strategy type: {metadata['components']['SPY_compiled_strategy_0']['strategy_type']}")
    print(f"Signal changes: {metadata['components']['SPY_compiled_strategy_0']['signal_changes']}")
    print(f"Parameters: {metadata['components']['SPY_compiled_strategy_0']['parameters']}")