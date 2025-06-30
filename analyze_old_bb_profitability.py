#!/usr/bin/env python3
"""
Detailed profitability analysis of the old BB implementation
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load old signals
signals = pd.read_parquet("/Users/daws/ADMF-PC/workspaces/signal_generation_f88793ad/traces/SPY_1m/signals/bollinger_bands/SPY_compiled_strategy_0.parquet")

# Extract all trades
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
            pnl_dollars = price - entry_price
        else:  # Short
            pnl_pct = (entry_price - price) / entry_price * 100
            pnl_dollars = entry_price - price
            
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': bar_idx,
            'duration': bar_idx - entry_idx,
            'entry_price': entry_price,
            'exit_price': price,
            'pnl_pct': pnl_pct,
            'pnl_dollars': pnl_dollars,
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

print("="*60)
print("OLD BOLLINGER BANDS PROFITABILITY ANALYSIS")
print("="*60)

print(f"\nTotal trades: {len(trades_df)}")
print(f"Average duration: {trades_df['duration'].mean():.1f} bars")

# Overall statistics
print("\nOVERALL PERFORMANCE:")
print(f"Gross return: {trades_df['pnl_pct'].sum():.2f}%")
print(f"Transaction costs (1bp): {len(trades_df) * 0.01:.2f}%")
print(f"Net return: {trades_df['pnl_pct'].sum() - len(trades_df) * 0.01:.2f}%")
print(f"Win rate: {(trades_df['pnl_pct'] > 0).mean():.1%}")
print(f"Average win: {trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean():.4f}%")
print(f"Average loss: {trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean():.4f}%")

# Size of wins/losses
print("\nTRADE SIZE DISTRIBUTION:")
pnl_buckets = [-10, -1, -0.5, -0.1, -0.05, 0, 0.05, 0.1, 0.5, 1, 10]
trades_df['pnl_bucket'] = pd.cut(trades_df['pnl_pct'], bins=pnl_buckets)
print("\nPnL % distribution:")
print(trades_df['pnl_bucket'].value_counts().sort_index())

# Calculate win/loss stats
winners = trades_df[trades_df['pnl_pct'] > 0]
losers = trades_df[trades_df['pnl_pct'] < 0]

print(f"\nWinning trades: {len(winners)} ({len(winners)/len(trades_df)*100:.1f}%)")
print(f"  Total win: {winners['pnl_pct'].sum():.2f}%")
print(f"  Average win: {winners['pnl_pct'].mean():.4f}%")
print(f"  Median win: {winners['pnl_pct'].median():.4f}%")
print(f"  Largest win: {winners['pnl_pct'].max():.4f}%")

print(f"\nLosing trades: {len(losers)} ({len(losers)/len(trades_df)*100:.1f}%)")
print(f"  Total loss: {losers['pnl_pct'].sum():.2f}%")
print(f"  Average loss: {losers['pnl_pct'].mean():.4f}%")
print(f"  Median loss: {losers['pnl_pct'].median():.4f}%")
print(f"  Largest loss: {losers['pnl_pct'].min():.4f}%")

# Detailed duration analysis
print("\nDETAILED DURATION ANALYSIS:")
print("Duration | Trades |  Gross |   Cost |    Net | Avg PnL | Median PnL | Max PnL | Min PnL")
print("-" * 90)

for d in range(1, 21):
    d_trades = trades_df[trades_df['duration'] == d]
    if len(d_trades) > 0:
        gross = d_trades['pnl_pct'].sum()
        cost = len(d_trades) * 0.01
        net = gross - cost
        avg_pnl = d_trades['pnl_pct'].mean()
        median_pnl = d_trades['pnl_pct'].median()
        max_pnl = d_trades['pnl_pct'].max()
        min_pnl = d_trades['pnl_pct'].min()
        
        print(f"{d:8d} | {len(d_trades):6d} | {gross:6.2f}% | {cost:6.2f}% | {net:7.2f}% | "
              f"{avg_pnl:7.4f}% | {median_pnl:10.4f}% | {max_pnl:7.4f}% | {min_pnl:7.4f}%")

# What about realistic costs (2-5 bps)?
print("\nCOST SENSITIVITY ANALYSIS:")
for cost_bps in [0.5, 1.0, 2.0, 3.0, 5.0]:
    total_cost = len(trades_df) * cost_bps / 100
    net = trades_df['pnl_pct'].sum() - total_cost
    print(f"{cost_bps:.1f} bps per trade: {net:7.2f}% net return")

# Analyze the profitable 2-5 bar trades
trades_2_5 = trades_df[trades_df['duration'].between(2, 5)]
print(f"\n2-5 BAR TRADES ANALYSIS ({len(trades_2_5)} trades):")
print(f"Win rate: {(trades_2_5['pnl_pct'] > 0).mean():.1%}")
print(f"Average PnL: {trades_2_5['pnl_pct'].mean():.4f}%")
print(f"Median PnL: {trades_2_5['pnl_pct'].median():.4f}%")

# Show distribution of these "winning" trades
print("\nPnL distribution for 2-5 bar trades:")
for pct in [10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(trades_2_5['pnl_pct'], pct)
    print(f"  {pct}th percentile: {val:.4f}%")

# How many are actually tiny wins?
tiny_wins = trades_2_5[(trades_2_5['pnl_pct'] > 0) & (trades_2_5['pnl_pct'] < 0.05)]
print(f"\nTiny wins (<0.05%): {len(tiny_wins)} trades ({len(tiny_wins)/len(trades_2_5)*100:.1f}% of 2-5 bar trades)")

# Dollar PnL analysis (assuming $10k per trade)
trade_size = 10000
print(f"\nDOLLAR P&L ANALYSIS (assuming ${trade_size} per trade):")
trades_df['dollar_pnl'] = trades_df['pnl_pct'] / 100 * trade_size
print(f"Average dollar P&L per trade: ${trades_df['dollar_pnl'].mean():.2f}")
print(f"Total dollar P&L: ${trades_df['dollar_pnl'].sum():.2f}")
print(f"After $1 commission per trade: ${trades_df['dollar_pnl'].sum() - len(trades_df):.2f}")