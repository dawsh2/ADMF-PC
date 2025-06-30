#!/usr/bin/env python3
"""Compare actual performance between universal analysis and execution."""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
universal_run = Path("/Users/daws/ADMF-PC/config/bollinger/results/20250627_185448")
execution_run = Path("/Users/daws/ADMF-PC/config/bollinger/results/20250628_194812")

print("PERFORMANCE COMPARISON")
print("=" * 80)

# 1. Universal Analysis Results (from optimization)
print("\n1. UNIVERSAL ANALYSIS (Notebook Optimization):")
print("-" * 40)
# These are the reported values from the notebook
print("Strategy: bollinger_bands (period=10, std_dev=2.0)")
print("Original performance:")
print("  Sharpe: -1.76")
print("  Return: -6.0%")
print("  Win rate: 47.9%")
print("  Avg return/trade: -0.006%")
print("  Number of trades: 1036")
print("\nWith 0.075% stop / 0.1% target:")
print("  Sharpe: 14.30")
print("  Return: 40.3%")
print("  Win rate: 66.5% (calculated in notebook)")
print("  Stop exits: 159")
print("  Target exits: 523")
print("  Signal exits: 354")

# 2. Execution Results
print("\n\n2. EXECUTION RESULTS (Event-driven system):")
print("-" * 40)

# Load position data
pos_open = pd.read_parquet(execution_run / "traces/portfolio/positions_open/positions_open.parquet")
pos_close = pd.read_parquet(execution_run / "traces/portfolio/positions_close/positions_close.parquet")

# Calculate returns
if 'position_id' in pos_open.columns and 'position_id' in pos_close.columns:
    trades = pd.merge(
        pos_open[['position_id', 'entry_price', 'quantity']],
        pos_close[['position_id', 'exit_price', 'exit_type', 'realized_pnl']],
        on='position_id'
    )
else:
    # Fallback
    trades = pd.DataFrame({
        'entry_price': pos_open['entry_price'].values if 'entry_price' in pos_open.columns else pos_open['px'].values,
        'exit_price': pos_close['exit_price'].values if 'exit_price' in pos_close.columns else pos_close['px'].values,
        'quantity': pos_open['quantity'].values if 'quantity' in pos_open.columns else 1,
        'exit_type': pos_close['exit_type'].values if 'exit_type' in pos_close.columns else 'unknown'
    })

# Calculate returns
trades['return_pct'] = ((trades['exit_price'] - trades['entry_price']) / trades['entry_price'])

# Metrics
total_trades = len(trades)
winning_trades = (trades['return_pct'] > 0).sum()
win_rate = winning_trades / total_trades

# Calculate cumulative return
cumulative_return = (1 + trades['return_pct']).prod() - 1
avg_return = trades['return_pct'].mean()

# Sharpe
if trades['return_pct'].std() > 0:
    sharpe = avg_return / trades['return_pct'].std() * np.sqrt(252)
else:
    sharpe = 0

print(f"Number of trades: {total_trades}")
print(f"Win rate: {win_rate*100:.1f}%")
print(f"Avg return/trade: {avg_return*100:.3f}%")
print(f"Total return: {cumulative_return*100:.1f}%")
print(f"Sharpe ratio: {sharpe:.2f}")

# Exit breakdown
if 'exit_type' in trades.columns:
    exit_counts = trades['exit_type'].value_counts()
    print("\nExit breakdown:")
    for exit_type, count in exit_counts.items():
        print(f"  {exit_type}: {count}")

# 3. COMPARISON
print("\n\n3. DISCREPANCY ANALYSIS:")
print("-" * 40)
print(f"Total return difference: {40.3 - cumulative_return*100:.1f}%")
print(f"Win rate difference: {66.5 - win_rate*100:.1f}%")
print(f"Sharpe difference: {14.30 - sharpe:.2f}")

# Check if stops/targets are being applied correctly
stop_exits = trades[trades['exit_type'] == 'stop_loss'] if 'exit_type' in trades.columns else pd.DataFrame()
target_exits = trades[trades['exit_type'] == 'take_profit'] if 'exit_type' in trades.columns else pd.DataFrame()

if len(stop_exits) > 0:
    avg_stop_loss = stop_exits['return_pct'].mean()
    print(f"\nAverage stop loss return: {avg_stop_loss*100:.3f}%")
    print(f"Expected stop loss: -0.075%")
    print(f"Match? {'YES' if abs(avg_stop_loss*100 + 0.075) < 0.01 else 'NO'}")

if len(target_exits) > 0:
    avg_take_profit = target_exits['return_pct'].mean()
    print(f"\nAverage take profit return: {avg_take_profit*100:.3f}%")
    print(f"Expected take profit: 0.100%")
    print(f"Match? {'YES' if abs(avg_take_profit*100 - 0.100) < 0.01 else 'NO'}")

# Look for any trades that exceed expected bounds
large_wins = trades[trades['return_pct'] > 0.001]
large_losses = trades[trades['return_pct'] < -0.00075]

print(f"\nTrades exceeding expected bounds:")
print(f"Wins > 0.1%: {len(large_wins)}")
print(f"Losses < -0.075%: {len(large_losses)}")

if len(large_wins) > 0:
    print(f"\nLargest wins: {large_wins['return_pct'].nlargest(5).values * 100}")
if len(large_losses) > 0:
    print(f"Largest losses: {large_losses['return_pct'].nsmallest(5).values * 100}")