"""Investigate the 100% win rate for 1-5 bar trades"""
import pandas as pd
import numpy as np
from pathlib import Path

# Workspace path
workspace = Path("workspaces/signal_generation_7ecda4b8")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"

# Read the sparse signal data
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

print("=== Investigating 1-5 Bar Trades with 100% Win Rate ===\n")

# Convert sparse signals to trades
trades = []
current_position = 0

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    new_signal = row['val']
    
    # Close existing position if changing
    if current_position != 0 and new_signal != current_position:
        entry_idx = i - 1
        entry_row = signals_df.iloc[entry_idx]
        
        # Calculate PnL
        entry_price = entry_row['px']
        exit_price = row['px']
        pnl_pct = (exit_price / entry_price - 1) * current_position * 100
        bars_held = row['idx'] - entry_row['idx']
        
        trades.append({
            'entry_time': entry_row['ts'],
            'exit_time': row['ts'],
            'direction': 'long' if current_position > 0 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'bars_held': bars_held,
            'entry_signal': current_position,
            'exit_signal': new_signal
        })
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

# Focus on 1-5 bar trades
short_trades = trades_df[trades_df['bars_held'] <= 5]

print(f"Total 1-5 bar trades: {len(short_trades)}")
print(f"Win rate: {(short_trades['pnl_pct'] > 0).mean()*100:.1f}%")
print(f"Average return: {short_trades['pnl_pct'].mean():.3f}%")

# Show some examples
print("\n=== Sample 1-5 Bar Trades ===")
print(f"{'Direction':<10} {'Bars':<6} {'PnL %':<10} {'Entry Price':<12} {'Exit Price':<12}")
print("-" * 60)

for idx, trade in short_trades.head(20).iterrows():
    print(f"{trade['direction']:<10} {trade['bars_held']:<6} {trade['pnl_pct']:>8.3f}%  ${trade['entry_price']:<11.2f} ${trade['exit_price']:<11.2f}")

# Check for any losers
losers = short_trades[short_trades['pnl_pct'] < 0]
print(f"\n=== Losing Trades in 1-5 Bar Range ===")
print(f"Count: {len(losers)}")

if len(losers) > 0:
    print("\nFirst 10 losing trades:")
    for idx, trade in losers.head(10).iterrows():
        print(f"{trade['direction']:<10} {trade['bars_held']:<6} {trade['pnl_pct']:>8.3f}%  ${trade['entry_price']:<11.2f} ${trade['exit_price']:<11.2f}")

# Analyze by bar count
print("\n=== Breakdown by Exact Bar Count ===")
for bars in range(1, 6):
    subset = trades_df[trades_df['bars_held'] == bars]
    if len(subset) > 0:
        winners = (subset['pnl_pct'] > 0).sum()
        losers = (subset['pnl_pct'] < 0).sum()
        avg_return = subset['pnl_pct'].mean()
        print(f"{bars} bar: {len(subset)} trades, {winners} winners, {losers} losers, avg: {avg_return:.3f}%")

# Check signal transitions
print("\n=== Signal Transition Analysis for Short Trades ===")
print("(What causes these quick exits?)")
transitions = short_trades.groupby(['entry_signal', 'exit_signal']).size()
print(transitions)

# Look at the actual signal changes
print("\n=== Understanding the Signal Logic ===")
print("The strategy uses three signals: 1 (long), 0 (flat), -1 (short)")
print("Quick exits happen when:")
print("- Long (1) → Flat (0): Price moves back to middle of bands")
print("- Short (-1) → Flat (0): Price moves back to middle of bands")
print("- Long (1) → Short (-1): Price quickly reverses from lower to upper band")
print("- Short (-1) → Long (1): Price quickly reverses from upper to lower band")

# Statistical check
print("\n=== Statistical Verification ===")
print(f"Mean return of 1-5 bar trades: {short_trades['pnl_pct'].mean():.4f}%")
print(f"Std dev: {short_trades['pnl_pct'].std():.4f}%")
print(f"Min return: {short_trades['pnl_pct'].min():.4f}%")
print(f"Max return: {short_trades['pnl_pct'].max():.4f}%")
print(f"Median return: {short_trades['pnl_pct'].median():.4f}%")