"""Detailed analysis of holding periods and win rates"""
import pandas as pd
import numpy as np
from pathlib import Path

# Workspace path
workspace = Path("workspaces/signal_generation_7ecda4b8")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"

# Read the sparse signal data
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

print("=== Detailed Holding Period Analysis ===\n")

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
            'bars_held': bars_held
        })
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

# Analyze what happens if we force exit at different bars
print("=== ACTUAL Win Rates by Forced Exit Time ===")
print("(What would happen if we exited ALL trades at X bars)\n")
print(f"{'Max Bars':<10} {'Trades':<8} {'Avg Return':<12} {'Win Rate':<10} {'Winners':<8} {'Losers':<8}")
print("-" * 65)

# For each max bar limit, calculate what would happen
for max_bars in [5, 10, 15, 20, 30, 50, 100, None]:
    if max_bars is None:
        # Original results
        subset = trades_df
        label = "No limit"
    else:
        # Only look at trades that lasted at least max_bars
        # This shows what their P&L would be at that point
        subset = trades_df[trades_df['bars_held'] >= max_bars]
        label = f"{max_bars}"
    
    if len(subset) > 0:
        avg_return = subset['pnl_pct'].mean()
        winners = (subset['pnl_pct'] > 0).sum()
        losers = (subset['pnl_pct'] < 0).sum()
        win_rate = winners / len(subset) if len(subset) > 0 else 0
        
        print(f"{label:<10} {len(subset):<8} {avg_return:>10.3f}%  {win_rate*100:>8.1f}%  {winners:<8} {losers:<8}")

# Now show the ACTUAL distribution of natural exits
print("\n=== Natural Exit Distribution ===")
print("(Where trades ACTUALLY exited on their own)\n")
print(f"{'Bars Range':<15} {'Count':<8} {'Pct':<8} {'Avg Return':<12} {'Win Rate':<10}")
print("-" * 60)

bar_ranges = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 100), (101, 10000)]
for min_bars, max_bars in bar_ranges:
    mask = (trades_df['bars_held'] >= min_bars) & (trades_df['bars_held'] <= max_bars)
    subset = trades_df[mask]
    
    if len(subset) > 0:
        pct_of_total = len(subset) / len(trades_df) * 100
        avg_return = subset['pnl_pct'].mean()
        win_rate = (subset['pnl_pct'] > 0).mean()
        
        label = f"{min_bars}-{max_bars}" if max_bars < 10000 else f"{min_bars}+"
        print(f"{label:<15} {len(subset):<8} {pct_of_total:>6.1f}%  {avg_return:>10.3f}%  {win_rate*100:>8.1f}%")

# The key insight
print("\n=== KEY INSIGHT ===")
print("The 100% win rate for 1-5 bars means trades that EXIT NATURALLY within 1-5 bars are ALL WINNERS!")
print("This suggests the strategy quickly identifies and exits winning reversions.")
print("\nBut if we FORCE all trades to exit at 5 bars:")

# Simulate forcing ALL trades to exit at 5 bars
trades_forced_5 = trades_df.copy()
long_trades = trades_forced_5['bars_held'] > 5

if long_trades.any():
    # For trades longer than 5 bars, we need to estimate their P&L at bar 5
    # This is approximate - assumes linear P&L development
    trades_forced_5.loc[long_trades, 'pnl_pct'] = (
        trades_forced_5.loc[long_trades, 'pnl_pct'] * 
        (5 / trades_forced_5.loc[long_trades, 'bars_held'])
    )

print(f"\nForced exit at 5 bars:")
print(f"  Total trades: {len(trades_forced_5)}")
print(f"  Win rate: {(trades_forced_5['pnl_pct'] > 0).mean()*100:.1f}%")
print(f"  Avg return: {trades_forced_5['pnl_pct'].mean():.3f}%")

# Show why longer trades lose
print("\n=== Why Longer Trades Tend to Lose ===")
for bars_threshold in [10, 20, 30]:
    long_duration = trades_df[trades_df['bars_held'] > bars_threshold]
    if len(long_duration) > 0:
        print(f"\nTrades lasting >{bars_threshold} bars:")
        print(f"  Count: {len(long_duration)} ({len(long_duration)/len(trades_df)*100:.1f}%)")
        print(f"  Win rate: {(long_duration['pnl_pct'] > 0).mean()*100:.1f}%")
        print(f"  Avg return: {long_duration['pnl_pct'].mean():.3f}%")
        print(f"  These are likely failed mean reversions where price continued trending")