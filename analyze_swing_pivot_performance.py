"""Analyze swing_pivot_bounce strategy performance using sparse trace analysis"""
import pandas as pd
import numpy as np
from pathlib import Path

# Workspace path
workspace = Path("workspaces/signal_generation_1c64d62f")
signal_file = workspace / "traces/SPY_1m/signals/swing_pivot_bounce/SPY_compiled_strategy_0.parquet"

# Load sparse signal data
signals = pd.read_parquet(signal_file)
print(f"Total signal changes: {len(signals)}")
print(f"\nSignal distribution:")
print(signals['val'].value_counts().sort_index())

# Calculate trades (signal changes that open/close positions)
trades = []
for i in range(1, len(signals)):
    prev_signal = signals.iloc[i-1]
    curr_signal = signals.iloc[i]
    
    # Position closed (prev != 0, curr == 0 or opposite sign)
    if prev_signal['val'] != 0:
        if curr_signal['val'] == 0 or np.sign(curr_signal['val']) != np.sign(prev_signal['val']):
            entry_price = prev_signal['px']
            exit_price = curr_signal['px']
            signal_type = prev_signal['val']
            
            # Calculate log return
            log_return = np.log(exit_price / entry_price) * signal_type
            
            trades.append({
                'entry_bar': prev_signal['idx'],
                'exit_bar': curr_signal['idx'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'signal': signal_type,
                'bars_held': curr_signal['idx'] - prev_signal['idx'],
                'log_return': log_return,
                'pct_return': (np.exp(log_return) - 1) * 100
            })

trades_df = pd.DataFrame(trades)
print(f"\n=== TRADE ANALYSIS ===")
print(f"Total trades: {len(trades_df)}")
print(f"Long trades: {len(trades_df[trades_df['signal'] > 0])}")
print(f"Short trades: {len(trades_df[trades_df['signal'] < 0])}")

# Performance metrics
if len(trades_df) > 0:
    print(f"\n=== PERFORMANCE METRICS (Gross) ===")
    total_log_return = trades_df['log_return'].sum()
    total_pct_return = (np.exp(total_log_return) - 1) * 100
    print(f"Total return: {total_pct_return:.2f}%")
    print(f"Average return per trade: {trades_df['pct_return'].mean():.3f}%")
    print(f"Win rate: {(trades_df['log_return'] > 0).sum() / len(trades_df) * 100:.1f}%")
    print(f"Average bars held: {trades_df['bars_held'].mean():.1f}")
    
    # Long vs Short performance
    long_trades = trades_df[trades_df['signal'] > 0]
    short_trades = trades_df[trades_df['signal'] < 0]
    
    if len(long_trades) > 0:
        long_return = (np.exp(long_trades['log_return'].sum()) - 1) * 100
        print(f"\nLong trades return: {long_return:.2f}%")
        print(f"Long win rate: {(long_trades['log_return'] > 0).sum() / len(long_trades) * 100:.1f}%")
    
    if len(short_trades) > 0:
        short_return = (np.exp(short_trades['log_return'].sum()) - 1) * 100
        print(f"Short trades return: {short_return:.2f}%")
        print(f"Short win rate: {(short_trades['log_return'] > 0).sum() / len(short_trades) * 100:.1f}%")
    
    # Apply execution costs
    print(f"\n=== PERFORMANCE WITH EXECUTION COSTS ===")
    for cost_pct in [0.1, 0.2, 0.5, 1.0]:
        cost_multiplier = 1 - (cost_pct / 100)
        net_log_returns = trades_df['log_return'] * cost_multiplier
        net_total_return = (np.exp(net_log_returns.sum()) - 1) * 100
        print(f"{cost_pct}% cost: {net_total_return:.2f}% return")
    
    # Trade duration analysis
    print(f"\n=== TRADE DURATION ANALYSIS ===")
    print(f"Min bars held: {trades_df['bars_held'].min()}")
    print(f"Max bars held: {trades_df['bars_held'].max()}")
    print(f"Median bars held: {trades_df['bars_held'].median()}")
    
    # Distribution of returns
    print(f"\n=== RETURN DISTRIBUTION ===")
    print(f"Best trade: {trades_df['pct_return'].max():.2f}%")
    print(f"Worst trade: {trades_df['pct_return'].min():.2f}%")
    print(f"Std dev: {trades_df['pct_return'].std():.2f}%")
    
    # Winners vs Losers analysis
    winners = trades_df[trades_df['log_return'] > 0]
    losers = trades_df[trades_df['log_return'] <= 0]
    
    if len(winners) > 0 and len(losers) > 0:
        print(f"\n=== WINNERS VS LOSERS ===")
        print(f"Avg winner: {winners['pct_return'].mean():.3f}%")
        print(f"Avg loser: {losers['pct_return'].mean():.3f}%")
        print(f"Win/Loss ratio: {abs(winners['pct_return'].mean() / losers['pct_return'].mean()):.2f}")
        print(f"Avg winner bars: {winners['bars_held'].mean():.1f}")
        print(f"Avg loser bars: {losers['bars_held'].mean():.1f}")
        print(f"Loser/Winner hold ratio: {losers['bars_held'].mean() / winners['bars_held'].mean():.2f}x")

else:
    print("\nNo trades found in the data!")