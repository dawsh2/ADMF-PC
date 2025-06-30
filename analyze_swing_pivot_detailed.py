"""Detailed analysis of swing_pivot_bounce per-trade returns"""
import pandas as pd
import numpy as np
from pathlib import Path

# Workspace path
workspace = Path("workspaces/signal_generation_1c64d62f")
signal_file = workspace / "traces/SPY_1m/signals/swing_pivot_bounce/SPY_compiled_strategy_0.parquet"

# Load sparse signal data
signals = pd.read_parquet(signal_file)

# Calculate trades
trades = []
for i in range(1, len(signals)):
    prev_signal = signals.iloc[i-1]
    curr_signal = signals.iloc[i]
    
    if prev_signal['val'] != 0:
        if curr_signal['val'] == 0 or np.sign(curr_signal['val']) != np.sign(prev_signal['val']):
            entry_price = prev_signal['px']
            exit_price = curr_signal['px']
            signal_type = prev_signal['val']
            
            # Calculate returns
            log_return = np.log(exit_price / entry_price) * signal_type
            pct_return = (exit_price / entry_price - 1) * signal_type * 100
            
            trades.append({
                'entry_bar': prev_signal['idx'],
                'exit_bar': curr_signal['idx'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'signal': signal_type,
                'bars_held': curr_signal['idx'] - prev_signal['idx'],
                'log_return': log_return,
                'pct_return': pct_return,
                'dollar_pnl': (exit_price - entry_price) * signal_type
            })

trades_df = pd.DataFrame(trades)

print("=== DETAILED PER-TRADE RETURNS ===")
print(f"Total trades: {len(trades_df)}")
print(f"\nPer-trade return statistics:")
print(f"Average return: {trades_df['pct_return'].mean():.4f}%")
print(f"Median return: {trades_df['pct_return'].median():.4f}%")
print(f"Std deviation: {trades_df['pct_return'].std():.4f}%")

print(f"\nIn basis points (bps):")
print(f"Average return: {trades_df['pct_return'].mean() * 100:.2f} bps")
print(f"Median return: {trades_df['pct_return'].median() * 100:.2f} bps")

print(f"\nDollar P&L per trade (assuming $1 position):")
print(f"Average: ${trades_df['dollar_pnl'].mean():.6f}")
print(f"Total: ${trades_df['dollar_pnl'].sum():.4f}")

# Break down by long/short
long_trades = trades_df[trades_df['signal'] > 0]
short_trades = trades_df[trades_df['signal'] < 0]

print(f"\n=== LONG TRADES ===")
print(f"Count: {len(long_trades)}")
print(f"Average return: {long_trades['pct_return'].mean():.4f}%")
print(f"Average return (bps): {long_trades['pct_return'].mean() * 100:.2f} bps")
print(f"Total return: {long_trades['pct_return'].sum():.2f}%")

print(f"\n=== SHORT TRADES ===")
print(f"Count: {len(short_trades)}")
print(f"Average return: {short_trades['pct_return'].mean():.4f}%")
print(f"Average return (bps): {short_trades['pct_return'].mean() * 100:.2f} bps")
print(f"Total return: {short_trades['pct_return'].sum():.2f}%")

# Distribution analysis
print(f"\n=== RETURN DISTRIBUTION ===")
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    value = np.percentile(trades_df['pct_return'], p)
    print(f"{p}th percentile: {value:.4f}%")

# Expected value calculation
print(f"\n=== EXPECTED VALUE ===")
win_rate = (trades_df['pct_return'] > 0).mean()
avg_win = trades_df[trades_df['pct_return'] > 0]['pct_return'].mean()
avg_loss = trades_df[trades_df['pct_return'] <= 0]['pct_return'].mean()
expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss
print(f"Win rate: {win_rate:.1%}")
print(f"Average win: {avg_win:.4f}%")
print(f"Average loss: {avg_loss:.4f}%")
print(f"Expected value per trade: {expected_value:.4f}%")
print(f"Expected value (bps): {expected_value * 100:.2f} bps")

# Kelly criterion estimate
if avg_win > 0 and avg_loss < 0:
    kelly_fraction = (win_rate * avg_win + avg_loss) / avg_win
    print(f"\nKelly fraction estimate: {kelly_fraction:.1%}")