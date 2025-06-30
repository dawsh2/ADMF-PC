"""Verify win rate calculations in swing pivot analysis"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the same data as the main script
workspace = Path("workspaces/signal_generation_1c64d62f")
signal_file = workspace / "traces/SPY_1m/signals/swing_pivot_bounce/SPY_compiled_strategy_0.parquet"
signals = pd.read_parquet(signal_file)

spy_data = pd.read_csv("./data/SPY_1m.csv")
spy_data['timestamp'] = pd.to_datetime(spy_data['timestamp'], utc=True)
spy_data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                         'Close': 'close', 'Volume': 'volume'}, inplace=True)

# Calculate trend indicators
spy_data['sma_50'] = spy_data['close'].rolling(50).mean()
spy_data['sma_200'] = spy_data['close'].rolling(200).mean()
spy_data['trend_up'] = (spy_data['close'] > spy_data['sma_50']) & (spy_data['sma_50'] > spy_data['sma_200'])

# Extract trades
trades = []
for i in range(1, len(signals)):
    prev_signal = signals.iloc[i-1]
    curr_signal = signals.iloc[i]
    
    if prev_signal['val'] != 0:
        if curr_signal['val'] == 0 or np.sign(curr_signal['val']) != np.sign(prev_signal['val']):
            entry_idx = prev_signal['idx']
            exit_idx = curr_signal['idx']
            
            if entry_idx < len(spy_data) and not pd.isna(spy_data.iloc[entry_idx]['sma_200']):
                entry_conditions = spy_data.iloc[entry_idx]
                entry_price = prev_signal['px']
                exit_price = curr_signal['px']
                signal_type = prev_signal['val']
                
                pct_return = (exit_price / entry_price - 1) * signal_type * 100
                
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': exit_idx,
                    'signal': signal_type,
                    'pct_return': pct_return,
                    'trend_up': entry_conditions['trend_up']
                })

trades_df = pd.DataFrame(trades)

print("=== WIN RATE VERIFICATION ===")
print(f"\nTotal trades: {len(trades_df)}")
print(f"Winning trades (return > 0): {(trades_df['pct_return'] > 0).sum()}")
print(f"Losing trades (return <= 0): {(trades_df['pct_return'] <= 0).sum()}")
print(f"Win rate (manual calc): {(trades_df['pct_return'] > 0).sum() / len(trades_df) * 100:.1f}%")
print(f"Win rate (mean): {(trades_df['pct_return'] > 0).mean() * 100:.1f}%")

# Check filter 1: Counter-trend shorts in uptrends
filter1 = (trades_df['trend_up']) & (trades_df['signal'] == -1)
filtered1 = trades_df[filter1]

print(f"\n=== FILTER 1: Counter-trend shorts in uptrends ===")
print(f"Total trades after filter: {len(filtered1)}")
print(f"Winning trades: {(filtered1['pct_return'] > 0).sum()}")
print(f"Losing trades: {(filtered1['pct_return'] <= 0).sum()}")
print(f"Win rate: {(filtered1['pct_return'] > 0).mean() * 100:.1f}%")

# Show distribution of returns
print(f"\n=== RETURN DISTRIBUTION ===")
print(f"Baseline avg return: {trades_df['pct_return'].mean():.4f}%")
print(f"Filter 1 avg return: {filtered1['pct_return'].mean():.4f}%")
print(f"\nBaseline return distribution:")
print(trades_df['pct_return'].describe())
print(f"\nFilter 1 return distribution:")
print(filtered1['pct_return'].describe())

# Verify the balanced filter calculation
balanced_filter = ((trades_df['trend_up']) & (trades_df['signal'] == -1)) | \
                  ((trades_df['trend_up']) & (trades_df['signal'] == 1))
balanced_filtered = trades_df[balanced_filter]

print(f"\n=== BALANCED FILTER VERIFICATION ===")
print(f"Total trades: {len(balanced_filtered)}")
print(f"Win rate: {(balanced_filtered['pct_return'] > 0).mean() * 100:.1f}%")

# Break down by component
shorts_in_uptrend = trades_df[(trades_df['trend_up']) & (trades_df['signal'] == -1)]
longs_in_uptrend = trades_df[(trades_df['trend_up']) & (trades_df['signal'] == 1)]

print(f"\nComponent breakdown:")
print(f"Shorts in uptrend: {len(shorts_in_uptrend)} trades, {(shorts_in_uptrend['pct_return'] > 0).mean() * 100:.1f}% win rate")
print(f"Longs in uptrend: {len(longs_in_uptrend)} trades, {(longs_in_uptrend['pct_return'] > 0).mean() * 100:.1f}% win rate")
print(f"Combined: {len(shorts_in_uptrend) + len(longs_in_uptrend)} trades")