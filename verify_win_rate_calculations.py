"""Verify the win rate calculations from analyze_swing_pivot_optimized.py"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the signal data
workspace = Path("workspaces/signal_generation_1c64d62f")
signal_file = workspace / "traces/SPY_1m/signals/swing_pivot_bounce/SPY_compiled_strategy_0.parquet"
signals = pd.read_parquet(signal_file)

# Load raw SPY data
spy_data = pd.read_csv("./data/SPY_1m.csv")
spy_data['timestamp'] = pd.to_datetime(spy_data['timestamp'], utc=True)
spy_data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                         'Close': 'close', 'Volume': 'volume'}, inplace=True)

# Calculate all indicators we need
# Trend
spy_data['sma_50'] = spy_data['close'].rolling(50).mean()
spy_data['sma_200'] = spy_data['close'].rolling(200).mean()
spy_data['trend_up'] = (spy_data['close'] > spy_data['sma_50']) & (spy_data['sma_50'] > spy_data['sma_200'])
spy_data['trend_down'] = (spy_data['close'] < spy_data['sma_50']) & (spy_data['sma_50'] < spy_data['sma_200'])
spy_data['trend_neutral'] = ~(spy_data['trend_up'] | spy_data['trend_down'])

# VWAP
spy_data['date'] = spy_data['timestamp'].dt.date
spy_data['typical_price'] = (spy_data['high'] + spy_data['low'] + spy_data['close']) / 3
spy_data['pv'] = spy_data['typical_price'] * spy_data['volume']
spy_data['cum_pv'] = spy_data.groupby('date')['pv'].cumsum()
spy_data['cum_volume'] = spy_data.groupby('date')['volume'].cumsum()
spy_data['vwap'] = spy_data['cum_pv'] / spy_data['cum_volume']
spy_data['above_vwap'] = spy_data['close'] > spy_data['vwap']

# Volatility
spy_data['returns'] = spy_data['close'].pct_change()
spy_data['volatility_20'] = spy_data['returns'].rolling(20).std() * np.sqrt(390) * 100
spy_data['vol_percentile'] = spy_data['volatility_20'].rolling(252).rank(pct=True) * 100

# Calculate all trades with conditions
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
                    'trend_up': entry_conditions['trend_up'],
                    'trend_down': entry_conditions['trend_down'],
                    'trend_neutral': entry_conditions['trend_neutral'],
                    'above_vwap': entry_conditions['above_vwap'],
                    'vol_percentile': entry_conditions['vol_percentile']
                })

trades_df = pd.DataFrame(trades)

print("=== DETAILED WIN RATE VERIFICATION ===\n")

# Baseline
print("BASELINE (All Trades):")
print(f"Total trades: {len(trades_df)}")
winners = trades_df['pct_return'] > 0
print(f"Winners: {winners.sum()}")
print(f"Losers: {(~winners).sum()}")
print(f"Win rate: {winners.mean():.1%}")
print(f"Average return: {trades_df['pct_return'].mean():.4f}% ({trades_df['pct_return'].mean()*100:.2f} bps)")

# Filter 1: Counter-trend shorts in uptrends
filter1 = (trades_df['trend_up']) & (trades_df['signal'] == -1)
filtered1 = trades_df[filter1]
print(f"\n\nCOUNTER-TREND SHORTS IN UPTRENDS:")
print(f"Total trades: {len(filtered1)}")
if len(filtered1) > 0:
    winners1 = filtered1['pct_return'] > 0
    print(f"Winners: {winners1.sum()}")
    print(f"Losers: {(~winners1).sum()}")
    print(f"Win rate: {winners1.mean():.1%}")
    print(f"Average return: {filtered1['pct_return'].mean():.4f}% ({filtered1['pct_return'].mean()*100:.2f} bps)")

# Filter 2: Add high volatility
filter2 = filter1 & (trades_df['vol_percentile'] > 70)
filtered2 = trades_df[filter2]
print(f"\n\n+ HIGH VOLATILITY (>70th percentile):")
print(f"Total trades: {len(filtered2)}")
if len(filtered2) > 0:
    winners2 = filtered2['pct_return'] > 0
    print(f"Winners: {winners2.sum()}")
    print(f"Losers: {(~winners2).sum()}")
    print(f"Win rate: {winners2.mean():.1%}")
    print(f"Average return: {filtered2['pct_return'].mean():.4f}% ({filtered2['pct_return'].mean()*100:.2f} bps)")

# Balanced approach
balanced_filter = ((trades_df['trend_up']) & (trades_df['signal'] == -1) & (trades_df['vol_percentile'] > 50)) | \
                  ((trades_df['trend_up']) & (trades_df['signal'] == 1) & (trades_df['above_vwap']))
balanced_filtered = trades_df[balanced_filter]
print(f"\n\nBALANCED APPROACH:")
print(f"Total trades: {len(balanced_filtered)}")
if len(balanced_filtered) > 0:
    winners_bal = balanced_filtered['pct_return'] > 0
    print(f"Winners: {winners_bal.sum()}")
    print(f"Losers: {(~winners_bal).sum()}")
    print(f"Win rate: {winners_bal.mean():.1%}")
    print(f"Average return: {balanced_filtered['pct_return'].mean():.4f}% ({balanced_filtered['pct_return'].mean()*100:.2f} bps)")
    
    # Break down the components
    shorts_component = balanced_filtered[(balanced_filtered['signal'] == -1)]
    longs_component = balanced_filtered[(balanced_filtered['signal'] == 1)]
    
    print(f"\n  Shorts component: {len(shorts_component)} trades, {(shorts_component['pct_return'] > 0).mean():.1%} win rate")
    print(f"  Longs component: {len(longs_component)} trades, {(longs_component['pct_return'] > 0).mean():.1%} win rate")

# Let's also check the math
print("\n\n=== VERIFYING THE MATH ===")
print(f"\nFor balanced approach:")
print(f"If we have {len(shorts_component)} shorts at {(shorts_component['pct_return'] > 0).mean():.1%} win rate")
print(f"And {len(longs_component)} longs at {(longs_component['pct_return'] > 0).mean():.1%} win rate")
weighted_win_rate = (len(shorts_component) * (shorts_component['pct_return'] > 0).mean() + 
                     len(longs_component) * (longs_component['pct_return'] > 0).mean()) / len(balanced_filtered)
print(f"The weighted average win rate should be: {weighted_win_rate:.1%}")
print(f"Actual calculated win rate: {winners_bal.mean():.1%}")

# Check for any calculation errors
print("\n\n=== CHECKING FOR CALCULATION ERRORS ===")
print(f"Sum of winners + losers = {winners_bal.sum()} + {(~winners_bal).sum()} = {winners_bal.sum() + (~winners_bal).sum()}")
print(f"Total trades = {len(balanced_filtered)}")
print(f"Match? {(winners_bal.sum() + (~winners_bal).sum()) == len(balanced_filtered)}")