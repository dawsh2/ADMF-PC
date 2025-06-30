"""Analyze the 15-minute swing pivot bounce workspace"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_ffbf0538")
signal_file = workspace / "traces/SPY_15m_1m/signals/swing_pivot_bounce/SPY_15m_compiled_strategy_0.parquet"

print("=== 15-MINUTE SWING PIVOT BOUNCE ANALYSIS ===\n")

# Load signals
signals = pd.read_parquet(signal_file)
print(f"Total signal changes: {len(signals)}")
print(f"Signal columns: {signals.columns.tolist()}")

# Load 15-minute SPY data
spy_15m = pd.read_csv("./data/SPY_15m.csv")
spy_15m['timestamp'] = pd.to_datetime(spy_15m['timestamp'])
spy_15m.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                        'Close': 'close', 'Volume': 'volume'}, inplace=True)

print(f"\nData period: {spy_15m['timestamp'].min()} to {spy_15m['timestamp'].max()}")
print(f"Total 15-minute bars: {len(spy_15m)}")

# Use appropriate subset
max_idx = min(signals['idx'].max() + 5, len(spy_15m))
spy_subset = spy_15m.iloc[:max_idx].copy()

# Calculate indicators
spy_subset['returns'] = spy_subset['close'].pct_change()
spy_subset['volatility_20'] = spy_subset['returns'].rolling(20).std() * np.sqrt(26) * 100  # 26 bars per day for 15m
spy_subset['vol_percentile'] = spy_subset['volatility_20'].rolling(window=26*20).rank(pct=True) * 100

# Trend
spy_subset['sma_50'] = spy_subset['close'].rolling(50).mean()
spy_subset['sma_200'] = spy_subset['close'].rolling(200).mean()
spy_subset['trend_up'] = (spy_subset['close'] > spy_subset['sma_50']) & (spy_subset['sma_50'] > spy_subset['sma_200'])

# Collect trades
trades = []
entry_data = None

for i in range(len(signals)):
    curr = signals.iloc[i]
    
    if entry_data is None and curr['val'] != 0:
        entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
    
    elif entry_data is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])):
        if entry_data['idx'] < len(spy_subset) and curr['idx'] < len(spy_subset):
            entry_conditions = spy_subset.iloc[entry_data['idx']]
            
            pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
            duration = curr['idx'] - entry_data['idx']
            
            trade = {
                'pct_return': pct_return,
                'direction': 'short' if entry_data['signal'] < 0 else 'long',
                'duration': duration,
                'trend_up': entry_conditions.get('trend_up', False),
                'vol_percentile': entry_conditions.get('vol_percentile', 50)
            }
            trades.append(trade)
        
        if curr['val'] != 0:
            entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
        else:
            entry_data = None

trades_df = pd.DataFrame(trades)
total_days = 5673 / 26  # From metadata, 26 bars per day for 15m

print(f"\n\nTotal trades collected: {len(trades_df)}")
print(f"Trading days: {total_days:.1f}")
print(f"Average trades per day: {len(trades_df)/total_days:.2f}")

# Overall performance
print(f"\n=== OVERALL PERFORMANCE ===")
print(f"Average return per trade: {trades_df['pct_return'].mean():.2f} bps")
print(f"Total return: {trades_df['pct_return'].sum():.2f}%")
print(f"Win rate: {(trades_df['pct_return'] > 0).mean():.1%}")
print(f"Average duration: {trades_df['duration'].mean():.1f} bars (15-min bars)")
print(f"Average hold time: {trades_df['duration'].mean() * 15:.0f} minutes")

# By direction
print(f"\n=== PERFORMANCE BY DIRECTION ===")
for direction in ['long', 'short']:
    dir_trades = trades_df[trades_df['direction'] == direction]
    if len(dir_trades) > 0:
        print(f"\n{direction.upper()}: {len(dir_trades)} trades")
        print(f"  Avg return: {dir_trades['pct_return'].mean():.2f} bps")
        print(f"  Win rate: {(dir_trades['pct_return'] > 0).mean():.1%}")

# Test key filters
print(f"\n=== FILTER ANALYSIS ===")

# Volatility filters
for threshold in [60, 70, 80]:
    vol_filter = trades_df[trades_df['vol_percentile'] > threshold]
    if len(vol_filter) > 5:
        print(f"\nVol > {threshold}th percentile: {len(vol_filter)} trades")
        print(f"  Avg return: {vol_filter['pct_return'].mean():.2f} bps")
        print(f"  Trades/day: {len(vol_filter)/total_days:.2f}")

# Counter-trend shorts
ct_shorts = trades_df[(trades_df['trend_up']) & (trades_df['direction'] == 'short')]
if len(ct_shorts) > 0:
    print(f"\nCounter-trend shorts in uptrend: {len(ct_shorts)} trades")
    print(f"  Avg return: {ct_shorts['pct_return'].mean():.2f} bps")
    print(f"  Trades/day: {len(ct_shorts)/total_days:.2f}")

# Distribution
print(f"\n=== RETURN DISTRIBUTION ===")
print(f"Min: {trades_df['pct_return'].min():.2f} bps")
print(f"25%: {trades_df['pct_return'].quantile(0.25):.2f} bps")
print(f"50%: {trades_df['pct_return'].median():.2f} bps")
print(f"75%: {trades_df['pct_return'].quantile(0.75):.2f} bps")
print(f"Max: {trades_df['pct_return'].max():.2f} bps")

print(f"\n\nCONCLUSION:")
print(f"15-minute swing pivot bounce trades {len(trades_df)/total_days:.2f} times per day")
print(f"This is likely too infrequent for your goal of 2-3+ trades per day")
print(f"The 5-minute timeframe (2.8 trades/day) remains the better choice")