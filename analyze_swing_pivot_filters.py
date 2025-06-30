"""Analyze if trend filtering or VWAP respect would help swing_pivot_bounce"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the signal data
workspace = Path("workspaces/signal_generation_1c64d62f")
signal_file = workspace / "traces/SPY_1m/signals/swing_pivot_bounce/SPY_compiled_strategy_0.parquet"
signals = pd.read_parquet(signal_file)

# Load raw SPY data to calculate additional indicators
spy_data = pd.read_csv("./data/SPY_1m.csv")
spy_data['timestamp'] = pd.to_datetime(spy_data['timestamp'], utc=True)
spy_data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                         'Close': 'close', 'Volume': 'volume'}, inplace=True)

# Calculate trend indicators
spy_data['sma_50'] = spy_data['close'].rolling(50).mean()
spy_data['sma_200'] = spy_data['close'].rolling(200).mean()
spy_data['ema_20'] = spy_data['close'].ewm(span=20).mean()

# Calculate VWAP (resets daily)
spy_data['date'] = spy_data['timestamp'].dt.date
spy_data['typical_price'] = (spy_data['high'] + spy_data['low'] + spy_data['close']) / 3
spy_data['pv'] = spy_data['typical_price'] * spy_data['volume']

# Calculate cumulative values within each day
spy_data['cum_pv'] = spy_data.groupby('date')['pv'].cumsum()
spy_data['cum_volume'] = spy_data.groupby('date')['volume'].cumsum()
spy_data['vwap'] = spy_data['cum_pv'] / spy_data['cum_volume']

# Define trend conditions
spy_data['trend_up'] = (spy_data['close'] > spy_data['sma_50']) & (spy_data['sma_50'] > spy_data['sma_200'])
spy_data['trend_down'] = (spy_data['close'] < spy_data['sma_50']) & (spy_data['sma_50'] < spy_data['sma_200'])
spy_data['trend_neutral'] = ~(spy_data['trend_up'] | spy_data['trend_down'])

# VWAP position
spy_data['above_vwap'] = spy_data['close'] > spy_data['vwap']
spy_data['vwap_distance'] = (spy_data['close'] - spy_data['vwap']) / spy_data['vwap'] * 100

# Calculate trades with filters
trades = []
for i in range(1, len(signals)):
    prev_signal = signals.iloc[i-1]
    curr_signal = signals.iloc[i]
    
    if prev_signal['val'] != 0:
        if curr_signal['val'] == 0 or np.sign(curr_signal['val']) != np.sign(prev_signal['val']):
            entry_idx = prev_signal['idx']
            exit_idx = curr_signal['idx']
            
            # Get market conditions at entry
            entry_conditions = spy_data.iloc[entry_idx] if entry_idx < len(spy_data) else None
            
            if entry_conditions is not None and not pd.isna(entry_conditions['sma_200']):
                entry_price = prev_signal['px']
                exit_price = curr_signal['px']
                signal_type = prev_signal['val']
                
                log_return = np.log(exit_price / entry_price) * signal_type
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
                    'vwap_distance': entry_conditions['vwap_distance'],
                    'close_vs_ema20': (entry_conditions['close'] / entry_conditions['ema_20'] - 1) * 100
                })

trades_df = pd.DataFrame(trades)
print(f"Total trades with complete data: {len(trades_df)}")

# Analyze by trend
print("\n=== PERFORMANCE BY TREND ===")
for trend in ['trend_up', 'trend_down', 'trend_neutral']:
    trend_trades = trades_df[trades_df[trend]]
    if len(trend_trades) > 0:
        avg_return = trend_trades['pct_return'].mean()
        win_rate = (trend_trades['pct_return'] > 0).mean()
        total_return = np.exp(np.log(1 + trend_trades['pct_return']/100).sum()) - 1
        print(f"\n{trend.replace('_', ' ').title()}:")
        print(f"  Trades: {len(trend_trades)} ({len(trend_trades)/len(trades_df)*100:.1f}%)")
        print(f"  Avg return: {avg_return:.4f}%")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Total return: {total_return*100:.2f}%")

# Analyze by signal type and trend
print("\n=== LONG vs SHORT BY TREND ===")
for signal_type, signal_name in [(1, "Long"), (-1, "Short")]:
    print(f"\n{signal_name} trades:")
    signal_trades = trades_df[trades_df['signal'] == signal_type]
    
    for trend in ['trend_up', 'trend_down', 'trend_neutral']:
        filtered = signal_trades[signal_trades[trend]]
        if len(filtered) > 0:
            print(f"  {trend.replace('_', ' ').title()}: "
                  f"{len(filtered)} trades, "
                  f"avg {filtered['pct_return'].mean():.4f}%, "
                  f"win rate {(filtered['pct_return'] > 0).mean():.1%}")

# Analyze VWAP respect
print("\n=== PERFORMANCE BY VWAP POSITION ===")
above_vwap = trades_df[trades_df['above_vwap']]
below_vwap = trades_df[~trades_df['above_vwap']]

print(f"\nAbove VWAP ({len(above_vwap)} trades):")
print(f"  Avg return: {above_vwap['pct_return'].mean():.4f}%")
print(f"  Win rate: {(above_vwap['pct_return'] > 0).mean():.1%}")

print(f"\nBelow VWAP ({len(below_vwap)} trades):")
print(f"  Avg return: {below_vwap['pct_return'].mean():.4f}%")
print(f"  Win rate: {(below_vwap['pct_return'] > 0).mean():.1%}")

# Analyze by VWAP distance
print("\n=== PERFORMANCE BY VWAP DISTANCE ===")
trades_df['vwap_dist_bucket'] = pd.cut(trades_df['vwap_distance'], 
                                        bins=[-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf],
                                        labels=['Far Below', 'Near Below', 'Near VWAP', 'Near Above', 'Far Above'])

for bucket in trades_df['vwap_dist_bucket'].unique():
    if pd.notna(bucket):
        bucket_trades = trades_df[trades_df['vwap_dist_bucket'] == bucket]
        if len(bucket_trades) > 0:
            print(f"\n{bucket}:")
            print(f"  Trades: {len(bucket_trades)}")
            print(f"  Avg return: {bucket_trades['pct_return'].mean():.4f}%")
            print(f"  Win rate: {(bucket_trades['pct_return'] > 0).mean():.1%}")

# Best filter combinations
print("\n=== BEST FILTER COMBINATIONS ===")

# Test various filter combinations
filters_to_test = [
    ("Trend Up + Long only", (trades_df['trend_up']) & (trades_df['signal'] == 1)),
    ("Trend Down + Short only", (trades_df['trend_down']) & (trades_df['signal'] == -1)),
    ("Above VWAP + Long only", (trades_df['above_vwap']) & (trades_df['signal'] == 1)),
    ("Below VWAP + Short only", (~trades_df['above_vwap']) & (trades_df['signal'] == -1)),
    ("Trend aligned (Up+Long or Down+Short)", 
     ((trades_df['trend_up']) & (trades_df['signal'] == 1)) | 
     ((trades_df['trend_down']) & (trades_df['signal'] == -1))),
    ("Counter-trend (Up+Short or Down+Long)", 
     ((trades_df['trend_up']) & (trades_df['signal'] == -1)) | 
     ((trades_df['trend_down']) & (trades_df['signal'] == 1))),
    ("VWAP aligned (Above+Short or Below+Long)",
     ((trades_df['above_vwap']) & (trades_df['signal'] == -1)) |
     ((~trades_df['above_vwap']) & (trades_df['signal'] == 1))),
]

print("\nTesting filter combinations:")
for name, filter_mask in filters_to_test:
    filtered = trades_df[filter_mask]
    if len(filtered) > 0:
        avg_return = filtered['pct_return'].mean()
        win_rate = (filtered['pct_return'] > 0).mean()
        total_return = np.exp(np.log(1 + filtered['pct_return']/100).sum()) - 1
        trades_kept = len(filtered) / len(trades_df) * 100
        
        print(f"\n{name}:")
        print(f"  Trades: {len(filtered)} ({trades_kept:.1f}% kept)")
        print(f"  Avg return: {avg_return:.4f}%")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Total return: {total_return*100:.2f}%")
        print(f"  Return per trade in bps: {avg_return * 100:.2f}")