"""Analyze swing_pivot_bounce performance under different volatility and ranging conditions"""
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

# Calculate volatility metrics
spy_data['returns'] = spy_data['close'].pct_change()
spy_data['volatility_20'] = spy_data['returns'].rolling(20).std() * np.sqrt(390) * 100  # Annualized
spy_data['volatility_50'] = spy_data['returns'].rolling(50).std() * np.sqrt(390) * 100
spy_data['atr_20'] = spy_data['high'].rolling(20).max() - spy_data['low'].rolling(20).min()
spy_data['atr_pct'] = spy_data['atr_20'] / spy_data['close'] * 100

# Calculate range metrics
spy_data['high_20'] = spy_data['high'].rolling(20).max()
spy_data['low_20'] = spy_data['low'].rolling(20).min()
spy_data['range_20'] = (spy_data['high_20'] - spy_data['low_20']) / spy_data['close'] * 100

# Moving averages for trend/range detection
spy_data['sma_20'] = spy_data['close'].rolling(20).mean()
spy_data['sma_50'] = spy_data['close'].rolling(50).mean()
spy_data['sma_200'] = spy_data['close'].rolling(200).mean()

# ADX for trend strength (proxy calculation)
spy_data['high_diff'] = spy_data['high'].diff()
spy_data['low_diff'] = -spy_data['low'].diff()
spy_data['hl_diff'] = spy_data['high'] - spy_data['low']
spy_data['tr'] = spy_data[['high_diff', 'low_diff', 'hl_diff']].max(axis=1)
spy_data['atr_14'] = spy_data['tr'].rolling(14).mean()

# Ranging market detection
spy_data['price_vs_sma20'] = (spy_data['close'] - spy_data['sma_20']) / spy_data['sma_20'] * 100
spy_data['is_ranging'] = spy_data['price_vs_sma20'].rolling(50).std() < 0.5  # Low deviation from MA

# Volatility percentiles
spy_data['vol_percentile'] = spy_data['volatility_20'].rolling(252).rank(pct=True) * 100

# Define volatility regimes
spy_data['low_vol'] = spy_data['vol_percentile'] < 30
spy_data['med_vol'] = (spy_data['vol_percentile'] >= 30) & (spy_data['vol_percentile'] < 70)
spy_data['high_vol'] = spy_data['vol_percentile'] >= 70

# Calculate trades with volatility data
trades = []
for i in range(1, len(signals)):
    prev_signal = signals.iloc[i-1]
    curr_signal = signals.iloc[i]
    
    if prev_signal['val'] != 0:
        if curr_signal['val'] == 0 or np.sign(curr_signal['val']) != np.sign(prev_signal['val']):
            entry_idx = prev_signal['idx']
            exit_idx = curr_signal['idx']
            
            if entry_idx < len(spy_data) and not pd.isna(spy_data.iloc[entry_idx]['volatility_50']):
                entry_conditions = spy_data.iloc[entry_idx]
                entry_price = prev_signal['px']
                exit_price = curr_signal['px']
                signal_type = prev_signal['val']
                
                pct_return = (exit_price / entry_price - 1) * signal_type * 100
                
                trades.append({
                    'signal': signal_type,
                    'pct_return': pct_return,
                    'volatility': entry_conditions['volatility_20'],
                    'vol_percentile': entry_conditions['vol_percentile'],
                    'low_vol': entry_conditions['low_vol'],
                    'med_vol': entry_conditions['med_vol'],
                    'high_vol': entry_conditions['high_vol'],
                    'is_ranging': entry_conditions['is_ranging'],
                    'range_20': entry_conditions['range_20'],
                    'atr_pct': entry_conditions['atr_pct']
                })

trades_df = pd.DataFrame(trades)
print(f"Total trades with volatility data: {len(trades_df)}")

# Performance by volatility regime
print("\n=== PERFORMANCE BY VOLATILITY REGIME ===")
for vol_regime in ['low_vol', 'med_vol', 'high_vol']:
    regime_trades = trades_df[trades_df[vol_regime]]
    if len(regime_trades) > 0:
        print(f"\n{vol_regime.replace('_', ' ').upper()} ({len(regime_trades)} trades):")
        avg_return = regime_trades['pct_return'].mean()
        total_return = np.exp(np.log(1 + regime_trades['pct_return']/100).sum()) - 1
        win_rate = (regime_trades['pct_return'] > 0).mean()
        
        print(f"  Avg return: {avg_return:.4f}% ({avg_return*100:.2f} bps)")
        print(f"  Total return: {total_return*100:.2f}%")
        print(f"  Win rate: {win_rate:.1%}")
        
        # Break down by direction
        for signal, direction in [(1, 'Long'), (-1, 'Short')]:
            direction_trades = regime_trades[regime_trades['signal'] == signal]
            if len(direction_trades) > 0:
                print(f"  {direction}: {len(direction_trades)} trades, "
                      f"{direction_trades['pct_return'].mean():.4f}% avg, "
                      f"{(direction_trades['pct_return'] > 0).mean():.1%} win rate")

# Ranging vs Trending markets
print("\n=== RANGING vs TRENDING MARKETS ===")
ranging_trades = trades_df[trades_df['is_ranging'] == True]
trending_trades = trades_df[trades_df['is_ranging'] == False]

for market_type, market_trades in [("Ranging", ranging_trades), ("Trending", trending_trades)]:
    if len(market_trades) > 0:
        print(f"\n{market_type} Markets ({len(market_trades)} trades):")
        avg_return = market_trades['pct_return'].mean()
        total_return = np.exp(np.log(1 + market_trades['pct_return']/100).sum()) - 1
        win_rate = (market_trades['pct_return'] > 0).mean()
        
        print(f"  Avg return: {avg_return:.4f}% ({avg_return*100:.2f} bps)")
        print(f"  Total return: {total_return*100:.2f}%")
        print(f"  Win rate: {win_rate:.1%}")
        
        # By direction
        for signal, direction in [(1, 'Long'), (-1, 'Short')]:
            direction_trades = market_trades[market_trades['signal'] == signal]
            if len(direction_trades) > 0:
                print(f"  {direction}: {len(direction_trades)} trades, "
                      f"{direction_trades['pct_return'].mean():.4f}% avg")

# Volatility percentile analysis
print("\n=== PERFORMANCE BY VOLATILITY PERCENTILE ===")
vol_buckets = pd.cut(trades_df['vol_percentile'], bins=[0, 20, 40, 60, 80, 100], 
                     labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
trades_df['vol_bucket'] = vol_buckets

for bucket in ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']:
    bucket_trades = trades_df[trades_df['vol_bucket'] == bucket]
    if len(bucket_trades) > 0:
        avg_return = bucket_trades['pct_return'].mean()
        win_rate = (bucket_trades['pct_return'] > 0).mean()
        print(f"\nVolatility {bucket}:")
        print(f"  Trades: {len(bucket_trades)}")
        print(f"  Avg return: {avg_return:.4f}% ({avg_return*100:.2f} bps)")
        print(f"  Win rate: {win_rate:.1%}")

# Range analysis
print("\n=== PERFORMANCE BY 20-DAY RANGE ===")
range_buckets = pd.cut(trades_df['range_20'], bins=[0, 1, 2, 3, 5, 100], 
                       labels=['<1%', '1-2%', '2-3%', '3-5%', '>5%'])
trades_df['range_bucket'] = range_buckets

for bucket in ['<1%', '1-2%', '2-3%', '3-5%', '>5%']:
    bucket_trades = trades_df[trades_df['range_bucket'] == bucket]
    if len(bucket_trades) > 0:
        avg_return = bucket_trades['pct_return'].mean()
        total_return = np.exp(np.log(1 + bucket_trades['pct_return']/100).sum()) - 1
        print(f"\n20-day range {bucket}:")
        print(f"  Trades: {len(bucket_trades)}")
        print(f"  Avg return: {avg_return:.4f}% ({avg_return*100:.2f} bps)")
        print(f"  Total return: {total_return*100:.2f}%")

# Best conditions summary
print("\n=== OPTIMAL CONDITIONS SUMMARY ===")
conditions = [
    ("Low volatility", trades_df['low_vol'] == True),
    ("High volatility", trades_df['high_vol'] == True),
    ("Ranging markets", trades_df['is_ranging'] == True),
    ("Low vol + Ranging", (trades_df['low_vol'] == True) & (trades_df['is_ranging'] == True)),
    ("High vol + Trending", (trades_df['high_vol'] == True) & (trades_df['is_ranging'] == False)),
    ("Medium range (1-3%)", trades_df['range_20'].between(1, 3))
]

for desc, condition in conditions:
    filtered = trades_df[condition]
    if len(filtered) > 0:
        avg_return = filtered['pct_return'].mean()
        print(f"\n{desc}: {len(filtered)} trades, {avg_return:.4f}% avg ({avg_return*100:.2f} bps)")