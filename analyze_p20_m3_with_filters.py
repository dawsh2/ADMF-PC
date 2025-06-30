#!/usr/bin/env python3
"""Apply filters to P=20, M=3.0 strategy and analyze long/short performance"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Load the test data trace for P=20, M=3.0
trace_file = Path("config/keltner/robust_config/results/20250622_213055/traces/mean_reversion/SPY_5m_kb_robust_p10_m3.parquet")
metadata_file = Path("config/keltner/robust_config/results/20250622_213055/metadata.json")

# Load SPY 5m data to calculate indicators
spy_data = pd.read_csv("data/SPY_5m.csv")
spy_data['datetime'] = pd.to_datetime(spy_data['timestamp'])
spy_data = spy_data.set_index('datetime').sort_index()

print("FILTER ANALYSIS FOR P=20, M=3.0 STRATEGY")
print("="*80)

# Calculate indicators
print("\nCalculating indicators...")

# Trend indicators
spy_data['sma_20'] = spy_data['close'].rolling(20).mean()
spy_data['sma_50'] = spy_data['close'].rolling(50).mean()
spy_data['sma_200'] = spy_data['close'].rolling(200).mean()
spy_data['trend_strength'] = (spy_data['close'] - spy_data['sma_50']) / spy_data['sma_50']
spy_data['trend_20'] = (spy_data['close'] - spy_data['sma_20']) / spy_data['sma_20']

# Volume indicators
spy_data['volume_sma_20'] = spy_data['volume'].rolling(20).mean()
spy_data['volume_ratio'] = spy_data['volume'] / spy_data['volume_sma_20']

# Volatility indicators
spy_data['returns'] = spy_data['close'].pct_change()
spy_data['volatility_20'] = spy_data['returns'].rolling(20).std()
spy_data['volatility_50'] = spy_data['returns'].rolling(50).std()
spy_data['vol_percentile'] = spy_data['volatility_20'].rolling(50).rank(pct=True)

# ATR
spy_data['high_low'] = spy_data['high'] - spy_data['low']
spy_data['high_close'] = abs(spy_data['high'] - spy_data['close'].shift())
spy_data['low_close'] = abs(spy_data['low'] - spy_data['close'].shift())
spy_data['true_range'] = spy_data[['high_low', 'high_close', 'low_close']].max(axis=1)
spy_data['atr_14'] = spy_data['true_range'].rolling(14).mean()
spy_data['atr_50'] = spy_data['true_range'].rolling(50).mean()
spy_data['atr_ratio'] = spy_data['atr_14'] / spy_data['atr_50']

# RSI
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

spy_data['rsi_14'] = calculate_rsi(spy_data['close'])

# Load trace data
df_trace = pd.read_parquet(trace_file)
print(f"\nLoaded {len(df_trace)} signal changes")

# Convert trace to full timeline
with open(metadata_file, 'r') as f:
    metadata = json.load(f)
total_bars = metadata['total_bars']

# Create full signal array
signals = np.zeros(total_bars)
for i in range(len(df_trace)):
    start_idx = df_trace.iloc[i]['idx']
    signal_value = df_trace.iloc[i]['val']
    
    if i < len(df_trace) - 1:
        end_idx = df_trace.iloc[i + 1]['idx']
    else:
        end_idx = total_bars
    
    signals[start_idx:end_idx] = signal_value

# Analyze with different filters
def analyze_with_filter(signals, spy_data, filter_func, filter_name):
    """Apply filter and calculate performance separately for long and short"""
    filtered_signals = signals.copy()
    
    # Apply filter
    filter_count = 0
    for i in range(len(filtered_signals)):
        if filtered_signals[i] != 0 and i < len(spy_data):
            if not filter_func(spy_data.iloc[i], filtered_signals[i]):
                filtered_signals[i] = 0
                filter_count += 1
    
    # Analyze trades
    long_trades = []
    short_trades = []
    in_trade = False
    entry_price = None
    trade_direction = 0
    
    for i in range(len(filtered_signals) - 1):
        if not in_trade and filtered_signals[i] != 0:
            in_trade = True
            entry_price = spy_data.iloc[i]['close']
            trade_direction = filtered_signals[i]
        elif in_trade and filtered_signals[i] == 0:
            in_trade = False
            exit_price = spy_data.iloc[i]['close']
            
            if trade_direction > 0:  # Long
                ret = (exit_price - entry_price) / entry_price
                long_trades.append(ret)
            else:  # Short
                ret = (entry_price - exit_price) / entry_price
                short_trades.append(ret)
    
    # Calculate metrics
    trading_days = total_bars / 78
    
    def calc_metrics(trades):
        if trades:
            win_rate = sum(1 for r in trades if r > 0) / len(trades) * 100
            avg_return = np.mean(trades) * 100
            total_return = sum(trades) * 100
            annual_return = (total_return / trading_days) * 252
            return {
                'trades': len(trades),
                'trades_per_day': len(trades) / trading_days,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'annual_return': annual_return
            }
        else:
            return {
                'trades': 0,
                'trades_per_day': 0,
                'win_rate': 0,
                'avg_return': 0,
                'annual_return': 0
            }
    
    long_metrics = calc_metrics(long_trades)
    short_metrics = calc_metrics(short_trades)
    all_metrics = calc_metrics(long_trades + short_trades)
    
    return {
        'filter': filter_name,
        'filtered_out': filter_count,
        'long': long_metrics,
        'short': short_metrics,
        'all': all_metrics
    }

# Define filters (some can be direction-specific)
filters = [
    ('No Filter', lambda row, signal: True),
    
    # Trend filters
    ('Trend: Flat market (±1%)', lambda row, signal: abs(row['trend_strength']) < 0.01 if pd.notna(row['trend_strength']) else False),
    ('Trend: Flat market (±2%)', lambda row, signal: abs(row['trend_strength']) < 0.02 if pd.notna(row['trend_strength']) else False),
    ('Trend: With trend only', lambda row, signal: (signal > 0 and row['trend_strength'] > 0) or (signal < 0 and row['trend_strength'] < 0) if pd.notna(row['trend_strength']) else False),
    ('Trend: Counter-trend only', lambda row, signal: (signal > 0 and row['trend_strength'] < 0) or (signal < 0 and row['trend_strength'] > 0) if pd.notna(row['trend_strength']) else False),
    
    # Volume filters
    ('Volume: High (>1.2x avg)', lambda row, signal: row['volume_ratio'] > 1.2 if pd.notna(row['volume_ratio']) else False),
    ('Volume: Very high (>1.5x)', lambda row, signal: row['volume_ratio'] > 1.5 if pd.notna(row['volume_ratio']) else False),
    ('Volume: Normal (0.8-1.2x)', lambda row, signal: 0.8 < row['volume_ratio'] < 1.2 if pd.notna(row['volume_ratio']) else False),
    
    # Volatility filters
    ('Vol: Normal (30-70%ile)', lambda row, signal: 0.3 < row['vol_percentile'] < 0.7 if pd.notna(row['vol_percentile']) else False),
    ('Vol: High (>70%ile)', lambda row, signal: row['vol_percentile'] > 0.7 if pd.notna(row['vol_percentile']) else False),
    ('Vol: Low (<30%ile)', lambda row, signal: row['vol_percentile'] < 0.3 if pd.notna(row['vol_percentile']) else False),
    ('Vol: ATR normal', lambda row, signal: 0.8 < row['atr_ratio'] < 1.2 if pd.notna(row['atr_ratio']) else False),
    
    # RSI filters
    ('RSI: Long oversold (<40)', lambda row, signal: signal <= 0 or row['rsi_14'] < 40 if pd.notna(row['rsi_14']) else False),
    ('RSI: Short overbought (>60)', lambda row, signal: signal >= 0 or row['rsi_14'] > 60 if pd.notna(row['rsi_14']) else False),
    ('RSI: Extreme reversal', lambda row, signal: (signal > 0 and row['rsi_14'] < 30) or (signal < 0 and row['rsi_14'] > 70) if pd.notna(row['rsi_14']) else False),
    
    # Combined filters
    ('Best: Flat + Volume', lambda row, signal: abs(row['trend_strength']) < 0.015 and row['volume_ratio'] > 1.1 if pd.notna(row['trend_strength']) and pd.notna(row['volume_ratio']) else False),
    ('Conservative: All', lambda row, signal: abs(row['trend_strength']) < 0.02 and 0.9 < row['volume_ratio'] < 1.5 and 0.2 < row['vol_percentile'] < 0.8 if all(pd.notna([row['trend_strength'], row['volume_ratio'], row['vol_percentile']])) else False),
]

# Analyze each filter
results = []
for filter_name, filter_func in filters:
    result = analyze_with_filter(signals, spy_data, filter_func, filter_name)
    results.append(result)

# Display results
print("\nFILTER PERFORMANCE ANALYSIS:")
print("="*80)
print(f"{'Filter':<25} {'Total':>6} {'T/Day':>6} {'Win%':>6} {'Ann%':>7} | {'Long':>6} {'L/Day':>6} {'LWin%':>6} {'LAnn%':>7} | {'Short':>6} {'S/Day':>6} {'SWin%':>6} {'SAnn%':>7}")
print("-"*140)

for r in results:
    print(f"{r['filter']:<25} "
          f"{r['all']['trades']:>6} {r['all']['trades_per_day']:>6.2f} {r['all']['win_rate']:>6.1f} {r['all']['annual_return']:>7.2f} | "
          f"{r['long']['trades']:>6} {r['long']['trades_per_day']:>6.2f} {r['long']['win_rate']:>6.1f} {r['long']['annual_return']:>7.2f} | "
          f"{r['short']['trades']:>6} {r['short']['trades_per_day']:>6.2f} {r['short']['win_rate']:>6.1f} {r['short']['annual_return']:>7.2f}")

# Summary analysis
print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)

# Original performance
original = results[0]
print(f"\nOriginal (no filter): {original['all']['trades']} trades, {original['all']['trades_per_day']:.2f}/day, {original['all']['annual_return']:.2f}% annual")
print(f"  Long: {original['long']['trades']} trades, {original['long']['win_rate']:.1f}% win, {original['long']['annual_return']:.2f}% annual")
print(f"  Short: {original['short']['trades']} trades, {original['short']['win_rate']:.1f}% win, {original['short']['annual_return']:.2f}% annual")

# Find best filters
best_overall = max(results[1:], key=lambda x: x['all']['annual_return'])
best_long = max(results[1:], key=lambda x: x['long']['annual_return'])
best_short = max(results[1:], key=lambda x: x['short']['annual_return'])

print(f"\nBest overall filter: {best_overall['filter']}")
print(f"  {best_overall['all']['trades']} trades ({best_overall['all']['trades_per_day']:.2f}/day), {best_overall['all']['annual_return']:.2f}% annual")

print(f"\nBest long filter: {best_long['filter']}")
print(f"  {best_long['long']['trades']} trades, {best_long['long']['annual_return']:.2f}% annual")

print(f"\nBest short filter: {best_short['filter']}")
print(f"  {best_short['short']['trades']} trades, {best_short['short']['annual_return']:.2f}% annual")

# Direction analysis
if original['long']['annual_return'] > original['short']['annual_return']:
    print(f"\n📈 LONG BIAS: Longs outperform shorts by {original['long']['annual_return'] - original['short']['annual_return']:.2f}%")
else:
    print(f"\n📉 SHORT BIAS: Shorts outperform longs by {original['short']['annual_return'] - original['long']['annual_return']:.2f}%")