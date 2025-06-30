#!/usr/bin/env python3
"""Apply trend/volume/volatility filters to the robust parameter results"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Load the test data trace
trace_file = Path("config/keltner/robust_config/results/20250622_212146/traces/mean_reversion/SPY_5m_kb_robust_p10_m3.parquet")
metadata_file = Path("config/keltner/robust_config/results/20250622_212146/metadata.json")

# Load SPY 5m data to calculate indicators
spy_data = pd.read_csv("data/SPY_5m.csv")
spy_data['datetime'] = pd.to_datetime(spy_data['timestamp'])
spy_data = spy_data.set_index('datetime').sort_index()

print("APPLYING FILTERS TO ROBUST PARAMETERS (P=10, M=3.0)")
print("="*80)

# Calculate indicators for filtering
print("\nCalculating indicators...")

# Trend indicators
spy_data['sma_50'] = spy_data['close'].rolling(50).mean()
spy_data['sma_200'] = spy_data['close'].rolling(200).mean()
spy_data['trend_distance'] = (spy_data['close'] - spy_data['sma_50']) / spy_data['sma_50']

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

# RSI
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

spy_data['rsi_14'] = calculate_rsi(spy_data['close'])

# ADX
def calculate_adx(df, period=14):
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = df['true_range']
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx

spy_data['adx_14'] = calculate_adx(spy_data)

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
    
    # Set end index
    if i < len(df_trace) - 1:
        end_idx = df_trace.iloc[i + 1]['idx']
    else:
        end_idx = total_bars
    
    signals[start_idx:end_idx] = signal_value

# Apply different filters and analyze performance
def analyze_with_filter(signals, spy_data, filter_func, filter_name):
    """Apply filter and calculate performance"""
    filtered_signals = signals.copy()
    
    # Apply filter
    for i in range(len(filtered_signals)):
        if filtered_signals[i] != 0 and i < len(spy_data):
            if not filter_func(spy_data.iloc[i]):
                filtered_signals[i] = 0
    
    # Count trades
    trades = 0
    trade_returns = []
    in_trade = False
    entry_price = None
    entry_idx = None
    trade_direction = 0
    
    for i in range(len(filtered_signals) - 1):
        if not in_trade and filtered_signals[i] != 0:
            trades += 1
            in_trade = True
            entry_price = spy_data.iloc[i]['close']
            entry_idx = i
            trade_direction = filtered_signals[i]
        elif in_trade and filtered_signals[i] == 0:
            in_trade = False
            exit_price = spy_data.iloc[i]['close']
            
            if trade_direction > 0:  # Long
                ret = (exit_price - entry_price) / entry_price
            else:  # Short
                ret = (entry_price - exit_price) / entry_price
            trade_returns.append(ret)
    
    # Calculate metrics
    if trade_returns:
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100
        avg_return = np.mean(trade_returns) * 100
        total_return = sum(trade_returns) * 100
        
        # Annualized
        trading_days = total_bars / 78  # 78 5-min bars per day
        annual_return = (total_return / trading_days) * 252
    else:
        win_rate = avg_return = total_return = annual_return = 0
    
    return {
        'filter': filter_name,
        'trades': trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return,
        'annual_return': annual_return
    }

# Define filters
filters = [
    ('No Filter', lambda row: True),
    ('Trend: Near SMA50 (±2%)', lambda row: abs(row['trend_distance']) < 0.02 if pd.notna(row['trend_distance']) else False),
    ('Trend: Near SMA50 (±1%)', lambda row: abs(row['trend_distance']) < 0.01 if pd.notna(row['trend_distance']) else False),
    ('Volume: Above Average (>1.2x)', lambda row: row['volume_ratio'] > 1.2 if pd.notna(row['volume_ratio']) else False),
    ('Volume: Above Average (>1.5x)', lambda row: row['volume_ratio'] > 1.5 if pd.notna(row['volume_ratio']) else False),
    ('Volatility: Normal (30-70%ile)', lambda row: 0.3 < row['vol_percentile'] < 0.7 if pd.notna(row['vol_percentile']) else False),
    ('RSI: Not Extreme (30-70)', lambda row: 30 < row['rsi_14'] < 70 if pd.notna(row['rsi_14']) else False),
    ('RSI: Oversold/Overbought', lambda row: row['rsi_14'] < 40 or row['rsi_14'] > 60 if pd.notna(row['rsi_14']) else False),
    ('Low ADX (<25)', lambda row: row['adx_14'] < 25 if pd.notna(row['adx_14']) else False),
    ('Combined: Trend + Volume', lambda row: abs(row['trend_distance']) < 0.02 and row['volume_ratio'] > 1.2 if pd.notna(row['trend_distance']) and pd.notna(row['volume_ratio']) else False),
    ('Best Combo: All Conditions', lambda row: abs(row['trend_distance']) < 0.015 and row['volume_ratio'] > 1.1 and 30 < row['rsi_14'] < 70 and row['adx_14'] < 30 if all(pd.notna([row['trend_distance'], row['volume_ratio'], row['rsi_14'], row['adx_14']])) else False)
]

# Analyze each filter
results = []
for filter_name, filter_func in filters:
    result = analyze_with_filter(signals, spy_data, filter_func, filter_name)
    results.append(result)

# Display results
print("\nFILTER ANALYSIS RESULTS:")
print("-"*80)
print(f"{'Filter':<30} {'Trades':>8} {'Win Rate':>10} {'Avg Ret':>10} {'Annual':>10}")
print("-"*80)

for r in results:
    print(f"{r['filter']:<30} {r['trades']:>8} {r['win_rate']:>9.1f}% {r['avg_return']:>9.3f}% {r['annual_return']:>9.1f}%")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)

# Find best performing filter
best_filter = max(results, key=lambda x: x['annual_return'])
print(f"\nBest Filter: {best_filter['filter']}")
print(f"  Annual Return: {best_filter['annual_return']:.1f}%")
print(f"  Win Rate: {best_filter['win_rate']:.1f}%")
print(f"  Trades: {best_filter['trades']}")

# Compare to unfiltered
unfiltered = results[0]
if best_filter['annual_return'] > unfiltered['annual_return']:
    improvement = best_filter['annual_return'] - unfiltered['annual_return']
    print(f"\n✅ Filters IMPROVE performance by {improvement:.1f}% annually")
else:
    print(f"\n❌ No filter improves the already poor performance")
    print(f"   The base strategy (P=10, M=3.0) is fundamentally flawed")