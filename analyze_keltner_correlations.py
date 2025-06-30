#!/usr/bin/env python3
"""Analyze Keltner Bands performance correlations with market conditions."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

workspace_path = Path("workspaces/signal_generation_11d63547")

# Load metadata
with open(workspace_path / "metadata.json", 'r') as f:
    metadata = json.load(f)

# Load SPY price data
data_path = Path("data/SPY_1m.parquet")
if data_path.exists():
    price_df = pd.read_parquet(data_path)
else:
    data_path = Path("data/SPY_1m.csv")
    price_df = pd.read_csv(data_path)
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])

print("=== KELTNER BANDS CORRELATION ANALYSIS ===\n")
print(f"Analyzing {len(price_df)} bars of market data...")

# Calculate market indicators
print("Calculating market indicators...")

# Basic returns
price_df['returns'] = price_df['close'].pct_change()

# Moving averages
price_df['sma20'] = price_df['close'].rolling(20).mean()
price_df['sma50'] = price_df['close'].rolling(50).mean()
price_df['sma200'] = price_df['close'].rolling(200).mean()

# Volatility
price_df['volatility_20'] = price_df['returns'].rolling(20).std() * np.sqrt(390 * 252)  # Annualized
price_df['volatility_rank'] = price_df['volatility_20'].rolling(1000).rank(pct=True)

# Volume
price_df['volume_sma'] = price_df['volume'].rolling(20).mean()
price_df['volume_ratio'] = price_df['volume'] / price_df['volume_sma']

# VWAP
price_df['typical_price'] = (price_df['high'] + price_df['low'] + price_df['close']) / 3
price_df['pv'] = price_df['typical_price'] * price_df['volume']
price_df['cumulative_pv'] = price_df['pv'].cumsum()
price_df['cumulative_volume'] = price_df['volume'].cumsum()
price_df['vwap'] = price_df['cumulative_pv'] / price_df['cumulative_volume']

# Session VWAP (resets daily)
price_df['date'] = price_df['timestamp'].dt.date
price_df['session_pv'] = price_df.groupby('date')['pv'].cumsum()
price_df['session_volume'] = price_df.groupby('date')['volume'].cumsum()
price_df['session_vwap'] = price_df['session_pv'] / price_df['session_volume']

# Price position
price_df['price_to_vwap'] = (price_df['close'] / price_df['vwap'] - 1) * 100
price_df['price_to_session_vwap'] = (price_df['close'] / price_df['session_vwap'] - 1) * 100
price_df['price_to_sma20'] = (price_df['close'] / price_df['sma20'] - 1) * 100
price_df['price_to_sma50'] = (price_df['close'] / price_df['sma50'] - 1) * 100

# Trend
price_df['trend'] = np.where(price_df['sma50'] > price_df['sma200'], 'up', 'down')
price_df['trend_strength'] = np.where(price_df['sma20'] > price_df['sma50'], 'strong', 'weak')

# Time features
price_df['hour'] = price_df['timestamp'].dt.hour
price_df['minute'] = price_df['timestamp'].dt.minute
price_df['time_of_day'] = price_df['hour'] + price_df['minute'] / 60

# Market sessions
price_df['session'] = 'regular'
price_df.loc[(price_df['hour'] == 9) & (price_df['minute'] < 30), 'session'] = 'premarket'
price_df.loc[(price_df['hour'] >= 16), 'session'] = 'afterhours'
price_df.loc[(price_df['hour'] == 9) & (price_df['minute'] >= 30) & (price_df['minute'] < 45), 'session'] = 'opening'
price_df.loc[(price_df['hour'] == 15) & (price_df['minute'] >= 45), 'session'] = 'closing'

# RSI
delta = price_df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
price_df['rsi'] = 100 - (100 / (1 + rs))

# Now analyze each strategy
all_trade_results = []

# From config: period: [10, 20, 30], multiplier: [1.5, 2.0, 2.5]
periods = [10, 20, 30]
multipliers = [1.5, 2.0, 2.5]

idx = 0
for period in periods:
    for multiplier in multipliers:
        signal_file = workspace_path / f"traces/SPY_1m/signals/keltner_bands/SPY_compiled_strategy_{idx}.parquet"
        
        if signal_file.exists():
            print(f"\nAnalyzing Strategy {idx} (Period={period}, Multiplier={multiplier})...")
            
            # Load signals
            signals_df = pd.read_parquet(signal_file)
            
            # Calculate trades with market conditions
            trades = []
            current_position = None
            
            for i in range(len(signals_df)):
                signal = signals_df.iloc[i]
                bar_idx = signal['idx']
                signal_val = signal['val']
                
                # Entry
                if current_position is None and signal_val != 0:
                    if bar_idx < len(price_df):
                        current_position = {
                            'entry_idx': bar_idx,
                            'entry_price': signal['px'],
                            'direction': signal_val,
                            'entry_conditions': price_df.iloc[bar_idx].to_dict()
                        }
                
                # Exit
                elif current_position is not None and (signal_val == 0 or signal_val != current_position['direction']):
                    if bar_idx < len(price_df):
                        exit_price = signal['px']
                        entry_price = current_position['entry_price']
                        
                        # Calculate return
                        if current_position['direction'] > 0:  # Long
                            gross_return = (exit_price / entry_price) - 1
                        else:  # Short
                            gross_return = (entry_price / exit_price) - 1
                        
                        # Get market conditions at entry
                        conditions = current_position['entry_conditions']
                        
                        trade_result = {
                            'strategy_idx': idx,
                            'period': period,
                            'multiplier': multiplier,
                            'direction': 'long' if current_position['direction'] > 0 else 'short',
                            'gross_return': gross_return,
                            'gross_return_bps': gross_return * 10000,
                            'bars_held': bar_idx - current_position['entry_idx'],
                            # Market conditions at entry
                            'volatility_rank': conditions.get('volatility_rank', np.nan),
                            'volume_ratio': conditions.get('volume_ratio', np.nan),
                            'price_to_vwap': conditions.get('price_to_vwap', np.nan),
                            'price_to_session_vwap': conditions.get('price_to_session_vwap', np.nan),
                            'price_to_sma20': conditions.get('price_to_sma20', np.nan),
                            'trend': conditions.get('trend', 'unknown'),
                            'trend_strength': conditions.get('trend_strength', 'unknown'),
                            'rsi': conditions.get('rsi', np.nan),
                            'hour': conditions.get('hour', np.nan),
                            'session': conditions.get('session', 'unknown')
                        }
                        
                        trades.append(trade_result)
                        all_trade_results.append(trade_result)
                        
                        # Reset or flip position
                        if signal_val != 0 and bar_idx < len(price_df):
                            current_position = {
                                'entry_idx': bar_idx,
                                'entry_price': signal['px'],
                                'direction': signal_val,
                                'entry_conditions': price_df.iloc[bar_idx].to_dict()
                            }
                        else:
                            current_position = None
        
        idx += 1

# Convert to DataFrame
trades_df = pd.DataFrame(all_trade_results)
print(f"\n\nTotal trades analyzed: {len(trades_df)}")

# Save raw trades for further analysis
trades_df.to_csv('keltner_trades_with_conditions.csv', index=False)
print("✓ Saved detailed trades to keltner_trades_with_conditions.csv")

print("\n=== PERFORMANCE BY MARKET CONDITIONS ===")

# Helper function to analyze a subset
def analyze_subset(df, name):
    if len(df) == 0:
        return None
    return {
        'condition': name,
        'trades': len(df),
        'avg_return_bps': df['gross_return_bps'].mean(),
        'win_rate': (df['gross_return'] > 0).mean() * 100,
        'long_pct': (df['direction'] == 'long').mean() * 100,
        'profitable_2bp': df['gross_return_bps'].mean() > 2
    }

# 1. Direction Analysis
print("\n--- BY DIRECTION ---")
results = []
for direction in ['long', 'short']:
    subset = trades_df[trades_df['direction'] == direction]
    result = analyze_subset(subset, direction.upper())
    if result:
        results.append(result)
        print(f"{direction.upper()}: {result['trades']} trades, {result['avg_return_bps']:.2f} bps, {result['win_rate']:.1f}% win")

# 2. Volatility Analysis
print("\n--- BY VOLATILITY RANK ---")
vol_thresholds = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
for i in range(len(vol_thresholds)-1):
    mask = (trades_df['volatility_rank'] >= vol_thresholds[i]) & (trades_df['volatility_rank'] < vol_thresholds[i+1])
    subset = trades_df[mask]
    result = analyze_subset(subset, f"Vol {int(vol_thresholds[i]*100)}-{int(vol_thresholds[i+1]*100)}%")
    if result and result['trades'] > 50:
        results.append(result)
        print(f"Vol {int(vol_thresholds[i]*100)}-{int(vol_thresholds[i+1]*100)}%: {result['trades']} trades, {result['avg_return_bps']:.2f} bps")

# 3. Volume Analysis
print("\n--- BY VOLUME RATIO ---")
for threshold in [0.5, 1.0, 1.5, 2.0, 3.0]:
    subset = trades_df[trades_df['volume_ratio'] > threshold]
    result = analyze_subset(subset, f"Volume >{threshold}x")
    if result and result['trades'] > 50:
        results.append(result)
        print(f"Volume >{threshold}x: {result['trades']} trades, {result['avg_return_bps']:.2f} bps")

# 4. VWAP Position
print("\n--- BY VWAP POSITION ---")
for condition, mask in [
    ('Far Below VWAP (<-0.2%)', trades_df['price_to_vwap'] < -0.2),
    ('Below VWAP (-0.2% to 0%)', (trades_df['price_to_vwap'] >= -0.2) & (trades_df['price_to_vwap'] < 0)),
    ('Above VWAP (0% to 0.2%)', (trades_df['price_to_vwap'] >= 0) & (trades_df['price_to_vwap'] < 0.2)),
    ('Far Above VWAP (>0.2%)', trades_df['price_to_vwap'] >= 0.2)
]:
    subset = trades_df[mask]
    result = analyze_subset(subset, condition)
    if result and result['trades'] > 50:
        results.append(result)
        print(f"{condition}: {result['trades']} trades, {result['avg_return_bps']:.2f} bps")

# 5. Trend Analysis
print("\n--- BY TREND ---")
for trend in ['up', 'down']:
    for strength in ['strong', 'weak']:
        mask = (trades_df['trend'] == trend) & (trades_df['trend_strength'] == strength)
        subset = trades_df[mask]
        result = analyze_subset(subset, f"{trend.upper()} {strength}")
        if result and result['trades'] > 50:
            results.append(result)
            print(f"{trend.upper()} {strength}: {result['trades']} trades, {result['avg_return_bps']:.2f} bps")

# 6. Time-based Analysis
print("\n--- BY TIME OF DAY ---")
for hour in [9, 10, 11, 12, 13, 14, 15]:
    subset = trades_df[trades_df['hour'] == hour]
    result = analyze_subset(subset, f"Hour {hour}:00")
    if result and result['trades'] > 50:
        results.append(result)
        print(f"Hour {hour}:00: {result['trades']} trades, {result['avg_return_bps']:.2f} bps")

# 7. RSI Analysis
print("\n--- BY RSI ---")
for condition, mask in [
    ('RSI <30', trades_df['rsi'] < 30),
    ('RSI 30-50', (trades_df['rsi'] >= 30) & (trades_df['rsi'] < 50)),
    ('RSI 50-70', (trades_df['rsi'] >= 50) & (trades_df['rsi'] < 70)),
    ('RSI >70', trades_df['rsi'] >= 70)
]:
    subset = trades_df[mask]
    result = analyze_subset(subset, condition)
    if result and result['trades'] > 50:
        results.append(result)
        print(f"{condition}: {result['trades']} trades, {result['avg_return_bps']:.2f} bps")

# Find best combinations
print("\n=== SEARCHING FOR PROFITABLE COMBINATIONS ===")

# Test specific combinations
combinations = [
    ('High Vol + Shorts', (trades_df['volatility_rank'] > 0.8) & (trades_df['direction'] == 'short')),
    ('High Vol + Longs', (trades_df['volatility_rank'] > 0.8) & (trades_df['direction'] == 'long')),
    ('High Volume + Shorts', (trades_df['volume_ratio'] > 2) & (trades_df['direction'] == 'short')),
    ('Low Vol + Mean Rev', (trades_df['volatility_rank'] < 0.3)),
    ('Opening Hour', trades_df['hour'] == 9),
    ('Closing Hour', trades_df['hour'] == 15),
    ('Extreme RSI Shorts', (trades_df['rsi'] > 70) & (trades_df['direction'] == 'short')),
    ('Extreme RSI Longs', (trades_df['rsi'] < 30) & (trades_df['direction'] == 'long')),
    ('Trend Following', ((trades_df['trend'] == 'up') & (trades_df['direction'] == 'long')) | 
                       ((trades_df['trend'] == 'down') & (trades_df['direction'] == 'short'))),
    ('Counter Trend', ((trades_df['trend'] == 'up') & (trades_df['direction'] == 'short')) | 
                      ((trades_df['trend'] == 'down') & (trades_df['direction'] == 'long'))),
]

combo_results = []
for name, mask in combinations:
    subset = trades_df[mask]
    if len(subset) > 20:
        avg_return = subset['gross_return_bps'].mean()
        combo_results.append({
            'filter': name,
            'trades': len(subset),
            'avg_return_bps': avg_return,
            'net_2bp': avg_return - 2,
            'win_rate': (subset['gross_return'] > 0).mean() * 100,
            'trades_per_day': len(subset) / (metadata['total_bars'] / 390)
        })

# Sort by net return
combo_df = pd.DataFrame(combo_results).sort_values('net_2bp', ascending=False)
print("\nTop combinations after 2bp cost:")
print(combo_df.head(10))

# Save results
combo_df.to_csv('keltner_filter_combinations.csv', index=False)
print("\n✓ Saved filter combinations to keltner_filter_combinations.csv")