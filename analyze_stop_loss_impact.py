#!/usr/bin/env python3
"""Analyze how different stop-loss levels would impact performance."""

import pandas as pd
import numpy as np

# First, run the ensemble strategy
print("Running strategy to generate signals...")
import subprocess
result = subprocess.run([
    'python3', 'main.py',
    '--config', 'config/ensemble/config.yaml',
    '--signal-generation',
    '--dataset', 'train'
], capture_output=True, text=True)

if result.returncode != 0:
    print("Error running strategy:")
    print(result.stderr[-500:])
    exit(1)

# Load signals and price data
print("\nLoading data...")
signals = pd.read_parquet('config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
prices = pd.read_csv('data/SPY_5m.csv')

# Merge signals with prices
signals['timestamp'] = pd.to_datetime(signals['ts'])
prices['timestamp'] = pd.to_datetime(prices['timestamp'])
data = pd.merge_asof(signals, prices[['timestamp', 'close', 'high', 'low']], 
                     on='timestamp', direction='backward')

# Identify trades
data['position'] = data['val']
data['trade'] = data['position'].diff().fillna(0) != 0

# Analyze each trade with different stop-loss levels
stop_levels = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]  # 0.5% to 3%
results = []

print("\n=== STOP-LOSS ANALYSIS ===")
for stop_loss in stop_levels:
    trades = []
    in_position = False
    entry_price = 0
    position_type = 0
    
    for i, row in data.iterrows():
        if row['trade'] and row['position'] != 0:
            # Entry
            in_position = True
            entry_price = row['close']
            position_type = row['position']
            entry_time = row['timestamp']
            
        elif in_position:
            # Check stop-loss
            if position_type > 0:  # Long
                pct_change = (row['low'] - entry_price) / entry_price
                if pct_change <= -stop_loss:
                    # Stop hit
                    exit_price = entry_price * (1 - stop_loss)
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': row['timestamp'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': -stop_loss,
                        'stopped': True
                    })
                    in_position = False
                    
            else:  # Short
                pct_change = (row['high'] - entry_price) / entry_price
                if pct_change >= stop_loss:
                    # Stop hit
                    exit_price = entry_price * (1 + stop_loss)
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': row['timestamp'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': -stop_loss,
                        'stopped': True
                    })
                    in_position = False
            
            # Normal exit
            if row['trade'] and not in_position:
                exit_price = row['close']
                if position_type > 0:
                    trade_return = (exit_price - entry_price) / entry_price
                else:
                    trade_return = (entry_price - exit_price) / entry_price
                    
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row['timestamp'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'stopped': False
                })
                in_position = False
    
    # Calculate metrics
    if trades:
        trade_df = pd.DataFrame(trades)
        total_return = (1 + trade_df['return']).prod() - 1
        win_rate = (trade_df['return'] > 0).mean()
        avg_win = trade_df[trade_df['return'] > 0]['return'].mean() if any(trade_df['return'] > 0) else 0
        avg_loss = trade_df[trade_df['return'] < 0]['return'].mean() if any(trade_df['return'] < 0) else 0
        stopped_pct = trade_df['stopped'].mean()
        
        results.append({
            'stop_loss': stop_loss,
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'num_trades': len(trades),
            'stopped_pct': stopped_pct
        })

# Display results
results_df = pd.DataFrame(results)
print("\nStop-Loss Impact Analysis:")
print(results_df.to_string(index=False, float_format='%.4f'))

# Find optimal stop-loss
best_idx = results_df['total_return'].idxmax()
print(f"\n✅ Optimal stop-loss: {results_df.iloc[best_idx]['stop_loss']*100:.1f}%")
print(f"   Total return: {results_df.iloc[best_idx]['total_return']*100:.2f}%")
print(f"   Win rate: {results_df.iloc[best_idx]['win_rate']*100:.1f}%")