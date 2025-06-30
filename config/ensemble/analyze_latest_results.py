#!/usr/bin/env python3
"""Analyze the latest ensemble results."""

import duckdb
import pandas as pd
import numpy as np
import json

# Load metadata
with open('results/latest/metadata.json') as f:
    metadata = json.load(f)

# Load signals and calculate returns
con = duckdb.connect()

# Load signals
signals = con.execute("""
    SELECT ts, val as signal
    FROM read_parquet('results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
    ORDER BY ts
""").df()

# Load market data
market = con.execute("""
    SELECT timestamp as ts, close, open, high, low, volume
    FROM read_parquet('../../data/SPY_5m.parquet')
    WHERE timestamp >= '2024-03-26' AND timestamp <= '2025-01-27'
    ORDER BY timestamp
""").df()

# Convert timestamps to match
market['ts'] = pd.to_datetime(market['ts'])
signals['ts'] = pd.to_datetime(signals['ts'])

# Merge and forward fill signals
df = market.merge(signals, on='ts', how='left')
df['signal'] = df['signal'].ffill().fillna(0)

# Calculate returns
df['returns'] = df['close'].pct_change()
df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1

# Calculate metrics
total_return = df['cumulative_returns'].iloc[-1]
trading_days = len(df) / 78  # ~78 5-min bars per day
annualized_return = (1 + total_return) ** (252 / trading_days) - 1

# Calculate Sharpe ratio
if df['strategy_returns'].std() > 0:
    sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252 * 78)
else:
    sharpe = 0

# Max drawdown
running_max = (1 + df['cumulative_returns']).cummax()
drawdown = (1 + df['cumulative_returns']) / running_max - 1
max_drawdown = drawdown.min()

# Analyze trades - look at all signal changes
all_signal_changes = signals.copy()
print(f'\nDEBUG: Total signal changes: {len(all_signal_changes)}')
print(f'Signal value counts:')
print(all_signal_changes['signal'].value_counts())

# Calculate trade returns by pairing entries and exits
trade_returns = []
long_trades = []
short_trades = []

# Track current position
current_position = 0
entry_price = None
entry_time = None

for _, row in df.iterrows():
    signal = row['signal']
    
    # Position change
    if signal != current_position:
        # Exit current position
        if current_position != 0 and entry_price is not None:
            exit_price = row['close']
            trade_return = (exit_price / entry_price - 1) * current_position
            trade_returns.append(trade_return)
            
            if current_position > 0:
                long_trades.append(trade_return)
            else:
                short_trades.append(trade_return)
        
        # Enter new position
        if signal != 0:
            entry_price = row['close']
            entry_time = row['ts']
        else:
            entry_price = None
            entry_time = None
            
        current_position = signal

# Calculate win rate and other metrics
if trade_returns:
    win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
    avg_win = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
    avg_loss = np.mean([r for r in trade_returns if r <= 0]) if any(r <= 0 for r in trade_returns) else 0
    profit_factor = abs(sum(r for r in trade_returns if r > 0) / sum(r for r in trade_returns if r < 0)) if any(r < 0 for r in trade_returns) else np.inf
else:
    win_rate = avg_win = avg_loss = profit_factor = 0

print('=== BOLLINGER BANDS PERFORMANCE (config/ensemble) ===')
print(f'Configuration: period=15, std_dev=3.0, exit_threshold=0.0001')
print(f'Data Range: {df["ts"].min()} to {df["ts"].max()}')
print(f'Total Bars: {len(df):,}')
print('')
print('=== RETURNS ===')
print(f'Total Return: {total_return*100:.2f}%')
print(f'Annualized Return: {annualized_return*100:.2f}%')
print(f'Sharpe Ratio: {sharpe:.2f}')
print(f'Max Drawdown: {max_drawdown*100:.2f}%')
print('')
print('=== TRADING STATISTICS ===')
print(f'Total Trades: {len(trade_returns)}')
print(f'  Long Trades: {len(long_trades)} ({len(long_trades)/len(trade_returns)*100:.1f}%)')
print(f'  Short Trades: {len(short_trades)} ({len(short_trades)/len(trade_returns)*100:.1f}%)')
print(f'Win Rate: {win_rate*100:.1f}%')
print(f'Avg Win: {avg_win*100:.3f}%')
print(f'Avg Loss: {avg_loss*100:.3f}%')
print(f'Profit Factor: {profit_factor:.2f}')
print('')
print(f'Signal Changes: {len(all_signal_changes)} ({len(all_signal_changes)/len(df)*100:.2f}% of bars)')
print(f'Time in Market: {(df["signal"] != 0).sum()/len(df)*100:.1f}%')

# Analyze trade distribution
print('')
print('=== TRADE ANALYSIS ===')
if long_trades:
    print(f'Long Trade Performance:')
    print(f'  Count: {len(long_trades)}')
    print(f'  Win Rate: {sum(1 for r in long_trades if r > 0)/len(long_trades)*100:.1f}%')
    print(f'  Avg Return: {np.mean(long_trades)*100:.3f}%')

if short_trades:
    print(f'Short Trade Performance:')
    print(f'  Count: {len(short_trades)}')
    print(f'  Win Rate: {sum(1 for r in short_trades if r > 0)/len(short_trades)*100:.1f}%')
    print(f'  Avg Return: {np.mean(short_trades)*100:.3f}%')

# Check if this is actually using the tight exit threshold
print('')
print('=== EXIT ANALYSIS ===')
print(f'Exit threshold configured: 0.0001 (0.01%)')
print('Checking if trades honor this tight exit...')

# Sample a few trades to see actual holding periods
if len(all_signal_changes) >= 4:
    print('\nSample trades (first 5):')
    count = 0
    for i in range(len(all_signal_changes)-1):
        if count >= 5:
            break
        entry = all_signal_changes.iloc[i]
        if i + 1 < len(all_signal_changes):
            exit = all_signal_changes.iloc[i + 1]
            if entry['signal'] != 0 and exit['signal'] == 0:
                duration = pd.Timedelta(exit['ts'] - entry['ts'])
                print(f'  Entry: {entry["ts"]}, Exit: {exit["ts"]}, Duration: {duration}')
                count += 1