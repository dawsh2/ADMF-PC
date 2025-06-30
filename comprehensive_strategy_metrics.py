#!/usr/bin/env python3
"""Comprehensive metrics report for all strategies."""

import pandas as pd
import numpy as np
from pathlib import Path

# Strategy results with all metrics
strategies = {
    '5m_basic': {
        'workspace': '/Users/daws/ADMF-PC/workspaces/signal_generation_31415f83',
        'timeframe_minutes': 5,
        'total_bars': 16614,
        'trades': 162,
        'win_rate': 0.654,
        'annual_return': 0.0259,
        'total_return': 0.0217,
        'days': 306
    },
    '5m_tuned': {
        'workspace': '/Users/daws/ADMF-PC/workspaces/signal_generation_1135e2a8',
        'timeframe_minutes': 5,
        'total_bars': 16614,
        'trades': 63,
        'win_rate': 0.635,
        'annual_return': 0.0186,
        'total_return': 0.0156,
        'days': 306
    },
    '15m_basic': {
        'workspace': '/Users/daws/ADMF-PC/workspaces/signal_generation_bc947151',
        'timeframe_minutes': 15,
        'total_bars': 5673,
        'trades': 42,
        'win_rate': 0.619,
        'annual_return': 0.0272,
        'total_return': 0.0227,
        'days': 306
    },
    '15m_optimized': {
        'workspace': '/Users/daws/ADMF-PC/workspaces/signal_generation_5d710d47',
        'timeframe_minutes': 15,
        'total_bars': 5673,
        'trades': 16,
        'win_rate': 0.562,
        'annual_return': -0.0038,
        'total_return': -0.0032,
        'days': 306
    }
}

def calculate_sharpe_proper(workspace_path):
    """Calculate proper Sharpe ratio from daily returns."""
    traces_dir = Path(workspace_path) / 'traces'
    signal_files = list(traces_dir.rglob('*.parquet'))
    if not signal_files:
        return 0
    
    df = pd.read_parquet(signal_files[0])
    
    # Build list of trades with dates
    trades = []
    current_position = None
    
    for i in range(len(df)):
        row = df.iloc[i]
        signal = row['val']
        price = row['px']
        timestamp = pd.to_datetime(row['ts'])
        
        if current_position is None and signal != 0:
            current_position = {
                'entry_price': price,
                'entry_time': timestamp,
                'direction': signal
            }
        elif current_position is not None and (signal == 0 or signal != current_position['direction']):
            exit_price = price
            entry_price = current_position['entry_price']
            
            if current_position['direction'] > 0:
                gross_return = (exit_price / entry_price) - 1
            else:
                gross_return = (entry_price / exit_price) - 1
            
            net_return = gross_return - 0.0001  # 1bp round trip
            
            trades.append({
                'date': timestamp.date(),
                'return': net_return
            })
            
            if signal != 0 and signal != current_position['direction']:
                current_position = {
                    'entry_price': price,
                    'entry_time': timestamp,
                    'direction': signal
                }
            else:
                current_position = None
    
    if not trades:
        return 0
    
    # Aggregate by day
    trades_df = pd.DataFrame(trades)
    daily_returns = trades_df.groupby('date')['return'].apply(lambda x: (1 + x).prod() - 1)
    
    # Fill missing days with 0
    all_dates = pd.date_range(start=daily_returns.index.min(), end=daily_returns.index.max(), freq='D')
    daily_returns = daily_returns.reindex(all_dates, fill_value=0)
    
    # Calculate Sharpe
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe = 0
    
    return sharpe

# Calculate Sharpe ratios
print("Calculating Sharpe ratios...")
for name, data in strategies.items():
    data['sharpe'] = calculate_sharpe_proper(data['workspace'])

# Display comprehensive report
print("\n=== COMPREHENSIVE STRATEGY METRICS ===")
print("Execution cost: 0.5 bps per trade | Dataset: TRAIN (80%)\n")

print(f"{'Strategy':<15} {'Win Rate':<10} {'Trades/Day':<12} {'Avg Return':<12} {'Sharpe':<8}")
print("=" * 60)

for name, data in strategies.items():
    trades_per_day = data['trades'] / data['days']
    avg_return_per_trade = data['total_return'] / data['trades'] if data['trades'] > 0 else 0
    
    print(f"{name:<15} {data['win_rate']*100:<10.1f}% {trades_per_day:<12.2f} "
          f"{avg_return_per_trade*100:<12.3f}% {data['sharpe']:<8.2f}")

print("\n=== ADDITIONAL METRICS ===\n")

print("Trading Frequency:")
for name, data in strategies.items():
    hours_between_trades = (data['days'] * 24) / data['trades'] if data['trades'] > 0 else 0
    print(f"  {name}: One trade every {hours_between_trades:.1f} hours")

print("\nAnnualized Returns:")
for name, data in strategies.items():
    print(f"  {name}: {data['annual_return']*100:.2f}% per year")

print("\nRisk-Adjusted Rankings:")
ranked = sorted(strategies.items(), key=lambda x: x[1]['sharpe'], reverse=True)
print("\nBy Sharpe Ratio:")
for i, (name, data) in enumerate(ranked):
    print(f"  {i+1}. {name}: {data['sharpe']:.2f}")

print("\nBy Annual Return:")
ranked = sorted(strategies.items(), key=lambda x: x[1]['annual_return'], reverse=True)
for i, (name, data) in enumerate(ranked):
    print(f"  {i+1}. {name}: {data['annual_return']*100:.2f}%")

print("\n=== SUMMARY TABLE ===\n")
print(f"{'Metric':<20} {'5m Basic':<12} {'5m Tuned':<12} {'15m Basic':<12} {'15m Opt':<12}")
print("-" * 68)
metrics = [
    ('Win Rate (%)', 'win_rate', 100),
    ('Trades/Day', 'trades_per_day', 1),
    ('Avg Return/Trade (%)', 'avg_return', 100),
    ('Annual Return (%)', 'annual_return', 100),
    ('Sharpe Ratio', 'sharpe', 1)
]

for metric_name, metric_key, multiplier in metrics:
    values = []
    for name in ['5m_basic', '5m_tuned', '15m_basic', '15m_optimized']:
        if metric_key == 'trades_per_day':
            val = strategies[name]['trades'] / strategies[name]['days']
        elif metric_key == 'avg_return':
            val = strategies[name]['total_return'] / strategies[name]['trades']
        elif metric_key in strategies[name]:
            val = strategies[name][metric_key]
        else:
            val = 0
        values.append(f"{val*multiplier:.2f}")
    
    print(f"{metric_name:<20} {values[0]:<12} {values[1]:<12} {values[2]:<12} {values[3]:<12}")