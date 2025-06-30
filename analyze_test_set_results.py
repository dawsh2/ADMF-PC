#!/usr/bin/env python3
"""Analyze test set results for keltner winner config"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

# Test set results
test_dir = Path('config/keltner/config_winrar/results/20250622_210842')
test_metadata_file = test_dir / 'metadata.json'
test_traces_dir = test_dir / 'traces' / 'mean_reversion'

# Train set results for comparison
train_dir = Path('config/keltner/config_winrar/results/20250622_210552')
train_metadata_file = train_dir / 'metadata.json'

# Load metadata
with open(test_metadata_file, 'r') as f:
    test_metadata = json.load(f)

with open(train_metadata_file, 'r') as f:
    train_metadata = json.load(f)

print('KELTNER WINNER - TEST SET PERFORMANCE')
print('='*80)
print(f"Test bars: {test_metadata['total_bars']:,} | Train bars: {train_metadata['total_bars']:,}")
print(f"Test signals: {test_metadata['total_signals']:,} | Train signals: {train_metadata['total_signals']:,}")

def analyze_strategy(trace_file, metadata, dataset_name):
    """Analyze a single strategy trace file"""
    df = pd.read_parquet(trace_file)
    
    # Count trades and calculate returns
    trades = 0
    in_trade = False
    trade_returns = []
    entry_price = None
    trade_direction = 0
    
    for i in range(len(df)):
        signal = df.iloc[i]['val']
        price = df.iloc[i]['px']
        
        if not in_trade and signal != 0:
            trades += 1
            in_trade = True
            entry_price = price
            trade_direction = signal
        elif in_trade and signal == 0:
            in_trade = False
            if entry_price is not None:
                if trade_direction > 0:  # Long
                    ret = (price - entry_price) / entry_price
                else:  # Short
                    ret = (entry_price - price) / entry_price
                trade_returns.append(ret)
    
    # Calculate metrics
    comp_name = trace_file.stem
    comp_data = metadata['components'][comp_name]
    
    # Trading days (5-min bars)
    bars_per_day = 78  # 6.5 hours * 12
    trading_days = metadata['total_bars'] / bars_per_day
    trades_per_day = trades / trading_days if trading_days > 0 else 0
    
    # Performance
    if trade_returns:
        avg_return = np.mean(trade_returns) * 100
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100
        total_return = sum(trade_returns) * 100
        
        # Simple annualization
        annual_return_simple = (total_return / trading_days) * 252
        
        # Compound annualization
        cumulative = 1.0
        for r in trade_returns:
            cumulative *= (1 + r)
        total_return_compound = (cumulative - 1) * 100
        if trading_days > 0:
            years = trading_days / 252
            annual_return_compound = ((cumulative ** (1/years)) - 1) * 100
        else:
            annual_return_compound = 0
    else:
        avg_return = 0
        win_rate = 0
        total_return = 0
        total_return_compound = 0
        annual_return_simple = 0
        annual_return_compound = 0
    
    return {
        'dataset': dataset_name,
        'trades': trades,
        'trades_per_day': trades_per_day,
        'signal_freq': comp_data['signal_frequency'] * 100,
        'avg_return': avg_return,
        'win_rate': win_rate,
        'total_return': total_return,
        'total_return_compound': total_return_compound,
        'annual_simple': annual_return_simple,
        'annual_compound': annual_return_compound
    }

# Analyze both strategies on both datasets
results = []

# Test set
for name, filename in [('P50_M1', 'SPY_5m_kb_winner_p50_m1.parquet'), 
                       ('P30_M1', 'SPY_5m_kb_winner_p30_m1.parquet')]:
    test_file = test_traces_dir / filename
    if test_file.exists():
        result = analyze_strategy(test_file, test_metadata, 'TEST')
        result['strategy'] = name
        results.append(result)

# Train set (for comparison)
train_traces_dir = train_dir / 'traces' / 'mean_reversion'
for name, filename in [('P50_M1', 'SPY_5m_kb_winner_p50_m1.parquet'), 
                       ('P30_M1', 'SPY_5m_kb_winner_p30_m1.parquet')]:
    train_file = train_traces_dir / filename
    if train_file.exists():
        result = analyze_strategy(train_file, train_metadata, 'TRAIN')
        result['strategy'] = name
        results.append(result)

# Display results
print('\n' + '-'*80)
print('DETAILED RESULTS:')
print('-'*80)

for strategy in ['P50_M1', 'P30_M1']:
    print(f'\nStrategy: Period={strategy[1:3]}, Multiplier=1.0')
    print('='*60)
    
    train_data = next((r for r in results if r['strategy'] == strategy and r['dataset'] == 'TRAIN'), None)
    test_data = next((r for r in results if r['strategy'] == strategy and r['dataset'] == 'TEST'), None)
    
    if train_data and test_data:
        print(f"{'Metric':<25} {'TRAIN':>15} {'TEST':>15} {'Difference':>15}")
        print('-'*70)
        
        metrics = [
            ('Trades per day', 'trades_per_day', '.2f'),
            ('Signal frequency %', 'signal_freq', '.2f'),
            ('Win rate %', 'win_rate', '.1f'),
            ('Avg return per trade %', 'avg_return', '.3f'),
            ('Total return %', 'total_return', '.2f'),
            ('Annual return % (simple)', 'annual_simple', '.2f'),
            ('Annual return % (compound)', 'annual_compound', '.2f')
        ]
        
        for label, key, fmt in metrics:
            train_val = train_data[key]
            test_val = test_data[key]
            diff = test_val - train_val
            print(f"{label:<25} {train_val:>15{fmt}} {test_val:>15{fmt}} {diff:>15{fmt}}")

print('\n' + '='*80)
print('SUMMARY:')
print('='*80)

# Calculate average performance degradation
avg_train_return = np.mean([r['annual_compound'] for r in results if r['dataset'] == 'TRAIN'])
avg_test_return = np.mean([r['annual_compound'] for r in results if r['dataset'] == 'TEST'])

print(f'Average Annual Return (Compound):')
print(f'  Train: {avg_train_return:.2f}%')
print(f'  Test:  {avg_test_return:.2f}%')
print(f'  Degradation: {avg_test_return - avg_train_return:.2f}%')

print('\nConclusion:')
if avg_test_return > 0:
    print('✓ Strategies remain profitable on test set')
    if abs(avg_test_return - avg_train_return) < 5:
        print('✓ Performance is relatively consistent between train and test')
    else:
        print('⚠ Significant performance degradation on test set')
else:
    print('✗ Strategies are not profitable on test set')