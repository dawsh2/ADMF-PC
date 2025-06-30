#!/usr/bin/env python3
"""Analyze the keltner winner config trace files"""

import pandas as pd
import json
from pathlib import Path

# Load the trace files
trace_dir = Path('config/keltner/config_winrar/results/20250622_210552/traces/mean_reversion')
p50_file = trace_dir / 'SPY_5m_kb_winner_p50_m1.parquet'
p30_file = trace_dir / 'SPY_5m_kb_winner_p30_m1.parquet'

# Load metadata
metadata_file = Path('config/keltner/config_winrar/results/20250622_210552/metadata.json')
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

print('KELTNER WINNER CONFIG - RESULTS ANALYSIS')
print('='*60)
print(f"Workflow ID: {metadata['workflow_id']}")
print(f"Total bars: {metadata['total_bars']:,}")
print(f"Total signals: {metadata['total_signals']:,}")

# Analyze each strategy
results = []
for name, file_path in [('Period 50, Mult 1.0', p50_file), ('Period 30, Mult 1.0', p30_file)]:
    print(f'\n{name} Strategy:')
    print('-'*40)
    
    if not file_path.exists():
        print(f"  ERROR: File not found - {file_path}")
        continue
        
    # Load trace
    df = pd.read_parquet(file_path)
    print(f"  Trace records: {len(df)}")
    
    # Count trades
    trades = 0
    in_trade = False
    trade_returns = []
    
    for i in range(len(df)):
        signal = df.iloc[i]['val']
        if not in_trade and signal != 0:
            trades += 1
            in_trade = True
            entry_price = df.iloc[i]['px']
            trade_direction = signal
        elif in_trade and signal == 0:
            in_trade = False
            exit_price = df.iloc[i]['px']
            # Calculate return
            if trade_direction > 0:  # Long
                ret = (exit_price - entry_price) / entry_price
            else:  # Short
                ret = (entry_price - exit_price) / entry_price
            trade_returns.append(ret)
    
    # Get component info
    comp_name = file_path.stem
    if comp_name in metadata['components']:
        comp_data = metadata['components'][comp_name]
        print(f"  Signal changes: {comp_data['signal_changes']:,}")
        print(f"  Signal frequency: {comp_data['signal_frequency']*100:.2f}%")
        print(f"  Completed trades: {trades}")
        
        # Trading days (5-min bars)
        bars_per_day = 78  # 6.5 hours * 12 (5-min bars per hour)
        trading_days = metadata['total_bars'] / bars_per_day
        trades_per_day = trades / trading_days if trading_days > 0 else 0
        print(f"  Trades per day: {trades_per_day:.2f}")
        
        # Performance
        if trade_returns:
            avg_return = sum(trade_returns) / len(trade_returns) * 100
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100
            total_return = sum(trade_returns) * 100
            print(f"  Avg return per trade: {avg_return:.3f}%")
            print(f"  Win rate: {win_rate:.1f}%")
            print(f"  Total return: {total_return:.2f}%")
            
            # Annualized
            annual_return = (total_return / trading_days) * 252
            print(f"  Annualized return (simple): {annual_return:.2f}%")
        
        results.append({
            'strategy': name,
            'trades': trades,
            'trades_per_day': trades_per_day,
            'signal_freq': comp_data['signal_frequency']*100
        })
    
    # Check signal distribution
    if len(df) > 0:
        long_signals = (df['val'] > 0).sum()
        short_signals = (df['val'] < 0).sum()
        neutral = (df['val'] == 0).sum()
        print(f"  Signal distribution: Long={long_signals}, Short={short_signals}, Neutral={neutral}")

print('\n' + '='*60)
print('SUMMARY:')
for r in results:
    print(f"{r['strategy']}: {r['trades_per_day']:.2f} trades/day, {r['signal_freq']:.2f}% signal frequency")

print('\nCOMPARISON TO ORIGINAL ANALYSIS:')
print('Original Period 50, Mult 1.0 (with filters): 1.89 trades/day, 16.92% annual')
print('Original Period 30, Mult 1.0 (with filters): ~4-8 trades/day')
print('\nNote: These results are WITHOUT filters. The original top performers')
print('included specific filter combinations (e.g., filter ID 41 for the top strategy).')