#!/usr/bin/env python3
"""Analyze the 15-minute strategy results."""

import pandas as pd
import numpy as np

# Load signal data from both runs - using sparse format
basic_signals = pd.read_parquet('/Users/daws/ADMF-PC/workspaces/signal_generation_bc947151/traces/SPY_15m_1m/signals/bollinger_rsi_simple_signals/SPY_15m_compiled_strategy_0.parquet')
optimized_signals = pd.read_parquet('/Users/daws/ADMF-PC/workspaces/signal_generation_5d710d47/traces/SPY_15m_1m/signals/bollinger_rsi_simple_signals/SPY_15m_compiled_strategy_0.parquet')

print('=== 15-MINUTE STRATEGY PERFORMANCE COMPARISON ===')
print('Data format: SPARSE (only signal changes stored)\n')

results = {}

for name, df in [('Basic 15m', basic_signals), ('Optimized 15m', optimized_signals)]:
    print(f'\n{name} Configuration:')
    print('-' * 50)
    
    # Display columns
    print(f'Columns: {df.columns.tolist()}')
    print(f'Shape: {df.shape}')
    
    # In sparse format, each row is a signal CHANGE
    signal_changes = len(df)
    print(f'Total signal changes: {signal_changes}')
    
    # Count actual trades (transitions from 0 to non-zero)
    trades = []
    positions = []
    
    for i in range(len(df)):
        current_row = df.iloc[i]
        current_val = current_row['val']
        prev_val = df.iloc[i-1]['val'] if i > 0 else 0
        
        # Track all position changes
        positions.append({
            'bar': current_row['idx'],
            'value': current_val,
            'price': current_row['px'],
            'timestamp': current_row['ts']
        })
        
        # New position opened (0 -> non-zero)
        if prev_val == 0 and current_val != 0:
            trades.append({
                'entry_idx': current_row['idx'],
                'entry_price': current_row['px'],
                'direction': 'LONG' if current_val > 0 else 'SHORT',
                'entry_val': current_val,
                'timestamp': current_row['ts']
            })
    
    print(f'Total trades opened: {len(trades)}')
    
    # Analyze trade characteristics
    if trades:
        long_trades = sum(1 for t in trades if t['direction'] == 'LONG')
        short_trades = sum(1 for t in trades if t['direction'] == 'SHORT')
        
        print(f'\nTrade breakdown:')
        print(f'  Long trades: {long_trades} ({long_trades/len(trades):.1%})')
        print(f'  Short trades: {short_trades} ({short_trades/len(trades):.1%})')
        
        # Calculate trade frequency
        if len(df) > 1:
            first_bar = df.iloc[0]['idx']
            last_bar = df.iloc[-1]['idx']
            total_bars = last_bar - first_bar
            
            if len(trades) > 0:
                bars_per_trade = total_bars / len(trades)
                hours_per_trade = (bars_per_trade * 15) / 60  # 15 min bars
                
                print(f'\nTrade frequency:')
                print(f'  Total bars analyzed: {total_bars}')
                print(f'  Average bars between trades: {bars_per_trade:.0f}')
                print(f'  Average hours between trades: {hours_per_trade:.1f}')
        
        # Show sample trades
        print(f'\nFirst 5 trades:')
        for i, trade in enumerate(trades[:5]):
            print(f"  Trade {i+1}: {trade['direction']} at ${trade['entry_price']:.2f} (bar {trade['entry_idx']})")
        
        # Store results
        results[name] = {
            'signal_changes': signal_changes,
            'total_trades': len(trades),
            'long_trades': long_trades,
            'short_trades': short_trades,
            'trades': trades
        }

# Compare the two approaches
print('\n\n=== COMPARISON SUMMARY ===')
print('\nBasic 15m:')
print(f"  - Signal changes: {results['Basic 15m']['signal_changes']}")
print(f"  - Total trades: {results['Basic 15m']['total_trades']}")
print(f"  - Long/Short split: {results['Basic 15m']['long_trades']}/{results['Basic 15m']['short_trades']}")

print('\nOptimized 15m:')
print(f"  - Signal changes: {results['Optimized 15m']['signal_changes']}")
print(f"  - Total trades: {results['Optimized 15m']['total_trades']}")
print(f"  - Long/Short split: {results['Optimized 15m']['long_trades']}/{results['Optimized 15m']['short_trades']}")

reduction = (1 - results['Optimized 15m']['total_trades'] / results['Basic 15m']['total_trades']) * 100
print(f'\nThe optimized version reduced trades by {reduction:.0f}%')
print('This means SIGNIFICANTLY lower execution costs!')

# Calculate potential impact
print('\n=== EXECUTION COST IMPACT ===')
print('\nAssuming 0.05% slippage + $1 commission per trade:')
for name in ['Basic 15m', 'Optimized 15m']:
    trades = results[name]['total_trades']
    # Rough estimate: 0.05% slippage each way + $1 commission
    cost_per_trade = 0.001 + (1/500)  # 0.1% + $1/$500 price
    total_cost = trades * cost_per_trade
    print(f'{name}: {trades} trades × {cost_per_trade:.2%} = {total_cost:.1%} total cost')