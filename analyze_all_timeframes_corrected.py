#!/usr/bin/env python3
"""Compare performance across 5m and 15m timeframes with correct dataset splits."""

import pandas as pd
import numpy as np
from pathlib import Path

# Updated workspace mappings
workspaces = {
    '5m_basic': '/Users/daws/ADMF-PC/workspaces/signal_generation_31415f83',
    '5m_tuned': '/Users/daws/ADMF-PC/workspaces/signal_generation_1135e2a8',
    '15m_basic': '/Users/daws/ADMF-PC/workspaces/signal_generation_bc947151',
    '15m_optimized': '/Users/daws/ADMF-PC/workspaces/signal_generation_5d710d47'
}

def analyze_strategy(workspace_path, name):
    """Analyze a single strategy's performance."""
    # Find the signal file
    traces_dir = Path(workspace_path) / 'traces'
    signal_files = list(traces_dir.rglob('*.parquet'))
    
    if not signal_files:
        print(f"No signal files found for {name}")
        return None
    
    # Load metadata first
    metadata_path = Path(workspace_path) / 'metadata.json'
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            total_bars_analyzed = metadata.get('total_bars', 0)
    else:
        total_bars_analyzed = 0
    
    # Load the signal file
    df = pd.read_parquet(signal_files[0])
    
    # Calculate returns with 0.5 bps cost
    trades = []
    current_position = None
    total_log_return = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        signal = row['val']
        price = row['px']
        bar_idx = row['idx']
        
        if current_position is None and signal != 0:
            current_position = {
                'entry_price': price,
                'entry_bar': bar_idx,
                'direction': signal
            }
        
        elif current_position is not None and (signal == 0 or (signal != 0 and signal != current_position['direction'])):
            exit_price = price
            entry_price = current_position['entry_price']
            
            if current_position['direction'] > 0:
                gross_log_return = np.log(exit_price / entry_price)
            else:
                gross_log_return = np.log(entry_price / exit_price)
            
            # Apply 0.5 bps cost each way (1 bp round trip)
            cost_multiplier = 1 - 0.0001
            net_log_return = gross_log_return + np.log(cost_multiplier)
            total_log_return += net_log_return
            
            trades.append({
                'gross_return': np.exp(gross_log_return) - 1,
                'net_return': np.exp(net_log_return) - 1,
                'bars_held': bar_idx - current_position['entry_bar']
            })
            
            if signal != 0 and signal != current_position['direction']:
                current_position = {
                    'entry_price': price,
                    'entry_bar': bar_idx,
                    'direction': signal
                }
            else:
                current_position = None
    
    if not trades:
        return None
    
    # Calculate statistics
    trades_df = pd.DataFrame(trades)
    num_trades = len(trades_df)
    win_rate = len(trades_df[trades_df['net_return'] > 0]) / num_trades
    
    # Total return
    total_return = np.exp(total_log_return) - 1
    
    # Annualize (assume 306 days for all)
    annualized_return = (1 + total_return) ** (365.25 / 306) - 1
    
    # Average trade metrics
    avg_win = trades_df[trades_df['net_return'] > 0]['net_return'].mean() if any(trades_df['net_return'] > 0) else 0
    avg_loss = trades_df[trades_df['net_return'] < 0]['net_return'].mean() if any(trades_df['net_return'] < 0) else 0
    avg_bars = trades_df['bars_held'].mean()
    
    # Sharpe calculation
    if len(trades_df) > 1:
        returns_std = trades_df['net_return'].std()
        avg_return = trades_df['net_return'].mean()
        sharpe = np.sqrt(252) * avg_return / returns_std if returns_std > 0 else 0
    else:
        sharpe = 0
    
    return {
        'trades': num_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'annual_return': annualized_return,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_bars_held': avg_bars,
        'signal_changes': len(df),
        'total_bars': total_bars_analyzed,
        'sharpe': sharpe
    }

print('=== CORRECTED TIMEFRAME COMPARISON: 5m vs 15m ===')
print('Dataset: TRAIN (80% of data)')
print('Execution cost: 0.5 bps per trade\n')

print(f"{'Strategy':<20} {'Bars':<10} {'Trades':<8} {'Win%':<8} {'Annual':<10} {'Sharpe':<8} {'Avg Win':<10} {'Avg Loss':<10} {'Bars/Trade':<12}")
print('-' * 110)

results = {}
for key, workspace in workspaces.items():
    result = analyze_strategy(workspace, key)
    if result:
        results[key] = result
        print(f"{key:<20} {result['total_bars']:<10} {result['trades']:<8} {result['win_rate']*100:<8.1f} "
              f"{result['annual_return']*100:<10.2f} {result['sharpe']:<8.2f} "
              f"{result['avg_win']*100:<10.2f} {result['avg_loss']*100:<10.2f} {result['avg_bars_held']:<12.1f}")

print('\n=== KEY INSIGHTS ===\n')

# Trade frequency analysis
print('1. Bars Analyzed:')
for key in results:
    print(f'   {key}: {results[key]["total_bars"]:,} bars')

print('\n2. Trade Frequency:')
timeframe_hours = {
    '5m': 5/60,
    '15m': 15/60
}

for key in results:
    timeframe = key.split('_')[0]
    hours_per_bar = timeframe_hours.get(timeframe, 1/60)
    avg_hours = results[key]['avg_bars_held'] * hours_per_bar
    trade_freq = results[key]['total_bars'] / results[key]['trades'] if results[key]['trades'] > 0 else 0
    freq_hours = trade_freq * hours_per_bar
    print(f'   {key}: One trade every {freq_hours:.1f} hours ({trade_freq:.0f} bars)')

print('\n3. Best Performers (Annual Return):')
sorted_results = sorted(results.items(), key=lambda x: x[1]['annual_return'], reverse=True)
for i, (key, result) in enumerate(sorted_results):
    print(f'   {i+1}. {key}: {result["annual_return"]*100:.2f}% (Sharpe: {result["sharpe"]:.2f})')

print('\n4. Trade Count Comparison:')
print('   5m Basic generated much more trades with proper dataset!')
print(f'   5m: ~{results["5m_basic"]["trades"]} trades vs 15m: ~{results["15m_basic"]["trades"]} trades')
print(f'   Ratio: {results["5m_basic"]["trades"]/results["15m_basic"]["trades"]:.1f}x more trades on 5m')

print('\n5. Final Recommendation:')
best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe'])
best_return = max(results.items(), key=lambda x: x[1]['annual_return'])
print(f'   Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]["sharpe"]:.2f})')
print(f'   Best Annual Return: {best_return[0]} ({best_return[1]["annual_return"]*100:.2f}%)')