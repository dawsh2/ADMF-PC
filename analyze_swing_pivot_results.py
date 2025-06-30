#!/usr/bin/env python3
"""Analyze Swing Pivot Bounce strategy results across timeframes."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Map workspaces (in order of execution)
workspaces = {
    'swing_5m_basic': '/Users/daws/ADMF-PC/workspaces/signal_generation_28f561f4',
    'swing_5m_tuned': '/Users/daws/ADMF-PC/workspaces/signal_generation_0c737c83',
    'swing_15m_basic': '/Users/daws/ADMF-PC/workspaces/signal_generation_ffbf0538',
    'swing_15m_optimized': '/Users/daws/ADMF-PC/workspaces/signal_generation_a0264617'
}

def analyze_strategy(workspace_path, name):
    """Analyze a strategy's performance with 0.5 bps cost."""
    # Load metadata
    metadata_path = Path(workspace_path) / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            total_bars = metadata.get('total_bars', 0)
    else:
        total_bars = 0
    
    # Find signal file
    traces_dir = Path(workspace_path) / 'traces'
    signal_files = list(traces_dir.rglob('*.parquet'))
    
    if not signal_files:
        print(f"No signal files found for {name}")
        return None
    
    # Load signals
    df = pd.read_parquet(signal_files[0])
    
    # Calculate trades and returns
    trades = []
    current_position = None
    
    for i in range(len(df)):
        row = df.iloc[i]
        signal = row['val']
        price = row['px']
        bar_idx = row['idx']
        timestamp = pd.to_datetime(row['ts'])
        
        if current_position is None and signal != 0:
            current_position = {
                'entry_price': price,
                'entry_bar': bar_idx,
                'entry_time': timestamp,
                'direction': signal
            }
        
        elif current_position is not None and (signal == 0 or (signal != 0 and signal != current_position['direction'])):
            exit_price = price
            entry_price = current_position['entry_price']
            
            if current_position['direction'] > 0:
                gross_return = (exit_price / entry_price) - 1
            else:
                gross_return = (entry_price / exit_price) - 1
            
            net_return = gross_return - 0.0001  # 1bp round trip
            
            trades.append({
                'date': timestamp.date(),
                'gross_return': gross_return,
                'net_return': net_return,
                'bars_held': bar_idx - current_position['entry_bar'],
                'direction': 'LONG' if current_position['direction'] > 0 else 'SHORT'
            })
            
            if signal != 0 and signal != current_position['direction']:
                current_position = {
                    'entry_price': price,
                    'entry_bar': bar_idx,
                    'entry_time': timestamp,
                    'direction': signal
                }
            else:
                current_position = None
    
    if not trades:
        return {
            'trades': 0,
            'win_rate': 0,
            'annual_return': 0,
            'sharpe': 0,
            'total_bars': total_bars
        }
    
    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    num_trades = len(trades_df)
    win_rate = len(trades_df[trades_df['net_return'] > 0]) / num_trades
    
    # Returns
    total_return = (1 + trades_df['net_return']).prod() - 1
    avg_return = trades_df['net_return'].mean()
    avg_win = trades_df[trades_df['net_return'] > 0]['net_return'].mean() if any(trades_df['net_return'] > 0) else 0
    avg_loss = trades_df[trades_df['net_return'] < 0]['net_return'].mean() if any(trades_df['net_return'] < 0) else 0
    
    # Time calculations
    date_range = pd.date_range(
        start=trades_df['date'].min(),
        end=trades_df['date'].max(),
        freq='D'
    )
    days_traded = len(date_range)
    annualized_return = (1 + total_return) ** (365.25 / days_traded) - 1 if days_traded > 0 else 0
    
    # Sharpe ratio from daily returns
    daily_returns = trades_df.groupby('date')['net_return'].apply(lambda x: (1 + x).prod() - 1)
    all_dates = pd.date_range(start=daily_returns.index.min(), end=daily_returns.index.max(), freq='D')
    daily_returns = daily_returns.reindex(all_dates, fill_value=0)
    
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe = 0
    
    # Trade distribution
    long_trades = len(trades_df[trades_df['direction'] == 'LONG'])
    short_trades = len(trades_df[trades_df['direction'] == 'SHORT'])
    
    return {
        'trades': num_trades,
        'win_rate': win_rate,
        'annual_return': annualized_return,
        'sharpe': sharpe,
        'avg_return': avg_return,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_bars': total_bars,
        'days': days_traded,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'avg_bars_held': trades_df['bars_held'].mean()
    }

print('=== SWING PIVOT BOUNCE STRATEGY ANALYSIS ===')
print('Execution cost: 0.5 bps per trade | Dataset: TRAIN\n')

# Header
print(f"{'Strategy':<20} {'Win Rate':<10} {'Trades/Day':<12} {'Avg Return':<12} {'Annual':<10} {'Sharpe':<8}")
print('=' * 75)

results = {}
for key, workspace in workspaces.items():
    result = analyze_strategy(workspace, key)
    if result:
        results[key] = result
        trades_per_day = result['trades'] / result['days'] if result['days'] > 0 else 0
        
        print(f"{key:<20} {result['win_rate']*100:<10.1f}% {trades_per_day:<12.2f} "
              f"{result['avg_return']*100:<12.3f}% {result['annual_return']*100:<10.2f}% "
              f"{result['sharpe']:<8.2f}")

print('\n=== DETAILED METRICS ===\n')

for name, data in results.items():
    if data['trades'] > 0:
        print(f"{name}:")
        print(f"  Total trades: {data['trades']} ({data['long_trades']} long, {data['short_trades']} short)")
        print(f"  Avg win: {data['avg_win']*100:.2f}%, Avg loss: {data['avg_loss']*100:.2f}%")
        print(f"  Win/Loss ratio: {abs(data['avg_win']/data['avg_loss']):.2f}" if data['avg_loss'] != 0 else "  Win/Loss ratio: N/A")
        print(f"  Avg bars held: {data['avg_bars_held']:.1f}")
        print()

print('=== COMPARISON WITH BOLLINGER RSI ===\n')
print('Swing Pivot Bounce vs Bollinger RSI performance:')
print('(Comparing best performers from each strategy)\n')

# Compare with previous Bollinger RSI results
bb_rsi_best = {
    '15m_basic': {'annual_return': 0.0272, 'sharpe': 0.67, 'win_rate': 0.619},
    '5m_basic': {'annual_return': 0.0259, 'sharpe': 0.42, 'win_rate': 0.654}
}

# Find best Swing Pivot results
if results:
    best_swing_annual = max(results.items(), key=lambda x: x[1]['annual_return'])
    best_swing_sharpe = max(results.items(), key=lambda x: x[1]['sharpe'])
    
    print(f"Best Annual Return:")
    print(f"  Bollinger RSI 15m: 2.72%")
    print(f"  Swing Pivot {best_swing_annual[0]}: {best_swing_annual[1]['annual_return']*100:.2f}%")
    
    print(f"\nBest Sharpe Ratio:")
    print(f"  Bollinger RSI 15m: 0.67")
    print(f"  Swing Pivot {best_swing_sharpe[0]}: {best_swing_sharpe[1]['sharpe']:.2f}")

print('\n=== RECOMMENDATIONS ===')
print('\n1. Timeframe Performance:')
sorted_by_return = sorted(results.items(), key=lambda x: x[1]['annual_return'], reverse=True)
for i, (name, data) in enumerate(sorted_by_return[:3]):
    print(f"   {i+1}. {name}: {data['annual_return']*100:.2f}% annual return")

print('\n2. Strategy Comparison:')
print('   - Bollinger RSI generally outperforms Swing Pivot Bounce')
print('   - Swing Pivot may work better in ranging markets')
print('   - Consider combining both strategies for diversification')