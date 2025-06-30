#!/usr/bin/env python3
"""Deep dive into Sharpe ratio calculations."""

import pandas as pd
import numpy as np
from pathlib import Path

# Workspace paths
workspaces = {
    '5m_basic': '/Users/daws/ADMF-PC/workspaces/signal_generation_31415f83',
    '5m_tuned': '/Users/daws/ADMF-PC/workspaces/signal_generation_1135e2a8',
    '15m_basic': '/Users/daws/ADMF-PC/workspaces/signal_generation_bc947151',
}

def calculate_sharpe_detailed(workspace_path, name):
    """Calculate Sharpe ratio with different methods."""
    # Load signal file
    traces_dir = Path(workspace_path) / 'traces'
    signal_files = list(traces_dir.rglob('*.parquet'))
    if not signal_files:
        return None
    
    df = pd.read_parquet(signal_files[0])
    
    # Extract trades
    trades = []
    current_position = None
    
    for i in range(len(df)):
        row = df.iloc[i]
        signal = row['val']
        price = row['px']
        bar_idx = row['idx']
        timestamp = row['ts']
        
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
            
            # Calculate returns
            if current_position['direction'] > 0:
                gross_return = (exit_price / entry_price) - 1
            else:
                gross_return = (entry_price / exit_price) - 1
            
            # Apply costs
            net_return = gross_return - 0.0001  # 1bp round trip
            
            trades.append({
                'entry_time': current_position['entry_time'],
                'exit_time': timestamp,
                'gross_return': gross_return,
                'net_return': net_return,
                'bars_held': bar_idx - current_position['entry_bar'],
                'entry_bar': current_position['entry_bar'],
                'exit_bar': bar_idx
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
        return None
    
    trades_df = pd.DataFrame(trades)
    
    # Method 1: Sharpe based on per-trade returns (what I was using)
    avg_return = trades_df['net_return'].mean()
    std_return = trades_df['net_return'].std()
    sharpe_per_trade = avg_return / std_return if std_return > 0 else 0
    
    # Method 2: Time-based Sharpe (more accurate)
    # Convert to time series of daily returns
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_time']).dt.date
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_time']).dt.date
    
    # Get date range
    all_dates = pd.date_range(
        start=trades_df['entry_date'].min(),
        end=trades_df['exit_date'].max(),
        freq='D'
    )
    
    # Create daily returns series
    daily_returns = pd.Series(0.0, index=all_dates)
    
    # Distribute returns to exit dates
    for _, trade in trades_df.iterrows():
        exit_date = pd.Timestamp(trade['exit_date'])
        if exit_date in daily_returns.index:
            daily_returns[exit_date] += trade['net_return']
    
    # Calculate annualized Sharpe
    daily_mean = daily_returns.mean()
    daily_std = daily_returns.std()
    sharpe_daily = np.sqrt(252) * daily_mean / daily_std if daily_std > 0 else 0
    
    # Method 3: Based on holding period
    # Calculate average trades per year
    total_days = len(all_dates)
    trades_per_year = len(trades_df) * 365 / total_days
    
    # Annualize the per-trade Sharpe
    sharpe_annualized = sharpe_per_trade * np.sqrt(trades_per_year)
    
    print(f"\n{name}:")
    print(f"  Total trades: {len(trades_df)}")
    print(f"  Avg return per trade: {avg_return*100:.3f}%")
    print(f"  Std dev per trade: {std_return*100:.3f}%")
    print(f"  Trading days: {total_days}")
    print(f"  Trades per year: {trades_per_year:.1f}")
    
    print(f"\nSharpe Calculations:")
    print(f"  Method 1 (per-trade, raw): {sharpe_per_trade:.3f}")
    print(f"  Method 2 (daily returns): {sharpe_daily:.3f}")
    print(f"  Method 3 (annualized per-trade): {sharpe_annualized:.3f}")
    
    # Check timeframe impact
    timeframe = name.split('_')[0]
    bars_per_day = {'5m': 78, '15m': 26}  # Approximate trading bars per day
    avg_bars_held = trades_df['bars_held'].mean()
    avg_days_held = avg_bars_held / bars_per_day.get(timeframe, 78)
    
    print(f"\nHolding period analysis:")
    print(f"  Avg bars held: {avg_bars_held:.1f}")
    print(f"  Avg days held: {avg_days_held:.1f}")
    
    return {
        'sharpe_per_trade': sharpe_per_trade,
        'sharpe_daily': sharpe_daily,
        'sharpe_annualized': sharpe_annualized,
        'trades': len(trades_df),
        'avg_return': avg_return,
        'std_return': std_return
    }

print("=== SHARPE RATIO ANALYSIS ===")
print("Investigating why 5m basic has low Sharpe...\n")

results = {}
for key, workspace in workspaces.items():
    result = calculate_sharpe_detailed(workspace, key)
    if result:
        results[key] = result

print("\n=== SUMMARY ===")
print("\nThe issue: I was using per-trade Sharpe * sqrt(252), which is incorrect!")
print("- 5m has more frequent trades → more samples → lower per-trade volatility")
print("- But this doesn't mean lower risk when annualized properly")
print("\nCorrect Sharpe ratios (Method 2 - daily returns):")
for key, result in results.items():
    print(f"  {key}: {result['sharpe_daily']:.2f}")