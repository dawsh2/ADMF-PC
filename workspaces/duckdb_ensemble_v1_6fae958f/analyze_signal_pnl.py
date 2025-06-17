#!/usr/bin/env python3
"""
Analyze signal trace P&L using simple calculation method.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_simple_pnl(df):
    """
    Calculate P&L using simple method:
    - First non-zero signal opens position
    - When signal goes to 0: trade P&L = (exit_price - entry_price) * entry_signal_value
    - When signal flips (e.g. -1 to 1): close previous trade and open new one
    """
    if df.empty:
        return {
            'total_pnl': 0,
            'trades': [],
            'num_trades': 0,
            'win_rate': 0,
            'avg_trade_pnl': 0,
            'max_drawdown': 0
        }
    
    trades = []
    current_position = 0
    entry_price = None
    entry_bar_idx = None
    cumulative_pnl = 0
    pnl_curve = []
    
    print(f"Processing {len(df)} signal records...")
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{len(df)} records processed, {len(trades)} trades so far")
        bar_idx = row['bar_idx']
        signal = row.get('signal_value', 0)
        price = row['price']
        
        # Track cumulative P&L for drawdown calculation
        pnl_curve.append(cumulative_pnl)
        
        if current_position == 0:
            # No position, check if we should open one
            if signal != 0:
                current_position = signal
                entry_price = price
                entry_bar_idx = bar_idx
        else:
            # We have a position
            if signal == 0:
                # Close position
                trade_pnl = (price - entry_price) * current_position
                trades.append({
                    'entry_bar': entry_bar_idx,
                    'exit_bar': bar_idx,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'signal': current_position,
                    'pnl': trade_pnl,
                    'bars_held': bar_idx - entry_bar_idx
                })
                cumulative_pnl += trade_pnl
                
                current_position = 0
                entry_price = None
                entry_bar_idx = None
                
            elif signal != current_position:
                # Signal flip - close current and open new
                trade_pnl = (price - entry_price) * current_position
                trades.append({
                    'entry_bar': entry_bar_idx,
                    'exit_bar': bar_idx,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'signal': current_position,
                    'pnl': trade_pnl,
                    'bars_held': bar_idx - entry_bar_idx
                })
                cumulative_pnl += trade_pnl
                
                # Open new position
                current_position = signal
                entry_price = price
                entry_bar_idx = bar_idx
    
    # If we still have an open position at the end, we can't calculate its P&L
    if current_position != 0:
        print(f"Warning: Open position at end of data (signal={current_position}, entry={entry_price:.4f})")
    
    # Calculate performance metrics
    if not trades:
        return {
            'total_pnl': 0,
            'trades': [],
            'num_trades': 0,
            'win_rate': 0,
            'avg_trade_pnl': 0,
            'max_drawdown': 0
        }
    
    total_pnl = sum(trade['pnl'] for trade in trades)
    winning_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0
    avg_trade_pnl = total_pnl / len(trades) if trades else 0
    
    # Calculate maximum drawdown
    pnl_curve = np.array(pnl_curve)
    running_max = np.maximum.accumulate(pnl_curve)
    drawdown = pnl_curve - running_max
    max_drawdown = np.min(drawdown)
    
    return {
        'total_pnl': total_pnl,
        'trades': trades,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_trade_pnl': avg_trade_pnl,
        'max_drawdown': max_drawdown
    }

def analyze_periods(df):
    """Analyze different time periods."""
    if df.empty:
        return {}
    
    print(f"\nTotal bars in dataset: {df['bar_idx'].max() - df['bar_idx'].min() + 1}")
    print(f"Signal records: {len(df)}")
    print(f"Bar index range: {df['bar_idx'].min()} to {df['bar_idx'].max()}")
    
    # Full period
    full_results = calculate_simple_pnl(df)
    
    # Last 22k bars
    max_bar = df['bar_idx'].max()
    last_22k_df = df[df['bar_idx'] >= (max_bar - 22000)]
    last_22k_results = calculate_simple_pnl(last_22k_df)
    
    # Last 12k bars  
    last_12k_df = df[df['bar_idx'] >= (max_bar - 12000)]
    last_12k_results = calculate_simple_pnl(last_12k_df)
    
    return {
        'full_period': full_results,
        'last_22k_bars': last_22k_results,
        'last_12k_bars': last_12k_results
    }

def print_results(period_name, results):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"RESULTS FOR {period_name.upper()}")
    print(f"{'='*60}")
    print(f"Total P&L: ${results['total_pnl']:.4f}")
    print(f"Number of trades: {results['num_trades']}")
    print(f"Win rate: {results['win_rate']:.2%}")
    print(f"Average trade P&L: ${results['avg_trade_pnl']:.4f}")
    print(f"Maximum drawdown: ${results['max_drawdown']:.4f}")
    
    if results['trades']:
        pnls = [t['pnl'] for t in results['trades']]
        print(f"Best trade: ${max(pnls):.4f}")
        print(f"Worst trade: ${min(pnls):.4f}")
        
        # Trade duration stats
        durations = [t['bars_held'] for t in results['trades']]
        print(f"Average trade duration: {np.mean(durations):.1f} bars")
        print(f"Median trade duration: {np.median(durations):.1f} bars")

def main():
    # Path to the signal trace file
    signal_file = Path("/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_v1_6fae958f/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet")
    
    if not signal_file.exists():
        print(f"Signal file not found: {signal_file}")
        return
    
    print(f"Reading signal trace file: {signal_file}")
    df = pd.read_parquet(signal_file)
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nFirst few records:")
    print(df.head())
    
    # Check for signal column variations
    signal_col = None
    for col in ['direction', 'value', 'signal', 'val']:
        if col in df.columns:
            signal_col = col
            break
    
    if signal_col is None:
        print("Error: No signal column found (looking for 'direction', 'value', 'signal', or 'val')")
        return
    
    print(f"\nUsing signal column: '{signal_col}'")
    print(f"Signal value counts:")
    print(df[signal_col].value_counts().sort_index())
    
    # Map column names to expected format
    df = df.rename(columns={
        'idx': 'bar_idx',
        'px': 'price',
        signal_col: 'signal_value'
    })
    
    # Ensure data is sorted by bar_idx
    df = df.sort_values('bar_idx').reset_index(drop=True)
    
    # Analyze different periods
    results = analyze_periods(df)
    
    # Print results for each period
    for period, period_results in results.items():
        print_results(period, period_results)

if __name__ == "__main__":
    main()