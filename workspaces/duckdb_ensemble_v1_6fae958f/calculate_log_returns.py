#!/usr/bin/env python3
"""
Calculate performance using proper log returns for duckdb_ensemble_v1_6fae958f.

For each trade, calculate: t_i = log(price_exit / price_entry) * signal_value
Sum all t_i values to get total log return
Convert to percentage: percentage_return = exp(total_log_return) - 1
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_log_return_pnl(df):
    """
    Calculate P&L using log returns:
    - First non-zero signal opens position
    - When signal goes to 0: trade log return = log(exit_price / entry_price) * entry_signal_value
    - When signal flips (e.g. -1 to 1): close previous trade and open new one
    - Sum all log returns and convert to percentage at the end
    """
    if df.empty:
        return {
            'total_log_return': 0,
            'percentage_return': 0,
            'trades': [],
            'num_trades': 0,
            'win_rate': 0,
            'avg_trade_log_return': 0,
            'max_drawdown_pct': 0
        }
    
    trades = []
    current_position = 0
    entry_price = None
    entry_bar_idx = None
    total_log_return = 0
    log_return_curve = []
    
    print(f"Processing {len(df)} signal records...")
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{len(df)} records processed, {len(trades)} trades so far")
        
        bar_idx = row['bar_idx']
        signal = row.get('signal_value', 0)
        price = row['price']
        
        # Track cumulative log return for drawdown calculation
        log_return_curve.append(total_log_return)
        
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
                if entry_price > 0 and price > 0:  # Avoid log(0) or log(negative)
                    trade_log_return = np.log(price / entry_price) * current_position
                    trades.append({
                        'entry_bar': entry_bar_idx,
                        'exit_bar': bar_idx,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'signal': current_position,
                        'log_return': trade_log_return,
                        'bars_held': bar_idx - entry_bar_idx
                    })
                    total_log_return += trade_log_return
                
                current_position = 0
                entry_price = None
                entry_bar_idx = None
                
            elif signal != current_position:
                # Signal flip - close current and open new
                if entry_price > 0 and price > 0:  # Avoid log(0) or log(negative)
                    trade_log_return = np.log(price / entry_price) * current_position
                    trades.append({
                        'entry_bar': entry_bar_idx,
                        'exit_bar': bar_idx,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'signal': current_position,
                        'log_return': trade_log_return,
                        'bars_held': bar_idx - entry_bar_idx
                    })
                    total_log_return += trade_log_return
                
                # Open new position
                current_position = signal
                entry_price = price
                entry_bar_idx = bar_idx
    
    # If we still have an open position at the end, we can't calculate its return
    if current_position != 0:
        print(f"Warning: Open position at end of data (signal={current_position}, entry={entry_price:.4f})")
    
    # Calculate performance metrics
    if not trades:
        return {
            'total_log_return': 0,
            'percentage_return': 0,
            'trades': [],
            'num_trades': 0,
            'win_rate': 0,
            'avg_trade_log_return': 0,
            'max_drawdown_pct': 0
        }
    
    # Convert total log return to percentage return
    percentage_return = np.exp(total_log_return) - 1
    
    winning_trades = [t for t in trades if t['log_return'] > 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0
    avg_trade_log_return = total_log_return / len(trades) if trades else 0
    
    # Calculate maximum drawdown in percentage terms
    log_return_curve = np.array(log_return_curve)
    percentage_curve = np.exp(log_return_curve) - 1  # Convert to percentage returns
    running_max = np.maximum.accumulate(1 + percentage_curve)  # Running max of (1 + return)
    drawdown = (1 + percentage_curve) / running_max - 1  # Drawdown as fraction
    max_drawdown_pct = np.min(drawdown)
    
    return {
        'total_log_return': total_log_return,
        'percentage_return': percentage_return,
        'trades': trades,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_trade_log_return': avg_trade_log_return,
        'max_drawdown_pct': max_drawdown_pct
    }

def analyze_periods(df):
    """Analyze different time periods."""
    if df.empty:
        return {}
    
    print(f"\nTotal bars in dataset: {df['bar_idx'].max() - df['bar_idx'].min() + 1}")
    print(f"Signal records: {len(df)}")
    print(f"Bar index range: {df['bar_idx'].min()} to {df['bar_idx'].max()}")
    
    # Full period
    print("\n" + "="*50)
    print("CALCULATING FULL PERIOD")
    print("="*50)
    full_results = calculate_log_return_pnl(df)
    
    # Last 12k bars  
    print("\n" + "="*50)
    print("CALCULATING LAST 12K BARS")
    print("="*50)
    max_bar = df['bar_idx'].max()
    last_12k_df = df[df['bar_idx'] >= (max_bar - 12000)]
    print(f"Last 12k bars: {len(last_12k_df)} signal records from bar {max_bar - 12000} to {max_bar}")
    last_12k_results = calculate_log_return_pnl(last_12k_df)
    
    return {
        'full_period': full_results,
        'last_12k_bars': last_12k_results
    }

def print_results(period_name, results):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"LOG RETURN RESULTS FOR {period_name.upper()}")
    print(f"{'='*60}")
    print(f"Total log return: {results['total_log_return']:.6f}")
    print(f"Percentage return: {results['percentage_return']:.4%}")
    print(f"Number of trades: {results['num_trades']}")
    print(f"Win rate: {results['win_rate']:.2%}")
    print(f"Average trade log return: {results['avg_trade_log_return']:.6f}")
    print(f"Maximum drawdown: {results['max_drawdown_pct']:.4%}")
    
    if results['trades']:
        log_returns = [t['log_return'] for t in results['trades']]
        print(f"Best trade (log return): {max(log_returns):.6f}")
        print(f"Worst trade (log return): {min(log_returns):.6f}")
        
        # Convert to percentage returns for intuition
        pct_returns = [np.exp(lr) - 1 for lr in log_returns]
        print(f"Best trade (% return): {max(pct_returns):.4%}")
        print(f"Worst trade (% return): {min(pct_returns):.4%}")
        
        # Trade duration stats
        durations = [t['bars_held'] for t in results['trades']]
        print(f"Average trade duration: {np.mean(durations):.1f} bars")
        print(f"Median trade duration: {np.median(durations):.1f} bars")

def print_comparison(simple_pnl, log_return_pct):
    """Print comparison between simple P&L and log return percentage."""
    print(f"\n{'='*60}")
    print("COMPARISON: SIMPLE P&L vs LOG RETURN METHOD")
    print(f"{'='*60}")
    print(f"Simple P&L total: ${simple_pnl:.4f}")
    print(f"Log return percentage: {log_return_pct:.4%}")
    print(f"Difference: {abs(simple_pnl - log_return_pct):.6f}")
    print(f"Method: Log returns properly account for compounding")

def main():
    # Path to the signal trace file
    signal_file = Path("traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet")
    
    if not signal_file.exists():
        print(f"Signal file not found: {signal_file}")
        return
    
    print(f"Reading signal trace file: {signal_file}")
    df = pd.read_parquet(signal_file)
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    # Map column names to expected format
    df = df.rename(columns={
        'idx': 'bar_idx',
        'px': 'price',
        'val': 'signal_value'
    })
    
    # Ensure data is sorted by bar_idx
    df = df.sort_values('bar_idx').reset_index(drop=True)
    
    print(f"\nSignal value distribution:")
    print(df['signal_value'].value_counts().sort_index())
    
    print(f"\nPrice range: ${df['price'].min():.2f} to ${df['price'].max():.2f}")
    print(f"Price statistics:")
    print(df['price'].describe())
    
    # Analyze different periods
    results = analyze_periods(df)
    
    # Print results for each period
    for period, period_results in results.items():
        print_results(period, period_results)

if __name__ == "__main__":
    main()