#!/usr/bin/env python3
"""
Calculate performance using proper log returns with zero commission.

Usage:
    python simple_log_returns.py <parquet_file_path>
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def calculate_log_return_pnl(df):
    """
    Calculate P&L using log returns with zero commission:
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
    
    print(f"Processing {len(df)} signal records with ZERO COMMISSION...")
    
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

def calculate_time_period(df):
    """Calculate the actual time period covered by the data."""
    if df.empty:
        return {"days": 0, "start": None, "end": None}
    
    # Get first and last timestamps if available
    if 'ts' in df.columns:
        df['timestamp'] = pd.to_datetime(df['ts'])
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        days = (end_time - start_time).total_seconds() / (24 * 3600)
        return {
            "days": days,
            "start": start_time,
            "end": end_time
        }
    else:
        # Fallback: estimate from bar count (assuming 1-minute bars)
        total_bars = df['bar_idx'].max() - df['bar_idx'].min() + 1
        trading_days = total_bars / (390)  # 390 minutes per trading day
        return {
            "days": trading_days,
            "start": "Unknown",
            "end": "Unknown",
            "total_bars": total_bars
        }

def print_results(results, time_info):
    """Print formatted results with annualized returns."""
    print(f"\n{'='*70}")
    print(f"75% AGREEMENT THRESHOLD - LOG RETURN RESULTS (ZERO COMMISSION)")
    print(f"{'='*70}")
    
    # Time period info
    if isinstance(time_info['start'], str):
        print(f"Time period: {time_info.get('total_bars', 'Unknown')} bars")
        print(f"Estimated trading days: {time_info['days']:.1f}")
    else:
        print(f"Time period: {time_info['start']} to {time_info['end']}")
        print(f"Trading days: {time_info['days']:.1f}")
    
    print(f"Total log return: {results['total_log_return']:.6f}")
    print(f"Percentage return: {results['percentage_return']:.4%}")
    
    # Calculate annualized return
    if time_info['days'] > 0:
        annualized_return = ((1 + results['percentage_return']) ** (252 / time_info['days'])) - 1
        print(f"Annualized return: {annualized_return:.4%}")
    
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
        print(f"Average trade duration: {np.mean(durations):.1f} bars ({np.mean(durations):.1f} minutes)")
        print(f"Median trade duration: {np.median(durations):.1f} bars")

def main():
    if len(sys.argv) != 2:
        print("Usage: python simple_log_returns.py <parquet_file_path>")
        print("Example: python simple_log_returns.py workspaces/duckdb_ensemble_v1_56028885/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet")
        sys.exit(1)
    
    # Path to the signal trace file from command line
    signal_file = Path(sys.argv[1])
    
    if not signal_file.exists():
        print(f"Signal file not found: {signal_file}")
        sys.exit(1)
    
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
    
    # Calculate time period
    time_info = calculate_time_period(df)
    
    # Calculate performance
    results = calculate_log_return_pnl(df)
    
    # Print results
    print_results(results, time_info)

if __name__ == "__main__":
    main()