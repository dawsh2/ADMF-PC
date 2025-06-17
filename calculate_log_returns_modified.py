#!/usr/bin/env python3
"""
Calculate performance using proper log returns with execution costs.

For each trade, calculate: t_i = log(price_exit / price_entry) * signal_value
Apply execution costs using src/execution/calc.py
Sum all t_i values to get total log return
Convert to percentage: percentage_return = exp(total_log_return) - 1

Usage:
    python calculate_log_returns_modified.py <parquet_file_path>
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from decimal import Decimal

# Import execution cost calculator
sys.path.append(str(Path(__file__).parent / 'src'))
from src.execution.calc import (
    calculate_commission, 
    calculate_slippage, 
    ensure_decimal,
    calculate_pnl,
    calculate_return_pct
)

# Execution cost settings
COMMISSION_RATE = Decimal('0.0')  # Zero commission as requested
SLIPPAGE_BPS = 0  # Zero slippage for pure strategy performance

def calculate_log_return_pnl_with_costs(df):
    """
    Calculate P&L using log returns with execution costs:
    - First non-zero signal opens position
    - When signal goes to 0: trade log return = log(exit_price / entry_price) * entry_signal_value
    - When signal flips (e.g. -1 to 1): close previous trade and open new one
    - Apply commission and slippage costs using execution/calc.py
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
            'max_drawdown_pct': 0,
            'total_commission': 0,
            'total_slippage': 0,
            'gross_log_return': 0,
            'net_log_return': 0
        }
    
    trades = []
    current_position = 0
    entry_price = None
    entry_bar_idx = None
    total_log_return = 0
    gross_log_return = 0
    total_commission = Decimal('0')
    total_slippage = Decimal('0')
    log_return_curve = []
    
    print(f"Processing {len(df)} signal records...")
    print(f"Commission rate: {COMMISSION_RATE}")
    print(f"Slippage: {SLIPPAGE_BPS} bps")
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{len(df)} records processed, {len(trades)} trades so far")
        
        bar_idx = row['bar_idx']
        signal = row.get('signal_value', 0)
        price = ensure_decimal(row['price'])
        
        # Track cumulative log return for drawdown calculation
        log_return_curve.append(total_log_return)
        
        if current_position == 0:
            # No position, check if we should open one
            if signal != 0:
                current_position = signal
                entry_price = price
                entry_bar_idx = bar_idx
                
                # Apply entry commission and slippage
                quantity = Decimal('1')  # Assume 1 share for simplicity
                entry_commission = calculate_commission(quantity, price, COMMISSION_RATE)
                entry_slippage = calculate_slippage(price, SLIPPAGE_BPS)
                
                total_commission += entry_commission
                total_slippage += entry_slippage
                
        else:
            # We have a position
            if signal == 0 or signal != current_position:
                # Close position (either to flat or flip)
                if entry_price > 0 and price > 0:  # Avoid log(0) or log(negative)
                    # Apply exit commission and slippage
                    quantity = Decimal('1')
                    exit_commission = calculate_commission(quantity, price, COMMISSION_RATE)
                    exit_slippage = calculate_slippage(price, SLIPPAGE_BPS)
                    
                    total_commission += exit_commission
                    total_slippage += exit_slippage
                    
                    # Calculate gross return (before costs)
                    gross_trade_log_return = float(np.log(float(price) / float(entry_price)) * current_position)
                    
                    # Calculate net return (after costs)
                    # For log returns, we convert to linear, apply costs, then back to log
                    gross_linear_return = np.exp(abs(gross_trade_log_return)) - 1
                    
                    # Apply costs as percentage of notional
                    cost_pct = float((exit_commission + exit_slippage + entry_commission + entry_slippage) / entry_price)
                    net_linear_return = gross_linear_return - cost_pct
                    
                    # Convert back to log return with proper sign
                    if net_linear_return > -1:  # Avoid log of negative numbers
                        net_trade_log_return = np.log(1 + net_linear_return) * np.sign(gross_trade_log_return)
                    else:
                        net_trade_log_return = -10  # Cap losses at very negative number
                    
                    trades.append({
                        'entry_bar': entry_bar_idx,
                        'exit_bar': bar_idx,
                        'entry_price': float(entry_price),
                        'exit_price': float(price),
                        'signal': current_position,
                        'gross_log_return': gross_trade_log_return,
                        'net_log_return': net_trade_log_return,
                        'bars_held': bar_idx - entry_bar_idx,
                        'commission': float(exit_commission + entry_commission),
                        'slippage': float(exit_slippage + entry_slippage)
                    })
                    
                    total_log_return += net_trade_log_return
                    gross_log_return += gross_trade_log_return
                
                # Reset position
                current_position = 0
                entry_price = None
                entry_bar_idx = None
                
                # If signal flip (not to zero), open new position
                if signal != 0:
                    current_position = signal
                    entry_price = price
                    entry_bar_idx = bar_idx
                    
                    # Apply entry costs for new position
                    quantity = Decimal('1')
                    entry_commission = calculate_commission(quantity, price, COMMISSION_RATE)
                    entry_slippage = calculate_slippage(price, SLIPPAGE_BPS)
                    
                    total_commission += entry_commission
                    total_slippage += entry_slippage
    
    # If we still have an open position at the end, we can't calculate its return
    if current_position != 0:
        print(f"Warning: Open position at end of data (signal={current_position}, entry={float(entry_price):.4f})")
    
    # Calculate performance metrics
    if not trades:
        return {
            'total_log_return': 0,
            'percentage_return': 0,
            'trades': [],
            'num_trades': 0,
            'win_rate': 0,
            'avg_trade_log_return': 0,
            'max_drawdown_pct': 0,
            'total_commission': 0,
            'total_slippage': 0,
            'gross_log_return': 0,
            'net_log_return': 0
        }
    
    # Convert total log return to percentage return
    percentage_return = np.exp(total_log_return) - 1
    gross_percentage_return = np.exp(gross_log_return) - 1
    
    winning_trades = [t for t in trades if t['net_log_return'] > 0]
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
        'gross_log_return': gross_log_return,
        'percentage_return': percentage_return,
        'gross_percentage_return': gross_percentage_return,
        'trades': trades,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_trade_log_return': avg_trade_log_return,
        'max_drawdown_pct': max_drawdown_pct,
        'total_commission': float(total_commission),
        'total_slippage': float(total_slippage)
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
    print("CALCULATING FULL PERIOD WITH EXECUTION COSTS")
    print("="*50)
    full_results = calculate_log_return_pnl_with_costs(df)
    
    # Last 12k bars  
    print("\n" + "="*50)
    print("CALCULATING LAST 12K BARS WITH EXECUTION COSTS")
    print("="*50)
    max_bar = df['bar_idx'].max()
    last_12k_df = df[df['bar_idx'] >= (max_bar - 12000)]
    print(f"Last 12k bars: {len(last_12k_df)} signal records from bar {max_bar - 12000} to {max_bar}")
    last_12k_results = calculate_log_return_pnl_with_costs(last_12k_df)
    
    return {
        'full_period': full_results,
        'last_12k_bars': last_12k_results
    }

def print_results(period_name, results):
    """Print formatted results."""
    print(f"\n{'='*70}")
    print(f"LOG RETURN RESULTS FOR {period_name.upper()} (WITH EXECUTION COSTS)")
    print(f"{'='*70}")
    print(f"Gross log return: {results['gross_log_return']:.6f}")
    print(f"Net log return: {results['total_log_return']:.6f}")
    print(f"Gross percentage return: {results['gross_percentage_return']:.4%}")
    print(f"Net percentage return: {results['percentage_return']:.4%}")
    print(f"Number of trades: {results['num_trades']}")
    print(f"Win rate: {results['win_rate']:.2%}")
    print(f"Average trade log return: {results['avg_trade_log_return']:.6f}")
    print(f"Maximum drawdown: {results['max_drawdown_pct']:.4%}")
    print(f"Total commission: ${results['total_commission']:.6f}")
    print(f"Total slippage: ${results['total_slippage']:.6f}")
    print(f"Total execution costs: ${results['total_commission'] + results['total_slippage']:.6f}")
    
    if results['trades']:
        net_log_returns = [t['net_log_return'] for t in results['trades']]
        gross_log_returns = [t['gross_log_return'] for t in results['trades']]
        
        print(f"Best trade (gross log return): {max(gross_log_returns):.6f}")
        print(f"Worst trade (gross log return): {min(gross_log_returns):.6f}")
        print(f"Best trade (net log return): {max(net_log_returns):.6f}")
        print(f"Worst trade (net log return): {min(net_log_returns):.6f}")
        
        # Convert to percentage returns for intuition
        gross_pct_returns = [np.exp(lr) - 1 for lr in gross_log_returns]
        net_pct_returns = [np.exp(lr) - 1 for lr in net_log_returns]
        
        print(f"Best trade (gross % return): {max(gross_pct_returns):.4%}")
        print(f"Worst trade (gross % return): {min(gross_pct_returns):.4%}")
        print(f"Best trade (net % return): {max(net_pct_returns):.4%}")
        print(f"Worst trade (net % return): {min(net_pct_returns):.4%}")
        
        # Trade duration stats
        durations = [t['bars_held'] for t in results['trades']]
        print(f"Average trade duration: {np.mean(durations):.1f} bars")
        print(f"Median trade duration: {np.median(durations):.1f} bars")

def main():
    if len(sys.argv) != 2:
        print("Usage: python calculate_log_returns_modified.py <parquet_file_path>")
        print("Example: python calculate_log_returns_modified.py workspaces/duckdb_ensemble_v1_56028885/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet")
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
    print(f"Price statistics:")
    print(df['price'].describe())
    
    # Analyze different periods
    results = analyze_periods(df)
    
    # Print results for each period
    for period, period_results in results.items():
        print_results(period, period_results)
    
    # Print execution cost impact summary
    print(f"\n{'='*70}")
    print("EXECUTION COST IMPACT SUMMARY")
    print(f"{'='*70}")
    full_results = results['full_period']
    cost_impact = full_results['gross_percentage_return'] - full_results['percentage_return']
    print(f"Gross return: {full_results['gross_percentage_return']:.4%}")
    print(f"Net return: {full_results['percentage_return']:.4%}")
    print(f"Cost impact: {cost_impact:.4%}")
    print(f"Cost impact (bps): {cost_impact * 10000:.1f} bps")

if __name__ == "__main__":
    main()