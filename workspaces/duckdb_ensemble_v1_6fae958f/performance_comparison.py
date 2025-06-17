#!/usr/bin/env python3
"""
Comprehensive performance comparison: Simple P&L vs Log Returns
for duckdb_ensemble_v1_6fae958f workspace.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_both_methods(df):
    """Calculate performance using both simple P&L and log returns."""
    if df.empty:
        return {
            'simple': {'total_pnl': 0, 'trades': [], 'num_trades': 0},
            'log_return': {'total_log_return': 0, 'percentage_return': 0, 'trades': [], 'num_trades': 0}
        }
    
    # Shared variables
    trades_simple = []
    trades_log = []
    current_position = 0
    entry_price = None
    entry_bar_idx = None
    
    # Simple P&L tracking
    cumulative_pnl = 0
    
    # Log return tracking
    total_log_return = 0
    
    for idx, row in df.iterrows():
        bar_idx = row['bar_idx']
        signal = row.get('signal_value', 0)
        price = row['price']
        
        if current_position == 0:
            # No position, check if we should open one
            if signal != 0:
                current_position = signal
                entry_price = price
                entry_bar_idx = bar_idx
        else:
            # We have a position
            if signal == 0 or signal != current_position:
                # Close position (either signal goes to 0 or flips)
                
                # Simple P&L calculation
                trade_pnl = (price - entry_price) * current_position
                trades_simple.append({
                    'entry_bar': entry_bar_idx,
                    'exit_bar': bar_idx,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'signal': current_position,
                    'pnl': trade_pnl,
                    'bars_held': bar_idx - entry_bar_idx
                })
                cumulative_pnl += trade_pnl
                
                # Log return calculation
                if entry_price > 0 and price > 0:
                    trade_log_return = np.log(price / entry_price) * current_position
                    trades_log.append({
                        'entry_bar': entry_bar_idx,
                        'exit_bar': bar_idx,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'signal': current_position,
                        'log_return': trade_log_return,
                        'bars_held': bar_idx - entry_bar_idx
                    })
                    total_log_return += trade_log_return
                
                # If signal flips, open new position
                if signal != 0 and signal != current_position:
                    current_position = signal
                    entry_price = price
                    entry_bar_idx = bar_idx
                else:
                    current_position = 0
                    entry_price = None
                    entry_bar_idx = None
    
    # Calculate final metrics
    simple_results = {
        'total_pnl': cumulative_pnl,
        'trades': trades_simple,
        'num_trades': len(trades_simple),
        'win_rate': len([t for t in trades_simple if t['pnl'] > 0]) / len(trades_simple) if trades_simple else 0,
        'avg_trade_pnl': cumulative_pnl / len(trades_simple) if trades_simple else 0
    }
    
    log_results = {
        'total_log_return': total_log_return,
        'percentage_return': np.exp(total_log_return) - 1,
        'trades': trades_log,
        'num_trades': len(trades_log),
        'win_rate': len([t for t in trades_log if t['log_return'] > 0]) / len(trades_log) if trades_log else 0,
        'avg_trade_log_return': total_log_return / len(trades_log) if trades_log else 0
    }
    
    return {
        'simple': simple_results,
        'log_return': log_results
    }

def print_comparison_table(period_name, results):
    """Print a comparison table for both methods."""
    simple = results['simple']
    log_ret = results['log_return']
    
    print(f"\n{'='*80}")
    print(f"PERFORMANCE COMPARISON - {period_name.upper()}")
    print(f"{'='*80}")
    
    print(f"{'Metric':<30} {'Simple P&L':<20} {'Log Returns':<20} {'Notes'}")
    print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*20}")
    
    print(f"{'Total Return':<30} ${simple['total_pnl']:<19.4f} {log_ret['percentage_return']:<19.4%} {'% vs $'}")
    print(f"{'Number of Trades':<30} {simple['num_trades']:<20} {log_ret['num_trades']:<20} {'Should match'}")
    print(f"{'Win Rate':<30} {simple['win_rate']:<19.2%} {log_ret['win_rate']:<19.2%} {'Should match'}")
    print(f"{'Avg Trade Return':<30} ${simple['avg_trade_pnl']:<19.6f} {log_ret['avg_trade_log_return']:<19.6f} {'$ vs log'}")
    
    # Calculate equivalent percentage for simple P&L (approximate)
    if simple['trades']:
        avg_price = np.mean([t['entry_price'] for t in simple['trades']])
        simple_pct_approx = simple['total_pnl'] / avg_price
        print(f"{'Simple P&L as % (approx)':<30} {simple_pct_approx:<19.4%} {'-':<20} {'Rough estimate'}")
    
    print(f"\n{'Key Differences:'}")
    print(f"• Simple P&L: ${simple['total_pnl']:.4f} (arithmetic sum of dollar gains/losses)")
    print(f"• Log Returns: {log_ret['percentage_return']:.4%} (proper compounding)")
    print(f"• Log returns account for the multiplicative nature of returns")
    print(f"• For small returns, the difference is minimal")
    
    if simple['trades'] and log_ret['trades']:
        # Show a few example trades
        print(f"\nExample Trade Comparisons (first 5 trades):")
        print(f"{'Trade':<6} {'Entry':<8} {'Exit':<8} {'Signal':<7} {'Simple P&L':<12} {'Log Return':<12} {'% Return'}")
        for i in range(min(5, len(simple['trades']))):
            s_trade = simple['trades'][i]
            l_trade = log_ret['trades'][i]
            pct_return = np.exp(l_trade['log_return']) - 1
            print(f"{i+1:<6} {s_trade['entry_price']:<8.2f} {s_trade['exit_price']:<8.2f} {s_trade['signal']:<7} "
                  f"${s_trade['pnl']:<11.4f} {l_trade['log_return']:<11.6f} {pct_return:<11.4%}")

def main():
    # Path to the signal trace file
    signal_file = Path("traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet")
    
    if not signal_file.exists():
        print(f"Signal file not found: {signal_file}")
        return
    
    print("="*80)
    print("DUCKDB ENSEMBLE V1 PERFORMANCE ANALYSIS")
    print("Comparing Simple P&L vs Log Returns")
    print("="*80)
    
    df = pd.read_parquet(signal_file)
    
    # Map column names
    df = df.rename(columns={
        'idx': 'bar_idx',
        'px': 'price',
        'val': 'signal_value'
    })
    
    # Ensure data is sorted by bar_idx
    df = df.sort_values('bar_idx').reset_index(drop=True)
    
    print(f"Dataset: {len(df)} signal records")
    print(f"Bar range: {df['bar_idx'].min()} to {df['bar_idx'].max()}")
    print(f"Price range: ${df['price'].min():.2f} to ${df['price'].max():.2f}")
    print(f"Signal distribution: {dict(df['signal_value'].value_counts().sort_index())}")
    
    # Full period analysis
    print(f"\nCalculating full period performance...")
    full_results = calculate_both_methods(df)
    print_comparison_table("Full Period", full_results)
    
    # Last 12k bars analysis
    print(f"\nCalculating last 12k bars performance...")
    max_bar = df['bar_idx'].max()
    last_12k_df = df[df['bar_idx'] >= (max_bar - 12000)]
    print(f"Last 12k bars: {len(last_12k_df)} signal records")
    last_12k_results = calculate_both_methods(last_12k_df)
    print_comparison_table("Last 12k Bars", last_12k_results)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("Log returns provide the mathematically correct way to measure performance")
    print("when returns compound over time. Key findings:")
    print()
    print(f"Full Period:")
    print(f"  • Simple P&L: ${full_results['simple']['total_pnl']:.4f}")
    print(f"  • Log Returns: {full_results['log_return']['percentage_return']:.4%}")
    print()
    print(f"Last 12k Bars:")
    print(f"  • Simple P&L: ${last_12k_results['simple']['total_pnl']:.4f}")
    print(f"  • Log Returns: {last_12k_results['log_return']['percentage_return']:.4%}")
    print()
    print("The log return method properly accounts for compounding and gives")
    print("the true percentage return that would be achieved by the strategy.")

if __name__ == "__main__":
    main()