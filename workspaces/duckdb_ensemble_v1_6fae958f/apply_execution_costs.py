#!/usr/bin/env python3
"""
Apply aggressive execution costs to DuckDB ensemble performance.

Uses aggressive calc.py defaults:
- Commission: $0.005/share
- Slippage: 5 basis points (0.0005)

For each trade:
- commission_cost_pct = $0.005 / trade_price (where trade_price is entry price)
- slippage_cost_pct = 0.0005 (5 basis points)
- total_execution_cost_pct = commission_cost_pct + slippage_cost_pct
- adjusted_trade_return = log(exit_price/entry_price) * signal_value - total_execution_cost_pct
"""

import pandas as pd
import numpy as np
from pathlib import Path
from decimal import Decimal

# Aggressive execution cost parameters from calc.py
COMMISSION_PER_SHARE = 0.005  # $0.005 per share
SLIPPAGE_BPS = 5  # 5 basis points
SLIPPAGE_PCT = SLIPPAGE_BPS / 10000  # Convert to decimal (0.0005)

def calculate_execution_costs(entry_price, exit_price, signal_value):
    """
    Calculate execution costs for a single trade.
    
    Args:
        entry_price: Price when entering the position
        exit_price: Price when exiting the position  
        signal_value: Signal strength (-1, 0, or 1)
        
    Returns:
        dict with cost components and total cost percentage
    """
    # Commission cost as percentage of trade value
    # Assuming $1000 notional trade size to calculate shares
    notional_value = 1000.0
    shares = notional_value / entry_price
    commission_total = shares * COMMISSION_PER_SHARE
    commission_cost_pct = commission_total / notional_value
    
    # Slippage cost (fixed 5 bps)
    slippage_cost_pct = SLIPPAGE_PCT
    
    # Total execution cost (applied on both entry and exit)
    total_execution_cost_pct = (commission_cost_pct + slippage_cost_pct) * 2  # Both entry and exit
    
    return {
        'commission_per_trade': commission_total,
        'commission_cost_pct': commission_cost_pct,
        'slippage_cost_pct': slippage_cost_pct,
        'total_execution_cost_pct': total_execution_cost_pct,
        'entry_price': entry_price,
        'shares_traded': shares
    }

def calculate_net_log_return_pnl(df):
    """
    Calculate P&L using log returns with execution costs applied.
    
    Returns both gross and net performance for comparison.
    """
    if df.empty:
        return {
            'gross': _empty_results(),
            'net': _empty_results(),
            'execution_costs': []
        }
    
    trades_gross = []
    trades_net = []
    execution_costs = []
    current_position = 0
    entry_price = None
    entry_bar_idx = None
    gross_total_log_return = 0
    net_total_log_return = 0
    gross_log_return_curve = []
    net_log_return_curve = []
    
    print(f"Processing {len(df)} signal records with execution costs...")
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{len(df)} records processed, {len(trades_gross)} trades so far")
        
        bar_idx = row['bar_idx']
        signal = row.get('signal_value', 0)
        price = row['price']
        
        # Track cumulative log returns for drawdown calculation
        gross_log_return_curve.append(gross_total_log_return)
        net_log_return_curve.append(net_total_log_return)
        
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
                if entry_price > 0 and price > 0:  # Avoid log(0) or log(negative)
                    # Calculate gross trade return
                    gross_trade_log_return = np.log(price / entry_price) * current_position
                    
                    # Calculate execution costs
                    exec_costs = calculate_execution_costs(entry_price, price, current_position)
                    execution_costs.append(exec_costs)
                    
                    # Calculate net trade return (subtract execution costs)
                    net_trade_log_return = gross_trade_log_return - exec_costs['total_execution_cost_pct']
                    
                    # Store gross trade
                    gross_trade = {
                        'entry_bar': entry_bar_idx,
                        'exit_bar': bar_idx,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'signal': current_position,
                        'log_return': gross_trade_log_return,
                        'bars_held': bar_idx - entry_bar_idx
                    }
                    trades_gross.append(gross_trade)
                    
                    # Store net trade
                    net_trade = gross_trade.copy()
                    net_trade['log_return'] = net_trade_log_return
                    net_trade['execution_cost_pct'] = exec_costs['total_execution_cost_pct']
                    net_trade['commission_cost_pct'] = exec_costs['commission_cost_pct'] * 2  # Entry + exit
                    net_trade['slippage_cost_pct'] = exec_costs['slippage_cost_pct'] * 2  # Entry + exit
                    trades_net.append(net_trade)
                    
                    gross_total_log_return += gross_trade_log_return
                    net_total_log_return += net_trade_log_return
                
                # If signal flips (not just goes to 0), open new position
                if signal != 0 and signal != current_position:
                    current_position = signal
                    entry_price = price
                    entry_bar_idx = bar_idx
                else:
                    current_position = 0
                    entry_price = None
                    entry_bar_idx = None
    
    # If we still have an open position at the end, we can't calculate its return
    if current_position != 0:
        print(f"Warning: Open position at end of data (signal={current_position}, entry={entry_price:.4f})")
    
    # Calculate performance metrics for both gross and net
    gross_results = _calculate_performance_metrics(trades_gross, gross_total_log_return, gross_log_return_curve)
    net_results = _calculate_performance_metrics(trades_net, net_total_log_return, net_log_return_curve)
    
    return {
        'gross': gross_results,
        'net': net_results,
        'execution_costs': execution_costs
    }

def _empty_results():
    """Return empty results structure."""
    return {
        'total_log_return': 0,
        'percentage_return': 0,
        'trades': [],
        'num_trades': 0,
        'win_rate': 0,
        'avg_trade_log_return': 0,
        'max_drawdown_pct': 0
    }

def _calculate_performance_metrics(trades, total_log_return, log_return_curve):
    """Calculate performance metrics from trades and log return curve."""
    if not trades:
        return _empty_results()
    
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
    max_drawdown_pct = np.min(drawdown) if len(drawdown) > 0 else 0
    
    return {
        'total_log_return': total_log_return,
        'percentage_return': percentage_return,
        'trades': trades,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_trade_log_return': avg_trade_log_return,
        'max_drawdown_pct': max_drawdown_pct
    }

def analyze_periods_with_costs(df):
    """Analyze different time periods with execution costs."""
    if df.empty:
        return {}
    
    print(f"\nTotal bars in dataset: {df['bar_idx'].max() - df['bar_idx'].min() + 1}")
    print(f"Signal records: {len(df)}")
    print(f"Bar index range: {df['bar_idx'].min()} to {df['bar_idx'].max()}")
    
    # Full period
    print("\n" + "="*50)
    print("CALCULATING FULL PERIOD WITH EXECUTION COSTS")
    print("="*50)
    full_results = calculate_net_log_return_pnl(df)
    
    # Last 12k bars  
    print("\n" + "="*50)
    print("CALCULATING LAST 12K BARS WITH EXECUTION COSTS")
    print("="*50)
    max_bar = df['bar_idx'].max()
    last_12k_df = df[df['bar_idx'] >= (max_bar - 12000)]
    print(f"Last 12k bars: {len(last_12k_df)} signal records from bar {max_bar - 12000} to {max_bar}")
    last_12k_results = calculate_net_log_return_pnl(last_12k_df)
    
    return {
        'full_period': full_results,
        'last_12k_bars': last_12k_results
    }

def print_comparison_results(period_name, results):
    """Print formatted comparison of gross vs net results."""
    gross = results['gross']
    net = results['net']
    
    print(f"\n{'='*80}")
    print(f"EXECUTION COST IMPACT ANALYSIS - {period_name.upper()}")
    print(f"{'='*80}")
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"{'Metric':<25} {'Gross':<15} {'Net':<15} {'Impact':<15}")
    print(f"{'-'*70}")
    
    # Total returns
    gross_pct = gross['percentage_return']
    net_pct = net['percentage_return']
    return_impact = gross_pct - net_pct
    print(f"{'Total Return':<25} {gross_pct:<15.4%} {net_pct:<15.4%} {-return_impact:<15.4%}")
    
    # Log returns
    gross_log = gross['total_log_return']
    net_log = net['total_log_return']
    log_impact = gross_log - net_log
    print(f"{'Total Log Return':<25} {gross_log:<15.6f} {net_log:<15.6f} {-log_impact:<15.6f}")
    
    # Average trade returns
    gross_avg = gross['avg_trade_log_return']
    net_avg = net['avg_trade_log_return']
    avg_impact = gross_avg - net_avg
    print(f"{'Avg Trade Log Return':<25} {gross_avg:<15.6f} {net_avg:<15.6f} {-avg_impact:<15.6f}")
    
    # Win rates
    print(f"{'Win Rate':<25} {gross['win_rate']:<15.2%} {net['win_rate']:<15.2%} {net['win_rate'] - gross['win_rate']:<15.2%}")
    
    # Max drawdown
    print(f"{'Max Drawdown':<25} {gross['max_drawdown_pct']:<15.4%} {net['max_drawdown_pct']:<15.4%} {net['max_drawdown_pct'] - gross['max_drawdown_pct']:<15.4%}")
    
    print(f"{'Number of Trades':<25} {gross['num_trades']:<15} {net['num_trades']:<15} {net['num_trades'] - gross['num_trades']:<15}")
    
    # Execution cost analysis
    if results['execution_costs']:
        exec_costs = results['execution_costs']
        avg_commission_pct = np.mean([ec['commission_cost_pct'] * 2 for ec in exec_costs])  # Entry + exit
        avg_slippage_pct = np.mean([ec['slippage_cost_pct'] * 2 for ec in exec_costs])  # Entry + exit
        avg_total_cost_pct = np.mean([ec['total_execution_cost_pct'] for ec in exec_costs])
        
        print(f"\n💰 EXECUTION COST BREAKDOWN:")
        print(f"Average Commission Cost:     {avg_commission_pct:.4%} per trade")
        print(f"Average Slippage Cost:       {avg_slippage_pct:.4%} per trade")
        print(f"Average Total Cost:          {avg_total_cost_pct:.4%} per trade")
        print(f"Total Cost Impact:           {log_impact:.6f} log return points")
        print(f"Total Cost Impact:           {return_impact:.4%} percentage points")
        
        # Cost as percentage of gross return
        if gross_pct != 0:
            cost_drag_pct = return_impact / gross_pct
            print(f"Cost Drag:                   {cost_drag_pct:.2%} of gross return")
    
    # Best and worst trades
    if gross['trades'] and net['trades']:
        gross_returns = [t['log_return'] for t in gross['trades']]
        net_returns = [t['log_return'] for t in net['trades']]
        
        print(f"\n🎯 TRADE ANALYSIS:")
        print(f"Best Gross Trade:            {max(gross_returns):.6f} ({np.exp(max(gross_returns))-1:.4%})")
        print(f"Best Net Trade:              {max(net_returns):.6f} ({np.exp(max(net_returns))-1:.4%})")
        print(f"Worst Gross Trade:           {min(gross_returns):.6f} ({np.exp(min(gross_returns))-1:.4%})")
        print(f"Worst Net Trade:             {min(net_returns):.6f} ({np.exp(min(net_returns))-1:.4%})")
        
        # Trade duration stats
        durations = [t['bars_held'] for t in gross['trades']]
        print(f"Average Trade Duration:      {np.mean(durations):.1f} bars")
        print(f"Median Trade Duration:       {np.median(durations):.1f} bars")

def print_execution_cost_summary():
    """Print summary of execution cost assumptions."""
    print(f"\n{'='*80}")
    print("AGGRESSIVE EXECUTION COST ASSUMPTIONS (from calc.py)")
    print(f"{'='*80}")
    print(f"Commission per Share:        ${COMMISSION_PER_SHARE:.3f}")
    print(f"Slippage:                    {SLIPPAGE_BPS} basis points ({SLIPPAGE_PCT:.4%})")
    print(f"Notional Trade Size:         $1,000 (for cost calculation)")
    print(f"Cost Application:            Applied on both entry and exit")
    print(f"")
    print(f"📝 Cost Calculation Method:")
    print(f"  1. Commission % = (shares × ${COMMISSION_PER_SHARE:.3f}) / $1,000")
    print(f"  2. Slippage % = {SLIPPAGE_PCT:.4%} (fixed)")
    print(f"  3. Total Cost % = (Commission % + Slippage %) × 2 (entry + exit)")
    print(f"  4. Net Return = Gross Log Return - Total Cost %")

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
    
    # Print execution cost assumptions
    print_execution_cost_summary()
    
    # Analyze different periods with execution costs
    results = analyze_periods_with_costs(df)
    
    # Print comparison results for each period
    for period, period_results in results.items():
        print_comparison_results(period, period_results)
    
    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY: IMPACT OF AGGRESSIVE EXECUTION COSTS")
    print(f"{'='*80}")
    
    for period, period_results in results.items():
        gross_return = period_results['gross']['percentage_return']
        net_return = period_results['net']['percentage_return']
        cost_impact = gross_return - net_return
        
        print(f"\n{period.replace('_', ' ').title()}:")
        print(f"  Gross Return: {gross_return:.4%}")
        print(f"  Net Return:   {net_return:.4%}")
        print(f"  Cost Impact:  -{cost_impact:.4%} ({cost_impact/gross_return:.1%} of gross return)")

if __name__ == "__main__":
    main()