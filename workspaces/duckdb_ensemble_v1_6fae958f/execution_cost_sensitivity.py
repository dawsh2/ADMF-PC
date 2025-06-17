#!/usr/bin/env python3
"""
Execution cost sensitivity analysis for DuckDB ensemble strategy.

Tests different execution cost scenarios from optimistic to aggressive.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_pnl_with_costs(df, commission_per_share, slippage_bps):
    """Calculate performance with specified execution costs."""
    
    if df.empty:
        return {'total_log_return': 0, 'percentage_return': 0, 'num_trades': 0, 'win_rate': 0}
    
    trades = []
    current_position = 0
    entry_price = None
    entry_bar_idx = None
    total_log_return = 0
    
    slippage_pct = slippage_bps / 10000  # Convert bps to decimal
    
    for idx, row in df.iterrows():
        bar_idx = row['bar_idx']
        signal = row.get('signal_value', 0)
        price = row['price']
        
        if current_position == 0:
            if signal != 0:
                current_position = signal
                entry_price = price
                entry_bar_idx = bar_idx
        else:
            if signal == 0 or signal != current_position:
                if entry_price > 0 and price > 0:
                    # Calculate gross trade return
                    gross_trade_log_return = np.log(price / entry_price) * current_position
                    
                    # Calculate execution costs
                    notional_value = 1000.0
                    shares = notional_value / entry_price
                    commission_total = shares * commission_per_share
                    commission_cost_pct = commission_total / notional_value
                    
                    # Total execution cost (both entry and exit)
                    total_execution_cost_pct = (commission_cost_pct + slippage_pct) * 2
                    
                    # Net trade return
                    net_trade_log_return = gross_trade_log_return - total_execution_cost_pct
                    
                    trades.append({
                        'gross_return': gross_trade_log_return,
                        'net_return': net_trade_log_return,
                        'execution_cost': total_execution_cost_pct,
                        'bars_held': bar_idx - entry_bar_idx
                    })
                    
                    total_log_return += net_trade_log_return
                
                if signal != 0 and signal != current_position:
                    current_position = signal
                    entry_price = price
                    entry_bar_idx = bar_idx
                else:
                    current_position = 0
                    entry_price = None
                    entry_bar_idx = None
    
    if not trades:
        return {'total_log_return': 0, 'percentage_return': 0, 'num_trades': 0, 'win_rate': 0}
    
    percentage_return = np.exp(total_log_return) - 1
    winning_trades = [t for t in trades if t['net_return'] > 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0
    
    return {
        'total_log_return': total_log_return,
        'percentage_return': percentage_return,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_execution_cost': np.mean([t['execution_cost'] for t in trades]),
        'trades': trades
    }

def run_sensitivity_analysis():
    """Run sensitivity analysis across different cost scenarios."""
    
    # Load data
    signal_file = Path("traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet")
    df = pd.read_parquet(signal_file)
    
    # Map column names
    df = df.rename(columns={'idx': 'bar_idx', 'px': 'price', 'val': 'signal_value'})
    df = df.sort_values('bar_idx').reset_index(drop=True)
    
    # Define cost scenarios
    scenarios = [
        {"name": "Optimistic", "commission": 0.000, "slippage": 0, "description": "No costs (theoretical)"},
        {"name": "Best Case", "commission": 0.001, "slippage": 1, "description": "$0.001/share, 1 bp slippage"},
        {"name": "Good Case", "commission": 0.002, "slippage": 2, "description": "$0.002/share, 2 bp slippage"},
        {"name": "Typical", "commission": 0.003, "slippage": 3, "description": "$0.003/share, 3 bp slippage"},
        {"name": "Realistic", "commission": 0.004, "slippage": 4, "description": "$0.004/share, 4 bp slippage"},
        {"name": "Aggressive", "commission": 0.005, "slippage": 5, "description": "$0.005/share, 5 bp slippage (calc.py)"},
        {"name": "Pessimistic", "commission": 0.007, "slippage": 7, "description": "$0.007/share, 7 bp slippage"},
        {"name": "Worst Case", "commission": 0.010, "slippage": 10, "description": "$0.010/share, 10 bp slippage"}
    ]
    
    print("Running execution cost sensitivity analysis...")
    print(f"Dataset: {len(df)} records")
    
    # Run analysis for each scenario
    results = []
    
    for scenario in scenarios:
        print(f"  Analyzing {scenario['name']} scenario...")
        result = calculate_pnl_with_costs(df, scenario['commission'], scenario['slippage'])
        result.update(scenario)
        results.append(result)
    
    # Print results
    print("\n" + "="*120)
    print("EXECUTION COST SENSITIVITY ANALYSIS - DUCKDB ENSEMBLE STRATEGY")
    print("="*120)
    
    print(f"\n{'Scenario':<12} {'Commission':<12} {'Slippage':<10} {'Net Return':<12} {'Win Rate':<10} {'Avg Cost':<10} {'Viability':<12}")
    print("-" * 120)
    
    for result in results:
        viability = "✅ Viable" if result['percentage_return'] > 0.02 else "⚠️  Marginal" if result['percentage_return'] > 0 else "❌ Not Viable"
        
        print(f"{result['name']:<12} ${result['commission']:.3f}/sh    {result['slippage']:>2} bps     "
              f"{result['percentage_return']:>10.2%} {result['win_rate']:>9.1%} {result['avg_execution_cost']:>9.3%} {viability}")
    
    # Detailed breakdown
    print(f"\n{'='*60}")
    print("DETAILED BREAKDOWN BY SCENARIO")
    print(f"{'='*60}")
    
    for result in results:
        print(f"\n🔸 {result['name']} ({result['description']}):")
        print(f"   Net Return: {result['percentage_return']:>8.2%}")
        print(f"   Win Rate:   {result['win_rate']:>8.1%}")
        print(f"   Avg Cost:   {result['avg_execution_cost']:>8.3%} per trade")
        print(f"   Total Cost: {result['avg_execution_cost'] * result['num_trades']:>8.1%} of capital")
        
        if result['percentage_return'] > 0:
            print(f"   Status:     ✅ Profitable")
        elif result['percentage_return'] > -0.1:
            print(f"   Status:     ⚠️  Small loss")
        else:
            print(f"   Status:     ❌ Major loss")
    
    # Break-even analysis
    print(f"\n{'='*60}")
    print("BREAK-EVEN ANALYSIS")
    print(f"{'='*60}")
    
    profitable_scenarios = [r for r in results if r['percentage_return'] > 0]
    
    if profitable_scenarios:
        best_profitable = max(profitable_scenarios, key=lambda x: x['percentage_return'])
        print(f"✅ Strategy remains profitable up to:")
        print(f"   Commission: ${best_profitable['commission']:.3f}/share")
        print(f"   Slippage:   {best_profitable['slippage']} basis points")
        print(f"   Combined:   ~{best_profitable['avg_execution_cost']:.3%} per trade")
    else:
        print(f"❌ Strategy is not profitable under any tested cost scenario")
        print(f"   Even the most optimistic assumptions result in losses")
        print(f"   Fundamental strategy redesign required")
    
    # Cost threshold analysis
    zero_cost_return = results[0]['percentage_return']  # No-cost scenario
    
    print(f"\n💡 COST IMPACT ANALYSIS:")
    print(f"   Theoretical (no cost) return: {zero_cost_return:.2%}")
    
    for result in results[1:]:  # Skip no-cost scenario
        cost_impact = zero_cost_return - result['percentage_return']
        cost_drag = cost_impact / zero_cost_return if zero_cost_return != 0 else 0
        
        if result['name'] == 'Aggressive':  # Highlight the calc.py scenario
            print(f"   📍 {result['name']} cost drag:     {cost_impact:.2%} ({cost_drag:.0%} of gross return) ⚠️")
        else:
            print(f"   {result['name']} cost drag:        {cost_impact:.2%} ({cost_drag:.0%} of gross return)")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"   1. Target execution costs < $0.003/share + 3 bps for viability")
    print(f"   2. Reduce trade frequency by 5-10x minimum")
    print(f"   3. Increase signal conviction thresholds")
    print(f"   4. Consider position sizing to amortize fixed costs")
    print(f"   5. Negotiate institutional-level execution terms")

if __name__ == "__main__":
    run_sensitivity_analysis()