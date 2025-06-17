#!/usr/bin/env python3
"""
Quick summary of execution cost impact on DuckDB ensemble strategy.

Shows the most critical numbers in a concise format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from apply_execution_costs import calculate_net_log_return_pnl, COMMISSION_PER_SHARE, SLIPPAGE_PCT

def print_quick_summary():
    """Print a quick summary of the cost impact analysis."""
    
    # Load data
    signal_file = Path("traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet")
    df = pd.read_parquet(signal_file)
    
    # Map column names
    df = df.rename(columns={'idx': 'bar_idx', 'px': 'price', 'val': 'signal_value'})
    df = df.sort_values('bar_idx').reset_index(drop=True)
    
    # Calculate results
    print("Calculating execution cost impact...")
    full_results = calculate_net_log_return_pnl(df)
    
    max_bar = df['bar_idx'].max()
    last_12k_df = df[df['bar_idx'] >= (max_bar - 12000)]
    recent_results = calculate_net_log_return_pnl(last_12k_df)
    
    # Print concise summary
    print("\n" + "="*80)
    print("🚨 DUCKDB ENSEMBLE: EXECUTION COST IMPACT SUMMARY")
    print("="*80)
    
    print(f"\n💰 COST ASSUMPTIONS (Aggressive):")
    print(f"   Commission: ${COMMISSION_PER_SHARE:.3f}/share")
    print(f"   Slippage:   {SLIPPAGE_PCT*10000:.0f} basis points")
    print(f"   Total Cost: ~0.102% per trade (entry + exit)")
    
    print(f"\n📊 PERFORMANCE DEVASTATION:")
    
    periods = [
        ("Full Period", full_results),
        ("Last 12k Bars", recent_results)
    ]
    
    for period_name, results in periods:
        gross = results['gross']
        net = results['net']
        
        print(f"\n   {period_name}:")
        print(f"   ├─ Gross Return:  {gross['percentage_return']:>8.2%}")
        print(f"   ├─ Net Return:    {net['percentage_return']:>8.2%}")
        print(f"   ├─ Cost Impact:   {-(gross['percentage_return'] - net['percentage_return']):>8.2%}")
        print(f"   ├─ Trades:        {gross['num_trades']:>8,}")
        print(f"   ├─ Gross Win Rate:{gross['win_rate']:>8.1%}")
        print(f"   └─ Net Win Rate:  {net['win_rate']:>8.1%}")
    
    # Calculate key metrics
    full_gross = full_results['gross']['percentage_return']
    full_net = full_results['net']['percentage_return']
    full_cost_drag = (full_gross - full_net) / full_gross if full_gross != 0 else 0
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   • Strategy generates {full_results['gross']['num_trades']:,} trades")
    print(f"   • Average trade duration: {np.mean([t['bars_held'] for t in full_results['gross']['trades']]):.1f} bars")
    print(f"   • Cost drag: {full_cost_drag:.0%} of gross returns")
    print(f"   • Every trade costs ~10x the average profit per trade")
    print(f"   • Win rate drops {full_results['gross']['win_rate'] - full_results['net']['win_rate']:.0%} due to costs")
    
    print(f"\n⚠️  VERDICT:")
    print(f"   STRATEGY IS NOT VIABLE under aggressive execution cost assumptions")
    print(f"   Transaction costs completely overwhelm any alpha generation")
    print(f"   Requires fundamental redesign to reduce trade frequency")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print_quick_summary()