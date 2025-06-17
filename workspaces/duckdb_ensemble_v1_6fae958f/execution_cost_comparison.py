#!/usr/bin/env python3
"""
Compare execution cost impacts between aggressive (5bp + commission) and realistic (1bp) costs
for the DuckDB ensemble performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_trade_frequency_metrics(df):
    """Calculate trade frequency metrics from signal data."""
    current_position = 0
    trades = []
    entry_bar_idx = None
    
    for idx, row in df.iterrows():
        bar_idx = row['bar_idx']
        signal = row.get('signal_value', 0)
        
        if current_position == 0:
            # No position, check if we should open one
            if signal != 0:
                current_position = signal
                entry_bar_idx = bar_idx
        else:
            # We have a position
            if signal == 0 or signal != current_position:
                # Close position
                if entry_bar_idx is not None:
                    trade_duration = bar_idx - entry_bar_idx
                    trades.append({
                        'entry_bar': entry_bar_idx,
                        'exit_bar': bar_idx,
                        'duration_bars': trade_duration,
                        'signal': current_position
                    })
                
                # If signal flips, open new position
                if signal != 0 and signal != current_position:
                    current_position = signal
                    entry_bar_idx = bar_idx
                else:
                    current_position = 0
                    entry_bar_idx = None
    
    return trades

def calculate_simple_execution_impact(num_trades, cost_per_trade_pct):
    """Calculate simple execution cost impact."""
    total_cost_pct = num_trades * cost_per_trade_pct
    return total_cost_pct

def print_comparison_summary(df):
    """Print comparison between different execution cost scenarios."""
    
    # Calculate trade frequency
    trades = calculate_trade_frequency_metrics(df)
    num_trades = len(trades)
    total_bars = df['bar_idx'].max() - df['bar_idx'].min() + 1
    signal_records = len(df)
    
    # Trade frequency metrics
    avg_duration = np.mean([t['duration_bars'] for t in trades]) if trades else 0
    median_duration = np.median([t['duration_bars'] for t in trades]) if trades else 0
    trade_frequency = num_trades / total_bars if total_bars > 0 else 0
    
    print(f"{'='*80}")
    print("EXECUTION COST COMPARISON ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\n📊 TRADE FREQUENCY ANALYSIS:")
    print(f"Total Bars:                  {total_bars:,}")
    print(f"Signal Records:              {signal_records:,}")
    print(f"Number of Trades:            {num_trades:,}")
    print(f"Trade Frequency:             {trade_frequency:.4f} trades per bar")
    print(f"Trades per Day (assuming 390 bars/day): {trade_frequency * 390:.1f}")
    print(f"Average Trade Duration:      {avg_duration:.1f} bars")
    print(f"Median Trade Duration:       {median_duration:.1f} bars")
    
    # Cost scenarios
    realistic_cost_per_trade = 0.0001  # 1 bp
    aggressive_cost_per_trade = 0.0015  # ~15 bps (5bp slippage + ~10bp commission for typical price)
    
    print(f"\n💰 EXECUTION COST SCENARIOS:")
    print(f"{'Scenario':<20} {'Cost/Trade':<12} {'Total Cost':<12} {'Impact'}")
    print(f"{'-'*60}")
    
    # Realistic costs (1 bp)
    realistic_total_cost = calculate_simple_execution_impact(num_trades, realistic_cost_per_trade)
    print(f"{'Realistic (1bp)':<20} {realistic_cost_per_trade:<12.4%} {realistic_total_cost:<12.2%} {'Severe'}")
    
    # Aggressive costs (5bp + commission)
    aggressive_total_cost = calculate_simple_execution_impact(num_trades, aggressive_cost_per_trade)
    print(f"{'Aggressive (15bp)':<20} {aggressive_cost_per_trade:<12.4%} {aggressive_total_cost:<12.2%} {'Catastrophic'}")
    
    # Zero costs (for comparison)
    print(f"{'Zero Cost':<20} {0:<12.4%} {0:<12.2%} {'None'}")
    
    print(f"\n🔍 KEY INSIGHTS:")
    print(f"1. Strategy trades {trade_frequency:.4f} times per bar ({trade_frequency * 390:.1f} times per day)")
    print(f"2. With {num_trades:,} trades over {total_bars:,} bars, even 1bp costs add up")
    print(f"3. Realistic 1bp cost creates {realistic_total_cost:.2%} total drag")
    print(f"4. Aggressive costs would create {aggressive_total_cost:.2%} total drag")
    print(f"5. Average trade duration of {avg_duration:.1f} bars suggests high frequency strategy")
    
    # Break-even analysis
    print(f"\n📈 BREAK-EVEN ANALYSIS:")
    print(f"For strategy to be profitable after costs:")
    
    required_gross_return_realistic = realistic_total_cost
    required_gross_return_aggressive = aggressive_total_cost
    
    print(f"- With realistic costs: Need >{required_gross_return_realistic:.2%} gross return")
    print(f"- With aggressive costs: Need >{required_gross_return_aggressive:.2%} gross return")
    
    # Trade duration impact
    durations = [t['duration_bars'] for t in trades] if trades else []
    if durations:
        short_trades = len([d for d in durations if d <= 2])
        medium_trades = len([d for d in durations if 3 <= d <= 10])
        long_trades = len([d for d in durations if d > 10])
        
        print(f"\n⏱️ TRADE DURATION BREAKDOWN:")
        print(f"Short (≤2 bars):             {short_trades:,} ({short_trades/num_trades:.1%})")
        print(f"Medium (3-10 bars):          {medium_trades:,} ({medium_trades/num_trades:.1%})")
        print(f"Long (>10 bars):             {long_trades:,} ({long_trades/num_trades:.1%})")
        
        print(f"\n💡 EXECUTION COST RECOMMENDATIONS:")
        print(f"1. Consider increasing minimum trade duration to reduce frequency")
        print(f"2. Add position sizing to trade larger amounts less frequently")  
        print(f"3. Implement trade filtering to eliminate marginal signals")
        print(f"4. Consider batch execution or time-based exits")
        print(f"5. With {short_trades/num_trades:.1%} trades lasting ≤2 bars, consider signal smoothing")

def analyze_cost_sensitivity():
    """Analyze sensitivity to different cost levels."""
    signal_file = Path("traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet")
    
    if not signal_file.exists():
        print(f"Signal file not found: {signal_file}")
        return
    
    df = pd.read_parquet(signal_file)
    df = df.rename(columns={'idx': 'bar_idx', 'px': 'price', 'val': 'signal_value'})
    df = df.sort_values('bar_idx').reset_index(drop=True)
    
    # Calculate trade frequency and cost impact
    trades = calculate_trade_frequency_metrics(df)
    num_trades = len(trades)
    
    print(f"{'='*80}")
    print("EXECUTION COST SENSITIVITY ANALYSIS")
    print(f"{'='*80}")
    
    # Test different cost levels
    cost_levels = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]  # basis points
    
    print(f"\n📊 COST SENSITIVITY (Total Drag from {num_trades:,} trades):")
    print(f"{'Cost (bp)':<10} {'Cost %':<10} {'Total Drag':<12} {'Assessment'}")
    print(f"{'-'*50}")
    
    for cost_bp in cost_levels:
        cost_pct = cost_bp / 10000
        total_drag = num_trades * cost_pct
        
        if total_drag < 0.05:
            assessment = "Manageable"
        elif total_drag < 0.15:
            assessment = "Concerning"
        elif total_drag < 0.30:
            assessment = "Severe"
        else:
            assessment = "Prohibitive"
            
        print(f"{cost_bp:<10.1f} {cost_pct:<10.4%} {total_drag:<12.2%} {assessment}")
    
    # Current strategy assessment
    print(f"\n🎯 CURRENT STRATEGY ASSESSMENT:")
    realistic_drag = num_trades * 0.0001
    print(f"With realistic 1bp costs: {realistic_drag:.2%} total drag")
    print(f"This represents a severe execution cost burden due to high trade frequency")
    
    return df

def main():
    df = analyze_cost_sensitivity()
    if df is not None:
        print_comparison_summary(df)

if __name__ == "__main__":
    main()