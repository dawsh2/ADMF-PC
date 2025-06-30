#!/usr/bin/env python3
"""Analyze multiplier sweep results to find optimal trade-off between frequency and edge."""

import pandas as pd
import numpy as np
from pathlib import Path
from src.analytics.sparse_trace_analysis.strategy_analysis import analyze_strategy_signals
from src.analytics.sparse_trace_analysis.performance_calculation import (
    calculate_log_returns_with_costs,
    ExecutionCostConfig
)
from src.analytics.workspace import WorkspaceManager

def analyze_multiplier_sweep(workspace_path: str):
    """Analyze results from multiplier sweep optimization."""
    
    # Load workspace
    workspace_manager = WorkspaceManager()
    
    # Use specified workspace
    target_workspace = Path(workspace_path)
    if not target_workspace.exists():
        print(f"Workspace not found: {workspace_path}")
        return
    
    print(f"Analyzing workspace: {target_workspace}")
    
    # Set up execution costs (2bp round trip)
    cost_config = ExecutionCostConfig(cost_multiplier=0.9998)
    
    # Collect results
    results = []
    
    # Load all strategy traces
    traces_dir = target_workspace / "traces"
    if not traces_dir.exists():
        print(f"No traces directory found in {target_workspace}")
        return
        
    for trace_file in sorted(traces_dir.glob("strategy_*.pkl")):
        strategy_num = int(trace_file.stem.split("_")[1])
        
        try:
            # Analyze signals
            summary = analyze_strategy_signals(
                str(target_workspace),
                f"strategy_{strategy_num}",
                cost_config
            )
            
            if summary and summary['total_signals'] > 0:
                # Extract parameters from metadata
                metadata = summary.get('metadata', {})
                params = metadata.get('parameters', {})
                
                result = {
                    'strategy': strategy_num,
                    'period': params.get('period', 'unknown'),
                    'multiplier': params.get('multiplier', 'unknown'),
                    'edge_bps': summary['edge_per_trade_bps'],
                    'trades_per_day': summary['trades_per_day'],
                    'total_trades': summary['total_signals'],
                    'win_rate': summary['win_rate'],
                    'avg_win_bps': summary['avg_win_bps'],
                    'avg_loss_bps': summary['avg_loss_bps'],
                    'sharpe': summary.get('sharpe_ratio', 0),
                    'annual_trades': summary['trades_per_day'] * 252
                }
                
                # Calculate expected annual return
                result['expected_annual_return'] = (
                    result['edge_bps'] * result['annual_trades'] / 10000
                )
                
                results.append(result)
                
        except Exception as e:
            print(f"Error analyzing strategy {strategy_num}: {e}")
            continue
    
    if not results:
        print("No results to analyze!")
        return
        
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by multiplier for each period
    df = df.sort_values(['period', 'multiplier'])
    
    print("\n=== MULTIPLIER SWEEP ANALYSIS ===\n")
    
    # Analyze by period
    for period in df['period'].unique():
        period_df = df[df['period'] == period].copy()
        
        print(f"\n--- Period {period} ---")
        print(f"Multipliers tested: {len(period_df)}")
        
        # Find strategies meeting criteria (>1 bps edge)
        good_strategies = period_df[period_df['edge_bps'] > 1.0]
        
        if not good_strategies.empty:
            print(f"\nStrategies with >1 bps edge:")
            for _, row in good_strategies.iterrows():
                print(f"  Multiplier {row['multiplier']:.2f}: "
                      f"{row['edge_bps']:.2f} bps, "
                      f"{row['trades_per_day']:.1f} trades/day, "
                      f"{row['annual_trades']:.0f} trades/year")
        
        # Find optimal trade-off point
        # Look for strategies with reasonable frequency (>100 trades/year) and positive edge
        viable = period_df[
            (period_df['annual_trades'] > 100) & 
            (period_df['edge_bps'] > 0)
        ]
        
        if not viable.empty:
            # Sort by expected annual return
            viable = viable.sort_values('expected_annual_return', ascending=False)
            best = viable.iloc[0]
            
            print(f"\nBest trade-off (>100 trades/year):")
            print(f"  Multiplier: {best['multiplier']:.2f}")
            print(f"  Edge: {best['edge_bps']:.2f} bps")
            print(f"  Trades/day: {best['trades_per_day']:.1f}")
            print(f"  Annual trades: {best['annual_trades']:.0f}")
            print(f"  Expected annual return: {best['expected_annual_return']:.2%}")
            print(f"  Win rate: {best['win_rate']:.1%}")
    
    # Overall analysis
    print("\n\n=== OVERALL FINDINGS ===\n")
    
    # Group by multiplier across all periods
    mult_summary = df.groupby('multiplier').agg({
        'edge_bps': 'mean',
        'trades_per_day': 'mean',
        'win_rate': 'mean',
        'expected_annual_return': 'mean'
    }).round(2)
    
    print("Average performance by multiplier (across all periods):")
    print(mult_summary.head(20))
    
    # Find the efficient frontier
    print("\n\n=== EFFICIENT FRONTIER ===")
    print("(Strategies with best return for given trade frequency)\n")
    
    # Bin by trade frequency
    df['freq_bin'] = pd.cut(df['annual_trades'], 
                            bins=[0, 50, 100, 200, 500, 1000, 5000, 10000],
                            labels=['<50', '50-100', '100-200', '200-500', 
                                   '500-1k', '1k-5k', '5k+'])
    
    for freq_bin in df['freq_bin'].cat.categories:
        bin_data = df[df['freq_bin'] == freq_bin]
        if not bin_data.empty:
            best = bin_data.loc[bin_data['expected_annual_return'].idxmax()]
            print(f"\n{freq_bin} trades/year:")
            print(f"  Best: Period={best['period']}, Mult={best['multiplier']:.2f}")
            print(f"  Edge: {best['edge_bps']:.2f} bps")
            print(f"  Expected return: {best['expected_annual_return']:.2%}")
            print(f"  Win rate: {best['win_rate']:.1%}")
    
    # Save detailed results
    output_file = "multiplier_sweep_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n\nDetailed results saved to: {output_file}")
    
    # Create text-based visualization
    print("\n\n=== VISUALIZATION: Edge vs Multiplier ===")
    print("(Using ASCII art since matplotlib not available)\n")
    
    # Show edge degradation as multiplier increases
    for period in sorted(df['period'].unique()):
        period_data = df[df['period'] == period].sort_values('multiplier')
        print(f"\nPeriod {period}:")
        print("Mult  | Edge (bps) | Trades/day | Visual")
        print("------|------------|------------|" + "-" * 30)
        
        for _, row in period_data.iterrows():
            edge_bar = '*' * min(int(row['edge_bps'] * 5), 30)
            print(f"{row['multiplier']:5.2f} | {row['edge_bps']:10.2f} | {row['trades_per_day']:10.1f} | {edge_bar}")

if __name__ == "__main__":
    import sys
    workspace = sys.argv[1] if len(sys.argv) > 1 else "workspaces/signal_generation_3f2b1535"
    analyze_multiplier_sweep(workspace)