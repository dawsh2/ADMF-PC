#!/usr/bin/env python3
"""Analyze Keltner Bands multiplier sweep results."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from src.analytics.sparse_trace_analysis.strategy_analysis import load_strategy_signals
from src.analytics.sparse_trace_analysis.performance_calculation import (
    calculate_log_returns_with_costs,
    ExecutionCostConfig
)

def analyze_multiplier_sweep(workspace_path: str):
    """Analyze results from multiplier sweep optimization."""
    
    workspace = Path(workspace_path)
    if not workspace.exists():
        print(f"Workspace not found: {workspace_path}")
        return
    
    print(f"Analyzing workspace: {workspace}")
    
    # Load metadata to get strategy configurations
    metadata_path = workspace / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            print(f"Found {len(metadata.get('strategies', []))} strategies in metadata")
    
    # Set up execution costs (2bp round trip)
    cost_config = ExecutionCostConfig(cost_multiplier=0.9998)
    
    # Collect results
    results = []
    
    # Check traces directory
    traces_dir = workspace / "traces"
    if not traces_dir.exists():
        print(f"No traces directory found in {workspace}")
        return
    
    # Load the multi-strategy trace file
    # Check for symbol/timeframe subdirectories
    trace_files = list(traces_dir.glob("**/*.pkl"))
    print(f"Found {len(trace_files)} trace files")
    
    if not trace_files:
        print("No trace files found!")
        return
    
    # Load the trace
    for trace_file in trace_files:
        print(f"\nAnalyzing trace file: {trace_file.name}")
        
        signals_df = load_strategy_signals(trace_file)
        if signals_df is None or signals_df.empty:
            print("No signals found in trace file")
            continue
            
        print(f"Loaded signals data with shape: {signals_df.shape}")
        print(f"Date range: {signals_df.index[0]} to {signals_df.index[-1]}")
        
        # Get unique strategies
        if 'strategy_id' in signals_df.columns:
            strategies = signals_df['strategy_id'].unique()
            print(f"Found {len(strategies)} unique strategies")
            
            for strategy_id in sorted(strategies):
                # Filter signals for this strategy
                strategy_signals = signals_df[signals_df['strategy_id'] == strategy_id].copy()
                
                if strategy_signals.empty:
                    continue
                
                # Calculate performance
                try:
                    performance = calculate_log_returns_with_costs(
                        signals_df=strategy_signals,
                        cost_config=cost_config,
                        initial_capital=10000.0
                    )
                    
                    # Get metadata for this strategy
                    strategy_meta = {}
                    if 'strategies' in metadata:
                        for s in metadata['strategies']:
                            if s.get('id') == strategy_id or s.get('name') == f'compiled_strategy_{strategy_id}':
                                strategy_meta = s.get('params', {})
                                break
                    
                    # Calculate metrics
                    total_signals = len(strategy_signals[strategy_signals['signal'] != 0])
                    if total_signals > 0:
                        trading_days = (strategy_signals.index[-1] - strategy_signals.index[0]).days or 1
                        trades_per_day = total_signals / trading_days * 252 / 365  # Adjust for trading days
                        
                        # Extract wins and losses
                        returns = performance.get('trade_returns', [])
                        if returns:
                            wins = [r for r in returns if r > 0]
                            losses = [r for r in returns if r <= 0]
                            win_rate = len(wins) / len(returns) * 100 if returns else 0
                            avg_win = np.mean(wins) * 10000 if wins else 0  # Convert to bps
                            avg_loss = np.mean(losses) * 10000 if losses else 0
                            
                            # Edge per trade
                            edge_bps = performance.get('total_return', 0) / total_signals * 10000
                            
                            result = {
                                'strategy': strategy_id,
                                'period': strategy_meta.get('period', 'unknown'),
                                'multiplier': strategy_meta.get('multiplier', 'unknown'),
                                'edge_bps': edge_bps,
                                'trades_per_day': trades_per_day,
                                'total_trades': total_signals,
                                'win_rate': win_rate,
                                'avg_win_bps': avg_win,
                                'avg_loss_bps': avg_loss,
                                'total_return_bps': performance.get('total_return', 0) * 10000,
                                'annual_trades': trades_per_day * 252
                            }
                            
                            # Calculate expected annual return
                            result['expected_annual_return'] = (
                                result['edge_bps'] * result['annual_trades'] / 10000
                            )
                            
                            results.append(result)
                            
                except Exception as e:
                    print(f"Error analyzing strategy {strategy_id}: {e}")
                    continue
        else:
            print("No strategy_id column found in signals")
    
    if not results:
        print("\nNo results to analyze!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by multiplier for each period
    df = df.sort_values(['period', 'multiplier'])
    
    print("\n=== MULTIPLIER SWEEP ANALYSIS ===\n")
    print(f"Total strategies analyzed: {len(df)}")
    
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
                print(f"  Multiplier {row['multiplier']}: "
                      f"{row['edge_bps']:.2f} bps, "
                      f"{row['trades_per_day']:.1f} trades/day, "
                      f"{row['annual_trades']:.0f} trades/year")
        
        # Find optimal trade-off point
        viable = period_df[
            (period_df['annual_trades'] > 100) & 
            (period_df['edge_bps'] > 0)
        ]
        
        if not viable.empty:
            # Sort by expected annual return
            viable = viable.sort_values('expected_annual_return', ascending=False)
            best = viable.iloc[0]
            
            print(f"\nBest trade-off (>100 trades/year):")
            print(f"  Multiplier: {best['multiplier']}")
            print(f"  Edge: {best['edge_bps']:.2f} bps")
            print(f"  Trades/day: {best['trades_per_day']:.1f}")
            print(f"  Annual trades: {best['annual_trades']:.0f}")
            print(f"  Expected annual return: {best['expected_annual_return']:.2%}")
            print(f"  Win rate: {best['win_rate']:.1f}%")
    
    # Overall analysis
    print("\n\n=== OVERALL FINDINGS ===\n")
    
    # Show top 10 by expected return
    df_sorted = df.sort_values('expected_annual_return', ascending=False)
    print("Top 10 strategies by expected annual return:")
    print("Period | Mult | Edge(bps) | Trades/yr | E[Return] | Win%")
    print("-------|------|-----------|-----------|-----------|-----")
    for _, row in df_sorted.head(10).iterrows():
        print(f"{row['period']:6} | {row['multiplier']:4} | {row['edge_bps']:9.2f} | "
              f"{row['annual_trades']:9.0f} | {row['expected_annual_return']:9.2%} | "
              f"{row['win_rate']:4.1f}%")
    
    # Find the efficient frontier
    print("\n\n=== EFFICIENT FRONTIER ===")
    print("(Best return for given trade frequency)\n")
    
    # Bin by trade frequency
    bins = [0, 50, 100, 200, 500, 1000, 5000, 10000]
    labels = ['<50', '50-100', '100-200', '200-500', '500-1k', '1k-5k', '5k+']
    df['freq_bin'] = pd.cut(df['annual_trades'], bins=bins, labels=labels)
    
    for freq_bin in labels:
        bin_data = df[df['freq_bin'] == freq_bin]
        if not bin_data.empty:
            best = bin_data.loc[bin_data['expected_annual_return'].idxmax()]
            print(f"\n{freq_bin} trades/year:")
            print(f"  Best: Period={best['period']}, Mult={best['multiplier']}")
            print(f"  Edge: {best['edge_bps']:.2f} bps")
            print(f"  Expected return: {best['expected_annual_return']:.2%}")
            print(f"  Win rate: {best['win_rate']:.1f}%")
    
    # Save detailed results
    output_file = "keltner_multiplier_sweep_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    import sys
    workspace = sys.argv[1] if len(sys.argv) > 1 else "workspaces/signal_generation_3f2b1535"
    analyze_multiplier_sweep(workspace)