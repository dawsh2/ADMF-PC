#!/usr/bin/env python3
"""
Analyze strategy performance by classification state/regime.

This script shows how easy it is to find top performers per regime
once the grid search data is available.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import json


def analyze_strategies_by_regime(workspace_path: str, min_trades: int = 50):
    """
    Analyze strategy performance broken down by classifier regimes.
    
    The key insight: Since we have sparse signal storage, we can efficiently
    compute performance metrics for each strategy under each regime state.
    """
    con = duckdb.connect()
    
    print("STRATEGY PERFORMANCE BY CLASSIFICATION STATE")
    print("=" * 80)
    
    # Get all classifier types from the workspace
    classifier_query = """
    SELECT DISTINCT 
        split_part(file, '/', -3) as classifier_type,
        split_part(file, '/', -1) as classifier_params
    FROM glob('{}/traces/*/classifiers/*/*.parquet')
    """.format(workspace_path)
    
    classifiers = con.execute(classifier_query).df()
    
    results = []
    
    for _, classifier in classifiers.iterrows():
        classifier_type = classifier['classifier_type']
        classifier_file = f"{workspace_path}/traces/SPY_1m/classifiers/{classifier_type}/{classifier['classifier_params']}"
        
        print(f"\n\nAnalyzing classifier: {classifier_type}")
        print("-" * 60)
        
        # First, let's see the regime distribution
        regime_dist_query = f"""
        WITH sparse_regimes AS (
            SELECT idx, val as regime
            FROM read_parquet('{classifier_file}')
        ),
        regime_changes AS (
            SELECT 
                regime,
                idx as start_idx,
                LEAD(idx, 1, 80000) OVER (ORDER BY idx) as end_idx
            FROM sparse_regimes
        )
        SELECT 
            regime,
            SUM(end_idx - start_idx) as total_bars,
            COUNT(*) as num_transitions
        FROM regime_changes
        GROUP BY regime
        """
        
        regime_dist = con.execute(regime_dist_query).df()
        total_bars = regime_dist['total_bars'].sum()
        
        print("\nRegime Distribution:")
        for _, r in regime_dist.iterrows():
            pct = r['total_bars'] / total_bars * 100
            print(f"  Regime {r['regime']}: {pct:.1f}% ({r['total_bars']:,} bars, {r['num_transitions']} transitions)")
        
        # Now analyze each strategy's performance under each regime
        strategy_query = """
        SELECT DISTINCT 
            split_part(file, '/', -2) as strategy_type,
            split_part(file, '/', -1) as strategy_params
        FROM glob('{}/traces/*/signals/*/*.parquet')
        """.format(workspace_path)
        
        strategies = con.execute(strategy_query).df()
        
        for _, strategy in strategies.iterrows():
            strategy_name = f"{strategy['strategy_type']}_{strategy['strategy_params'].replace('.parquet', '')}"
            signal_file = f"{workspace_path}/traces/SPY_1m/signals/{strategy['strategy_type']}/{strategy['strategy_params']}"
            
            # This is the key query that makes it easy!
            performance_query = f"""
            WITH 
            -- Expand classifier regimes
            classifier_expanded AS (
                SELECT 
                    idx as start_idx,
                    val as regime,
                    LEAD(idx, 1, 80000) OVER (ORDER BY idx) as end_idx
                FROM read_parquet('{classifier_file}')
            ),
            -- Expand signals  
            signal_expanded AS (
                SELECT 
                    idx as start_idx,
                    val as signal,
                    LEAD(idx, 1, 80000) OVER (ORDER BY idx) as end_idx
                FROM read_parquet('{signal_file}')
                WHERE val != 0  -- Only care about active signals
            ),
            -- Get price data
            prices AS (
                SELECT 
                    bar_index as idx,
                    close,
                    LEAD(close, 1) OVER (ORDER BY bar_index) as next_close
                FROM read_parquet('{workspace_path}/../data/SPY_1m.parquet')
            ),
            -- Join everything together
            regime_signals AS (
                SELECT 
                    c.regime,
                    s.signal,
                    p.close,
                    p.next_close,
                    CASE 
                        WHEN s.signal = 1 THEN (p.next_close - p.close) / p.close
                        WHEN s.signal = -1 THEN (p.close - p.next_close) / p.close
                        ELSE 0
                    END as bar_return
                FROM classifier_expanded c
                JOIN signal_expanded s 
                    ON s.start_idx >= c.start_idx 
                    AND s.start_idx < c.end_idx
                JOIN prices p 
                    ON p.idx = s.start_idx
                WHERE p.next_close IS NOT NULL
            )
            -- Calculate metrics by regime
            SELECT 
                regime,
                COUNT(*) as num_signals,
                AVG(bar_return) * 100 as avg_return_pct,
                STDDEV(bar_return) * 100 as volatility_pct,
                SUM(CASE WHEN bar_return > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) * 100 as win_rate,
                CASE 
                    WHEN STDDEV(bar_return) > 0 
                    THEN AVG(bar_return) / STDDEV(bar_return) * SQRT(252 * 390)
                    ELSE 0 
                END as sharpe_ratio
            FROM regime_signals
            GROUP BY regime
            HAVING COUNT(*) >= {min_trades}
            """
            
            try:
                regime_performance = con.execute(performance_query).df()
                
                for _, perf in regime_performance.iterrows():
                    results.append({
                        'classifier': classifier_type,
                        'classifier_params': classifier['classifier_params'],
                        'strategy': strategy['strategy_type'],
                        'strategy_params': strategy['strategy_params'],
                        'regime': perf['regime'],
                        'num_signals': perf['num_signals'],
                        'avg_return_pct': perf['avg_return_pct'],
                        'volatility_pct': perf['volatility_pct'],
                        'win_rate': perf['win_rate'],
                        'sharpe_ratio': perf['sharpe_ratio']
                    })
            except Exception as e:
                # Some strategy-classifier pairs might not have enough data
                continue
    
    # Convert to DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    
    # Now the fun part - finding top performers per regime!
    print("\n\n" + "=" * 80)
    print("TOP PERFORMERS BY REGIME")
    print("=" * 80)
    
    # Group by classifier and regime to find top strategies
    for classifier in results_df['classifier'].unique():
        print(f"\n\nClassifier: {classifier}")
        print("-" * 60)
        
        classifier_data = results_df[results_df['classifier'] == classifier]
        
        for regime in sorted(classifier_data['regime'].unique()):
            regime_data = classifier_data[classifier_data['regime'] == regime]
            
            # Sort by Sharpe ratio and get top 5
            top_strategies = regime_data.nlargest(5, 'sharpe_ratio')
            
            print(f"\nRegime: {regime}")
            print(f"{'Strategy':<40} {'Sharpe':<10} {'Avg Return':<12} {'Win Rate':<10} {'Trades':<8}")
            print("-" * 80)
            
            for _, row in top_strategies.iterrows():
                strategy_full = f"{row['strategy']}_{row['strategy_params'].replace('.parquet', '')}"
                print(f"{strategy_full:<40} {row['sharpe_ratio']:>9.3f} {row['avg_return_pct']:>11.4f}% {row['win_rate']:>9.1f}% {row['num_signals']:>7}")
    
    # Save results for further analysis
    results_df.to_csv(f"{workspace_path}/strategy_regime_performance.csv", index=False)
    
    # Create a pivot table for easy visualization
    print("\n\n" + "=" * 80)
    print("PERFORMANCE HEATMAP DATA")
    print("=" * 80)
    
    # Average Sharpe by strategy type and regime
    pivot_data = results_df.groupby(['strategy', 'regime'])['sharpe_ratio'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='strategy', columns='regime', values='sharpe_ratio')
    
    print("\nAverage Sharpe Ratio by Strategy Type and Regime:")
    print(pivot_table.round(3))
    
    # Save summary statistics
    summary_stats = {
        'total_strategies_analyzed': len(results_df['strategy'].unique()),
        'total_classifiers_analyzed': len(results_df['classifier'].unique()),
        'total_regimes_analyzed': len(results_df['regime'].unique()),
        'top_overall_sharpe': results_df.nlargest(10, 'sharpe_ratio')[['strategy', 'regime', 'sharpe_ratio']].to_dict('records'),
        'most_consistent_strategies': results_df.groupby('strategy')['sharpe_ratio'].agg(['mean', 'std']).nlargest(10, 'mean').to_dict()
    }
    
    with open(f"{workspace_path}/regime_analysis_summary.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    return results_df


def create_ensemble_recommendations(results_df: pd.DataFrame, top_n: int = 3):
    """
    Create ensemble strategy recommendations based on regime analysis.
    """
    print("\n\n" + "=" * 80)
    print("ENSEMBLE STRATEGY RECOMMENDATIONS")
    print("=" * 80)
    
    ensemble_config = {}
    
    # For each classifier
    for classifier in results_df['classifier'].unique():
        classifier_data = results_df[results_df['classifier'] == classifier]
        ensemble_config[classifier] = {}
        
        print(f"\n{classifier}:")
        
        # For each regime in this classifier
        for regime in sorted(classifier_data['regime'].unique()):
            regime_data = classifier_data[classifier_data['regime'] == regime]
            
            # Get top N strategies
            top_strategies = regime_data.nlargest(top_n, 'sharpe_ratio')
            
            # Only include if performance is good enough
            good_strategies = top_strategies[
                (top_strategies['sharpe_ratio'] > 0.5) & 
                (top_strategies['win_rate'] > 45)
            ]
            
            if len(good_strategies) > 0:
                strategies = []
                for _, row in good_strategies.iterrows():
                    strategies.append({
                        'name': row['strategy'],
                        'params': row['strategy_params'].replace('.parquet', ''),
                        'expected_sharpe': round(row['sharpe_ratio'], 3),
                        'win_rate': round(row['win_rate'], 1)
                    })
                
                ensemble_config[classifier][regime] = strategies
                
                print(f"  {regime}: {len(strategies)} strategies selected")
                for s in strategies:
                    print(f"    - {s['name']} (Sharpe: {s['expected_sharpe']})")
    
    # Save ensemble configuration
    with open('recommended_ensemble_config.json', 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    return ensemble_config


def quick_regime_performance_check(workspace_path: str, strategy_name: str, classifier_name: str):
    """
    Quick function to check a specific strategy's performance across regimes.
    
    This is what you'll use most often during analysis.
    """
    con = duckdb.connect()
    
    # This simple query gives you everything you need!
    query = f"""
    WITH regime_performance AS (
        -- Your magic query that joins signals with regimes
        -- and calculates performance metrics
    )
    SELECT * FROM regime_performance
    WHERE strategy = '{strategy_name}' 
    AND classifier = '{classifier_name}'
    ORDER BY sharpe_ratio DESC
    """
    
    return con.execute(query).df()


if __name__ == "__main__":
    # Example usage
    workspace_path = "/Users/daws/ADMF-PC/workspaces/expansive_grid_search_xxxxx"
    
    # Main analysis
    results_df = analyze_strategies_by_regime(workspace_path)
    
    # Create ensemble recommendations
    ensemble_config = create_ensemble_recommendations(results_df)
    
    print("\n\nAnalysis complete! Check the output files:")
    print(f"  - {workspace_path}/strategy_regime_performance.csv")
    print(f"  - {workspace_path}/regime_analysis_summary.json")
    print("  - ./recommended_ensemble_config.json")