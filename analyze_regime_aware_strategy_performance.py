#!/usr/bin/env python3
"""
Regime-Aware Strategy Performance Analysis

Analyzes strategy performance within specific market regimes using
expanded classifier data. Filters for minimum trade counts to ensure
statistical significance.

Features:
1. Expands sparse classifier data to regime-per-bar
2. Filters strategies by minimum trade count per regime
3. Compares strategy performance across regimes
4. Identifies regime-specific top performers
"""

import duckdb
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path


def expand_classifier_regimes(con, classifier_file, max_idx=80000):
    """Expand sparse classifier data to regime-per-bar."""
    
    expand_query = f'''
    WITH regime_changes AS (
        SELECT idx, val,
               LEAD(idx, 1, {max_idx}) OVER (ORDER BY idx) as next_idx
        FROM read_parquet('{classifier_file}')
        ORDER BY idx
    ),
    expanded AS (
        SELECT r.idx + i as bar_idx, r.val as regime
        FROM regime_changes r
        CROSS JOIN generate_series(0, r.next_idx - r.idx - 1) as t(i)
        WHERE r.idx + i < {max_idx}
    )
    SELECT bar_idx, regime
    FROM expanded
    ORDER BY bar_idx
    '''
    
    return con.execute(expand_query).df()


def analyze_strategy_in_regime(con, strategy_file, regime_df, regime_name, data_path, exit_bars=[5, 10, 15, 18, 20, 25, 30]):
    """Analyze strategy performance within a specific regime."""
    
    results = []
    
    # Get strategy info
    strategy_name = strategy_file.split('/')[-1].replace('.parquet', '')
    strategy_type = strategy_file.split('/')[-2]
    
    for exit_period in exit_bars:
        try:
            # Get regime-filtered entries
            regime_analysis_query = f'''
            WITH regime_bars AS (
                SELECT bar_idx
                FROM regime_df
                WHERE regime = '{regime_name}'
            ),
            strategy_entries AS (
                SELECT DISTINCT s.idx as entry_idx, s.val as signal_direction
                FROM read_parquet('{strategy_file}') s
                INNER JOIN regime_bars r ON s.idx = r.bar_idx
                WHERE s.val = 1 OR s.val = -1
            ),
            trades AS (
                SELECT 
                    e.entry_idx,
                    e.entry_idx + {exit_period} as exit_idx,
                    e.signal_direction
                FROM strategy_entries e
            )
            SELECT 
                COUNT(*) as total_trades,
                ROUND(AVG(
                    CASE WHEN t.signal_direction = 1 
                    THEN (m2.close - m1.close) / m1.close * 100
                    ELSE (m1.close - m2.close) / m1.close * 100
                    END
                ), 4) as avg_return_pct,
                ROUND(STDDEV(
                    CASE WHEN t.signal_direction = 1 
                    THEN (m2.close - m1.close) / m1.close * 100
                    ELSE (m1.close - m2.close) / m1.close * 100
                    END
                ), 4) as volatility,
                COUNT(CASE WHEN 
                    (t.signal_direction = 1 AND (m2.close - m1.close) / m1.close > 0) OR
                    (t.signal_direction = -1 AND (m1.close - m2.close) / m1.close > 0)
                    THEN 1 END) as winners,
                ROUND(COUNT(CASE WHEN 
                    (t.signal_direction = 1 AND (m2.close - m1.close) / m1.close > 0) OR
                    (t.signal_direction = -1 AND (m1.close - m2.close) / m1.close > 0)
                    THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate,
                ROUND(SUM(
                    CASE WHEN t.signal_direction = 1 
                    THEN (m2.close - m1.close) / m1.close * 100
                    ELSE (m1.close - m2.close) / m1.close * 100
                    END
                ), 2) as total_return,
                ROUND(MIN(
                    CASE WHEN t.signal_direction = 1 
                    THEN (m2.close - m1.close) / m1.close * 100
                    ELSE (m1.close - m2.close) / m1.close * 100
                    END
                ), 3) as worst_trade,
                ROUND(MAX(
                    CASE WHEN t.signal_direction = 1 
                    THEN (m2.close - m1.close) / m1.close * 100
                    ELSE (m1.close - m2.close) / m1.close * 100
                    END
                ), 3) as best_trade
            FROM trades t
            JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
            WHERE t.exit_idx <= (SELECT MAX(bar_index) FROM read_parquet('{data_path}'))
            '''
            
            # Execute with regime data as a table
            con.execute("CREATE OR REPLACE TABLE regime_df AS SELECT * FROM regime_df")
            result = con.execute(regime_analysis_query).df()
            
            if len(result) > 0 and result['total_trades'].iloc[0] > 0:
                result_row = result.iloc[0]
                
                # Calculate Sharpe ratio
                if result_row['volatility'] and result_row['volatility'] > 0:
                    sharpe_ratio = result_row['avg_return_pct'] / result_row['volatility']
                else:
                    sharpe_ratio = 0
                
                results.append({
                    'strategy_type': strategy_type,
                    'strategy_name': strategy_name,
                    'regime': regime_name,
                    'exit_bars': exit_period,
                    'total_trades': result_row['total_trades'],
                    'avg_return_pct': result_row['avg_return_pct'],
                    'volatility': result_row['volatility'],
                    'win_rate': result_row['win_rate'],
                    'sharpe_ratio': sharpe_ratio,
                    'total_return': result_row['total_return'],
                    'worst_trade': result_row['worst_trade'],
                    'best_trade': result_row['best_trade']
                })
                
        except Exception as e:
            print(f"      Error analyzing {exit_period}-bar exit: {e}")
            continue
    
    return results


def analyze_regime_aware_performance(workspace_path: str, data_path: str, min_trades: int = 250):
    """Analyze strategy performance within different market regimes."""
    
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print("=== REGIME-AWARE STRATEGY PERFORMANCE ANALYSIS ===\n")
    
    # Find suitable classifier
    classifier_file = f'{workspace_path}/traces/SPY_1m/classifiers/momentum_regime_grid/SPY_momentum_regime_grid_65_35_015.parquet'
    
    if not os.path.exists(classifier_file):
        print(f"Error: Classifier file not found: {classifier_file}")
        return
    
    print("Step 1: Expanding momentum regime classifier...")
    regime_df = expand_classifier_regimes(con, classifier_file)
    
    # Register regime data
    con.register('regime_df', regime_df)
    
    print(f"Regime distribution:")
    regime_dist = regime_df['regime'].value_counts()
    for regime, count in regime_dist.items():
        pct = count / len(regime_df) * 100
        print(f"  {regime}: {count:,} bars ({pct:.1f}%)")
    
    print(f"\nStep 2: Analyzing strategy performance by regime...")
    
    # Find all strategy files
    strategy_types = ['mean_reversion_grid', 'rsi_grid', 'ma_crossover_grid', 'momentum_grid']
    
    all_results = []
    
    for strategy_type in strategy_types:
        strategy_path = f'{workspace_path}/traces/SPY_1m/signals/{strategy_type}'
        
        if not os.path.exists(strategy_path):
            print(f"  Skipping {strategy_type}: Directory not found")
            continue
            
        strategy_files = glob.glob(f'{strategy_path}/*.parquet')
        print(f"\n  --- {strategy_type} ({len(strategy_files)} variants) ---")
        
        for strategy_file in strategy_files[:5]:  # Limit for performance
            strategy_name = strategy_file.split('/')[-1].replace('.parquet', '')
            print(f"    Analyzing {strategy_name}...")
            
            # Analyze in each regime
            for regime_name in regime_dist.index:
                print(f"      {regime_name} regime...")
                
                regime_results = analyze_strategy_in_regime(
                    con, strategy_file, regime_df, regime_name, data_path
                )
                
                all_results.extend(regime_results)
    
    con.close()
    
    if not all_results:
        print("No valid results found")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    print(f"\n=== REGIME-AWARE ANALYSIS RESULTS ===")
    print(f"Total strategy-regime-timeframe combinations: {len(results_df)}")
    
    # Filter by minimum trades
    significant_results = results_df[results_df['total_trades'] >= min_trades].copy()
    
    if len(significant_results) == 0:
        print(f"\nNo strategies meet {min_trades}+ trade requirement in any regime")
        print(f"\nTop strategies by trade count:")
        top_by_trades = results_df.nlargest(20, 'total_trades')
        print(top_by_trades[['strategy_type', 'strategy_name', 'regime', 'exit_bars', 'total_trades', 'avg_return_pct', 'sharpe_ratio']].to_string(index=False))
        return
    
    print(f"\nStrategies with {min_trades}+ trades: {len(significant_results)}")
    
    # 1. Top performers by regime
    print(f"\n1. TOP PERFORMERS BY REGIME (Sharpe Ratio):")
    for regime in significant_results['regime'].unique():
        regime_data = significant_results[significant_results['regime'] == regime]
        top_in_regime = regime_data.nlargest(10, 'sharpe_ratio')
        
        print(f"\n   === {regime.upper()} REGIME ===")
        print(top_in_regime[['strategy_type', 'strategy_name', 'exit_bars', 'total_trades', 'avg_return_pct', 'win_rate', 'sharpe_ratio']].to_string(index=False))
    
    # 2. Regime comparison for top strategies
    print(f"\n2. REGIME COMPARISON FOR TOP STRATEGIES:")
    
    # Find strategies that perform well in both regimes
    strategy_regime_comparison = significant_results.groupby(['strategy_type', 'strategy_name', 'exit_bars']).agg({
        'regime': 'count',
        'total_trades': 'mean',
        'avg_return_pct': 'mean',
        'sharpe_ratio': 'mean'
    }).rename(columns={'regime': 'regimes_analyzed'})
    
    # Only strategies analyzed in both regimes
    both_regimes = strategy_regime_comparison[strategy_regime_comparison['regimes_analyzed'] == 2]
    
    if len(both_regimes) > 0:
        print("Strategies performing well in BOTH regimes:")
        both_regimes_sorted = both_regimes.sort_values('sharpe_ratio', ascending=False)
        print(both_regimes_sorted.head(10).round(4).to_string())
    
    # 3. Regime-specific winners
    print(f"\n3. REGIME-SPECIFIC ANALYSIS:")
    
    regime_summary = significant_results.groupby('regime').agg({
        'total_trades': ['count', 'mean'],
        'avg_return_pct': 'mean',
        'win_rate': 'mean',
        'sharpe_ratio': 'mean'
    }).round(4)
    
    print("\nRegime performance summary:")
    print(regime_summary.to_string())
    
    # 4. Strategy type performance by regime
    print(f"\n4. STRATEGY TYPE PERFORMANCE BY REGIME:")
    
    strategy_regime_perf = significant_results.groupby(['strategy_type', 'regime']).agg({
        'total_trades': 'sum',
        'avg_return_pct': 'mean',
        'win_rate': 'mean',
        'sharpe_ratio': 'mean'
    }).round(4)
    
    print(strategy_regime_perf.to_string())
    
    # 5. Recommendations
    print(f"\n=== REGIME-AWARE RECOMMENDATIONS ===")
    
    # Find best strategy for each regime
    for regime in significant_results['regime'].unique():
        regime_data = significant_results[significant_results['regime'] == regime]
        best_in_regime = regime_data.loc[regime_data['sharpe_ratio'].idxmax()]
        
        print(f"\n🏆 BEST FOR {regime.upper()} REGIME:")
        print(f"   Strategy: {best_in_regime['strategy_type']} - {best_in_regime['strategy_name']}")
        print(f"   Exit: {best_in_regime['exit_bars']} bars")
        print(f"   Performance: {best_in_regime['avg_return_pct']:.4f}% avg return, {best_in_regime['win_rate']:.1f}% win rate")
        print(f"   Risk-Adjusted: {best_in_regime['sharpe_ratio']:.3f} Sharpe ratio")
        print(f"   Sample Size: {best_in_regime['total_trades']:,} trades")
    
    # Save results
    output_file = f"{workspace_path}/regime_aware_strategy_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return results_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python analyze_regime_aware_strategy_performance.py <workspace_path> <data_path> [min_trades]")
        sys.exit(1)
    
    min_trades = int(sys.argv[3]) if len(sys.argv) > 3 else 250
    analyze_regime_aware_performance(sys.argv[1], sys.argv[2], min_trades)