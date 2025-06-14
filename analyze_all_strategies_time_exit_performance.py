#!/usr/bin/env python3
"""
All Strategies Time Exit Performance Analysis

Analyzes all available strategies in the workspace to identify which ones
have good 'exit after x bars' performance. This helps us find strategies
that could benefit from similar exit frameworks to our RSI strategy.

Analyzes:
1. All strategy signals in workspace traces
2. Performance at different exit timeframes (5, 10, 15, 18, 20, 25, 30 bars)
3. Win rates, average returns, Sharpe ratios, and risk metrics
4. Identifies strategies with consistent positive edges
5. Ranks strategies by time-exit performance
"""
import duckdb
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path


def analyze_all_strategies_time_exit_performance(workspace_path: str, data_path: str):
    """Analyze time exit performance across all available strategies."""
    
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print("=== ALL STRATEGIES TIME EXIT PERFORMANCE ANALYSIS ===\n")
    print("Scanning workspace for all available strategy signals...")
    
    # Find all strategy signal directories
    traces_path = Path(workspace_path) / "traces" / "SPY_1m" / "signals"
    
    if not traces_path.exists():
        print(f"Error: Traces path not found: {traces_path}")
        return
    
    strategy_dirs = [d for d in traces_path.iterdir() if d.is_dir()]
    print(f"Found {len(strategy_dirs)} strategy types: {[d.name for d in strategy_dirs]}")
    
    # Exit timeframes to test
    exit_bars = [5, 10, 15, 18, 20, 25, 30]
    
    all_results = []
    
    for strategy_dir in strategy_dirs:
        strategy_name = strategy_dir.name
        print(f"\n--- Analyzing {strategy_name} ---")
        
        # Find all parquet files for this strategy
        parquet_files = list(strategy_dir.glob("*.parquet"))
        
        if not parquet_files:
            print(f"No parquet files found for {strategy_name}")
            continue
            
        print(f"Found {len(parquet_files)} variants for {strategy_name}")
        
        # Analyze each variant
        for parquet_file in parquet_files[:10]:  # Limit to first 10 to avoid too much processing
            variant_name = parquet_file.stem
            
            try:
                # Get all entry signals for this variant (both long and short)
                entry_query = f"""
                SELECT DISTINCT idx as entry_idx, val as signal_direction
                FROM read_parquet('{parquet_file}')
                WHERE val = 1 OR val = -1
                """
                
                entries_df = con.execute(entry_query).df()
                
                if len(entries_df) == 0:
                    continue
                    
                print(f"  {variant_name}: {len(entries_df)} entry signals")
                
                # Test each exit timeframe
                for exit_period in exit_bars:
                    try:
                        performance_query = f"""
                        WITH entries AS (
                            SELECT DISTINCT idx as entry_idx, val as signal_direction
                            FROM read_parquet('{parquet_file}')
                            WHERE val = 1 OR val = -1
                        ),
                        trades AS (
                            SELECT 
                                e.entry_idx,
                                e.entry_idx + {exit_period} as exit_idx,
                                e.signal_direction
                            FROM entries e
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
                            ROUND(AVG(
                                CASE WHEN t.signal_direction = 1 
                                THEN (m2.close - m1.close) / m1.close * 100
                                ELSE (m1.close - m2.close) / m1.close * 100
                                END
                            ) / STDDEV(
                                CASE WHEN t.signal_direction = 1 
                                THEN (m2.close - m1.close) / m1.close * 100
                                ELSE (m1.close - m2.close) / m1.close * 100
                                END
                            ), 3) as sharpe_ratio,
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
                        """
                        
                        result = con.execute(performance_query).df()
                        
                        if len(result) > 0 and result['total_trades'].iloc[0] >= 3:  # Lower minimum sample size
                            result_row = result.iloc[0]
                            all_results.append({
                                'strategy_type': strategy_name,
                                'variant': variant_name,
                                'exit_bars': exit_period,
                                'total_trades': result_row['total_trades'],
                                'avg_return_pct': result_row['avg_return_pct'],
                                'volatility': result_row['volatility'],
                                'win_rate': result_row['win_rate'],
                                'sharpe_ratio': result_row['sharpe_ratio'],
                                'total_return': result_row['total_return'],
                                'worst_trade': result_row['worst_trade'],
                                'best_trade': result_row['best_trade']
                            })
                            
                    except Exception as e:
                        print(f"    Error analyzing {exit_period}-bar exit: {e}")
                        continue
                        
            except Exception as e:
                print(f"  Error analyzing {variant_name}: {e}")
                continue
    
    con.close()
    
    if not all_results:
        print("No valid results found across all strategies")
        return
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    print(f"\n=== COMPREHENSIVE RESULTS SUMMARY ===")
    print(f"Total strategy-variant-timeframe combinations analyzed: {len(results_df)}")
    print(f"Unique strategies: {results_df['strategy_type'].nunique()}")
    print(f"Unique variants: {results_df['variant'].nunique()}")
    
    # 1. Top Performers by Sharpe Ratio
    print(f"\n1. TOP 20 PERFORMERS BY SHARPE RATIO:")
    top_sharpe = results_df.nlargest(20, 'sharpe_ratio')[
        ['strategy_type', 'variant', 'exit_bars', 'total_trades', 'avg_return_pct', 'win_rate', 'sharpe_ratio']
    ]
    print(top_sharpe.to_string(index=False))
    
    # 2. Top Performers by Average Return
    print(f"\n2. TOP 20 PERFORMERS BY AVERAGE RETURN:")
    top_return = results_df.nlargest(20, 'avg_return_pct')[
        ['strategy_type', 'variant', 'exit_bars', 'total_trades', 'avg_return_pct', 'win_rate', 'sharpe_ratio']
    ]
    print(top_return.to_string(index=False))
    
    # 3. Most Consistent Performers (Positive returns across multiple timeframes)
    print(f"\n3. CONSISTENCY ANALYSIS - Strategies with positive returns across multiple timeframes:")
    consistency_analysis = results_df[results_df['avg_return_pct'] > 0].groupby(['strategy_type', 'variant']).agg({
        'exit_bars': 'count',
        'avg_return_pct': 'mean',
        'win_rate': 'mean',
        'sharpe_ratio': 'mean',
        'total_trades': 'mean'
    }).rename(columns={
        'exit_bars': 'positive_timeframes',
        'avg_return_pct': 'avg_return_across_timeframes',
        'win_rate': 'avg_win_rate',
        'sharpe_ratio': 'avg_sharpe',
        'total_trades': 'avg_trades'
    }).sort_values('positive_timeframes', ascending=False)
    
    print(consistency_analysis.head(15).round(4).to_string())
    
    # 4. Strategy Type Performance Summary
    print(f"\n4. STRATEGY TYPE SUMMARY:")
    strategy_summary = results_df.groupby('strategy_type').agg({
        'avg_return_pct': ['mean', 'std', 'count'],
        'win_rate': 'mean',
        'sharpe_ratio': 'mean',
        'total_trades': 'sum'
    }).round(4)
    
    # Flatten column names
    strategy_summary.columns = ['_'.join(col).strip() for col in strategy_summary.columns]
    print(strategy_summary.to_string())
    
    # 5. Optimal Exit Timeframe Analysis
    print(f"\n5. OPTIMAL EXIT TIMEFRAME ANALYSIS:")
    timeframe_analysis = results_df.groupby('exit_bars').agg({
        'avg_return_pct': 'mean',
        'win_rate': 'mean', 
        'sharpe_ratio': 'mean',
        'total_trades': 'sum'
    }).round(4)
    print(timeframe_analysis.to_string())
    
    # 6. High-Quality Strategy Candidates
    print(f"\n6. HIGH-QUALITY STRATEGY CANDIDATES:")
    print("(Strategies with Sharpe > 0.1, Win Rate > 52%, Sample Size > 100)")
    
    high_quality = results_df[
        (results_df['sharpe_ratio'] > 0.1) & 
        (results_df['win_rate'] > 52) & 
        (results_df['total_trades'] > 100)
    ].sort_values('sharpe_ratio', ascending=False)
    
    if len(high_quality) > 0:
        print(high_quality[['strategy_type', 'variant', 'exit_bars', 'total_trades', 'avg_return_pct', 'win_rate', 'sharpe_ratio']].to_string(index=False))
    else:
        print("No strategies meet high-quality criteria")
    
    # 7. Risk-Adjusted Performance
    print(f"\n7. RISK-ADJUSTED PERFORMANCE (Return/Risk Ratio):")
    results_df['return_risk_ratio'] = results_df['avg_return_pct'] / results_df['volatility']
    top_risk_adjusted = results_df.nlargest(15, 'return_risk_ratio')[
        ['strategy_type', 'variant', 'exit_bars', 'avg_return_pct', 'volatility', 'return_risk_ratio', 'sharpe_ratio']
    ]
    print(top_risk_adjusted.to_string(index=False))
    
    # 8. Strategy Recommendations
    print(f"\n=== STRATEGY IMPLEMENTATION RECOMMENDATIONS ===")
    
    # Find best performing strategies
    best_strategies = results_df[results_df['sharpe_ratio'] > 0.05].groupby(['strategy_type', 'variant']).first().sort_values('sharpe_ratio', ascending=False)
    
    if len(best_strategies) > 0:
        print(f"\n🏆 TOP STRATEGY CANDIDATES FOR EXIT FRAMEWORK IMPLEMENTATION:")
        
        for i, (strategy_key, row) in enumerate(best_strategies.head(5).iterrows()):
            strategy_type, variant = strategy_key
            print(f"\n{i+1}. {strategy_type} - {variant}")
            print(f"   ✓ Optimal Exit: {row['exit_bars']} bars")
            print(f"   ✓ Performance: {row['avg_return_pct']}% avg return, {row['win_rate']}% win rate")
            print(f"   ✓ Risk Profile: {row['sharpe_ratio']} Sharpe, {row['volatility']}% volatility")
            print(f"   ✓ Sample Size: {row['total_trades']} trades")
            
            # Implementation suggestions
            if row['sharpe_ratio'] > 0.1:
                print(f"   🟢 EXCELLENT - Ready for exit framework implementation")
            elif row['sharpe_ratio'] > 0.05:
                print(f"   🟡 GOOD - Consider with additional filters")
            else:
                print(f"   🟠 MARGINAL - Needs improvement before implementation")
    
    print(f"\n📋 IMPLEMENTATION PRIORITIES:")
    print(f"1. Focus on strategies with Sharpe > 0.1 and large sample sizes")
    print(f"2. Test multiple exit timeframes (optimal seems to be {timeframe_analysis['sharpe_ratio'].idxmax()} bars)")
    print(f"3. Consider multi-layer exit frameworks for top performers")
    print(f"4. Add regime filtering for consistent performers")
    print(f"5. Combine multiple signals from different strategy types")
    
    # Save detailed results
    output_file = f"{workspace_path}/all_strategies_time_exit_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return results_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python analyze_all_strategies_time_exit_performance.py <workspace_path> <data_path>")
        sys.exit(1)
    
    analyze_all_strategies_time_exit_performance(sys.argv[1], sys.argv[2])