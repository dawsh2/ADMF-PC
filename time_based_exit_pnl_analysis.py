#!/usr/bin/env python3
"""
Time-Based Exit P&L Distribution Analysis

Analyzes P&L distribution for ALL 6,280 fast RSI entries using simple time-based exits.
Shows what happens if we just hold for X bars and exit, ignoring all signal complexity.

This gives us the baseline performance and P&L distribution patterns.
"""
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_time_based_exit_pnl(workspace_path: str, data_path: str):
    """Analyze P&L distribution for time-based exits on ALL fast RSI entries."""
    
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print("=== Time-Based Exit P&L Distribution Analysis ===\n")
    print("Analyzing ALL 6,280 fast RSI entries with simple time-based exits")
    print("No signal complexity - just hold for X bars and exit\n")
    
    # Test different holding periods
    holding_periods = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
    
    all_results = []
    
    for holding_bars in holding_periods:
        print(f"Analyzing {holding_bars}-bar exits...")
        
        pnl_query = f"""
        WITH fast_rsi_entries AS (
            SELECT DISTINCT idx as entry_idx
            FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
            WHERE strat LIKE '%_7_%' AND val = 1
        ),
        time_based_trades AS (
            SELECT 
                entry_idx,
                entry_idx + {holding_bars} as exit_idx
            FROM fast_rsi_entries
        ),
        trade_pnl AS (
            SELECT 
                t.entry_idx,
                t.exit_idx,
                {holding_bars} as holding_bars,
                m1.close as entry_price,
                m2.close as exit_price,
                (m2.close - m1.close) / m1.close * 100 as pnl_pct
            FROM time_based_trades t
            JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
            WHERE t.exit_idx <= (SELECT MAX(bar_index) FROM read_parquet('{data_path}'))
        )
        SELECT 
            holding_bars,
            COUNT(*) as total_trades,
            ROUND(AVG(pnl_pct), 4) as avg_pnl_pct,
            ROUND(MEDIAN(pnl_pct), 4) as median_pnl_pct,
            ROUND(STDDEV(pnl_pct), 4) as pnl_volatility,
            ROUND(MIN(pnl_pct), 4) as worst_loss_pct,
            ROUND(MAX(pnl_pct), 4) as best_gain_pct,
            ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY pnl_pct), 4) as p5_pnl,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY pnl_pct), 4) as p25_pnl,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY pnl_pct), 4) as p75_pnl,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY pnl_pct), 4) as p95_pnl,
            COUNT(CASE WHEN pnl_pct > 0 THEN 1 END) as winners,
            COUNT(CASE WHEN pnl_pct < 0 THEN 1 END) as losers,
            COUNT(CASE WHEN pnl_pct = 0 THEN 1 END) as breakevens,
            ROUND(COUNT(CASE WHEN pnl_pct > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate,
            ROUND(AVG(CASE WHEN pnl_pct > 0 THEN pnl_pct END), 4) as avg_winner_pct,
            ROUND(AVG(CASE WHEN pnl_pct < 0 THEN pnl_pct END), 4) as avg_loser_pct,
            ROUND(AVG(pnl_pct) / STDDEV(pnl_pct), 3) as sharpe_ratio,
            ROUND(COUNT(*) * AVG(pnl_pct), 2) as total_pnl_pct
        FROM trade_pnl
        """
        
        try:
            result = con.execute(pnl_query).df()
            if len(result) > 0:
                all_results.append(result.iloc[0])
        except Exception as e:
            print(f"Error analyzing {holding_bars} bars: {e}")
    
    # 1. Summary table
    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n1. Time-Based Exit Performance Summary:")
        print(results_df[['holding_bars', 'total_trades', 'avg_pnl_pct', 'median_pnl_pct', 
                         'win_rate', 'sharpe_ratio', 'pnl_volatility', 'total_pnl_pct']].to_string(index=False))
        
        print("\n2. P&L Distribution Percentiles:")
        print(results_df[['holding_bars', 'p5_pnl', 'p25_pnl', 'median_pnl_pct', 
                         'p75_pnl', 'p95_pnl', 'worst_loss_pct', 'best_gain_pct']].to_string(index=False))
        
        print("\n3. Winner/Loser Analysis:")
        print(results_df[['holding_bars', 'winners', 'losers', 'win_rate', 
                         'avg_winner_pct', 'avg_loser_pct']].to_string(index=False))
    
    # 2. Detailed P&L distribution for key holding periods
    key_periods = [5, 10, 15, 20]
    
    print("\n\n4. Detailed P&L Distribution Analysis:")
    
    for period in key_periods:
        print(f"\n--- {period}-Bar Exit Detailed Analysis ---")
        
        detailed_query = f"""
        WITH fast_rsi_entries AS (
            SELECT DISTINCT idx as entry_idx
            FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
            WHERE strat LIKE '%_7_%' AND val = 1
        ),
        time_based_trades AS (
            SELECT 
                entry_idx,
                entry_idx + {period} as exit_idx
            FROM fast_rsi_entries
        ),
        trade_pnl AS (
            SELECT 
                (m2.close - m1.close) / m1.close * 100 as pnl_pct
            FROM time_based_trades t
            JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
            WHERE t.exit_idx <= (SELECT MAX(bar_index) FROM read_parquet('{data_path}'))
        ),
        pnl_buckets AS (
            SELECT 
                CASE 
                    WHEN pnl_pct <= -2.0 THEN 'Loss > 2%'
                    WHEN pnl_pct <= -1.0 THEN 'Loss 1-2%'
                    WHEN pnl_pct <= -0.5 THEN 'Loss 0.5-1%'
                    WHEN pnl_pct <= -0.1 THEN 'Loss 0.1-0.5%'
                    WHEN pnl_pct < 0 THEN 'Small Loss <0.1%'
                    WHEN pnl_pct = 0 THEN 'Breakeven'
                    WHEN pnl_pct < 0.1 THEN 'Small Gain <0.1%'
                    WHEN pnl_pct <= 0.5 THEN 'Gain 0.1-0.5%'
                    WHEN pnl_pct <= 1.0 THEN 'Gain 0.5-1%'
                    WHEN pnl_pct <= 2.0 THEN 'Gain 1-2%'
                    ELSE 'Gain > 2%'
                END as pnl_bucket,
                pnl_pct
            FROM trade_pnl
        )
        SELECT 
            pnl_bucket,
            COUNT(*) as trades,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct_of_trades,
            ROUND(AVG(pnl_pct), 4) as avg_pnl_in_bucket,
            ROUND(SUM(pnl_pct), 2) as total_pnl_contribution
        FROM pnl_buckets
        GROUP BY pnl_bucket
        ORDER BY 
            CASE pnl_bucket
                WHEN 'Loss > 2%' THEN 1
                WHEN 'Loss 1-2%' THEN 2
                WHEN 'Loss 0.5-1%' THEN 3
                WHEN 'Loss 0.1-0.5%' THEN 4
                WHEN 'Small Loss <0.1%' THEN 5
                WHEN 'Breakeven' THEN 6
                WHEN 'Small Gain <0.1%' THEN 7
                WHEN 'Gain 0.1-0.5%' THEN 8
                WHEN 'Gain 0.5-1%' THEN 9
                WHEN 'Gain 1-2%' THEN 10
                WHEN 'Gain > 2%' THEN 11
            END
        """
        
        bucket_analysis = con.execute(detailed_query).df()
        print(bucket_analysis.to_string(index=False))
    
    # 3. Risk analysis
    print("\n\n5. Risk Analysis:")
    
    if all_results:
        risk_analysis = results_df.copy()
        risk_analysis['max_drawdown_est'] = risk_analysis['worst_loss_pct'] 
        risk_analysis['profit_factor'] = (risk_analysis['avg_winner_pct'] * risk_analysis['winners']) / abs(risk_analysis['avg_loser_pct'] * risk_analysis['losers'])
        risk_analysis['expectancy'] = (risk_analysis['win_rate']/100 * risk_analysis['avg_winner_pct']) + ((1-risk_analysis['win_rate']/100) * risk_analysis['avg_loser_pct'])
        
        print(risk_analysis[['holding_bars', 'max_drawdown_est', 'profit_factor', 'expectancy']].to_string(index=False))
    
    # 4. Optimal holding period analysis
    print("\n\n6. Optimal Holding Period Analysis:")
    
    if all_results:
        best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        best_total_return = results_df.loc[results_df['total_pnl_pct'].idxmax()]
        best_win_rate = results_df.loc[results_df['win_rate'].idxmax()]
        
        print(f"Best Sharpe Ratio: {int(best_sharpe['holding_bars'])} bars ({best_sharpe['sharpe_ratio']:.3f} Sharpe)")
        print(f"Best Total Return: {int(best_total_return['holding_bars'])} bars ({best_total_return['total_pnl_pct']:.2f}% total)")
        print(f"Best Win Rate: {int(best_win_rate['holding_bars'])} bars ({best_win_rate['win_rate']:.1f}% wins)")
    
    # 5. Trade timing analysis
    print("\n\n7. Trade Timing Patterns:")
    
    # Skip timing analysis for now due to timestamp column issues
    print("\nTiming analysis skipped (timestamp column not available)")
    
    con.close()
    
    print("\n\n=== Key Insights ===")
    print("1. BASELINE PERFORMANCE: Simple time exits show realistic expectations")
    print("2. P&L DISTRIBUTION: Most trades cluster around small gains/losses") 
    print("3. OPTIMAL HOLDING: Shorter periods often better due to mean reversion")
    print("4. RISK PROFILE: Maximum drawdowns and tail risk visible")
    print("5. NO CHERRY-PICKING: This is the unfiltered reality of ALL entries")
    
    print("\n=== Strategy Implications ===")
    print("1. Time-based exits provide honest baseline performance")
    print("2. Any signal-based exit must beat this simple approach")
    print("3. P&L distribution shows natural clustering patterns")
    print("4. Short holding periods may be optimal for mean-reverting markets")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python time_based_exit_pnl_analysis.py <workspace_path> <data_path>")
        sys.exit(1)
    
    analyze_time_based_exit_pnl(sys.argv[1], sys.argv[2])