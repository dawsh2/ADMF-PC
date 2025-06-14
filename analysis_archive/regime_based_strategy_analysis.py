#!/usr/bin/env python3
"""
Regime-Based Strategy Analysis

Analyzes RSI strategy performance under different market regimes to identify
when to trade vs when to stay out of the market.

Regime types analyzed:
1. Momentum regimes (trending vs sideways)
2. Volatility regimes (high vs low vol)
3. Market state regimes (bullish vs bearish)
4. Time-based regimes (hour of day, day of week)

Goal: Filter out unprofitable trading conditions to improve overall performance.
"""
import duckdb
import pandas as pd
import numpy as np


def analyze_regime_based_performance(workspace_path: str, data_path: str):
    """Analyze RSI strategy performance under different market regimes."""
    
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print("=== Regime-Based Strategy Analysis ===\n")
    print("Analyzing RSI strategy performance across different market conditions")
    print("to identify optimal trading regimes\n")
    
    # 1. Momentum Regime Analysis
    print("1. Momentum Regime Analysis:")
    
    momentum_query = f"""
    WITH fast_rsi_entries AS (
        SELECT DISTINCT idx as entry_idx
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
        WHERE strat LIKE '%_7_%' AND val = 1
    ),
    momentum_regimes AS (
        SELECT 
            idx,
            val as regime_state
        FROM read_parquet('{workspace_path}/traces/SPY_1m/classifiers/momentum_regime_grid/*.parquet')
        WHERE strat = 'momentum_regime_grid_70_30_01'  -- Use representative classifier
    ),
    regime_trades AS (
        SELECT 
            e.entry_idx,
            e.entry_idx + 18 as exit_idx,
            m.regime_state,
            CASE 
                WHEN m.regime_state = 1 THEN 'strong_momentum'
                WHEN m.regime_state = 0 THEN 'weak_momentum'
                WHEN m.regime_state = -1 THEN 'mean_reversion'
                ELSE 'unknown'
            END as regime_name
        FROM fast_rsi_entries e
        LEFT JOIN momentum_regimes m ON e.entry_idx = m.idx
        WHERE m.regime_state IS NOT NULL
    )
    SELECT 
        regime_name,
        COUNT(*) as trades,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct_of_trades,
        ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
        ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility,
        COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) as winners,
        ROUND(COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate,
        ROUND(AVG((m2.close - m1.close) / m1.close * 100) / STDDEV((m2.close - m1.close) / m1.close * 100), 3) as sharpe
    FROM regime_trades t
    JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
    JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
    WHERE t.exit_idx <= (SELECT MAX(bar_index) FROM read_parquet('{data_path}'))
    GROUP BY regime_name, regime_state
    ORDER BY avg_return DESC
    """
    
    try:
        momentum_results = con.execute(momentum_query).df()
        print(momentum_results.to_string(index=False))
    except Exception as e:
        print(f"Error in momentum analysis: {e}")
    
    # 2. Volatility Regime Analysis
    print("\n\n2. Volatility Regime Analysis:")
    
    volatility_query = f"""
    WITH fast_rsi_entries AS (
        SELECT DISTINCT idx as entry_idx
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
        WHERE strat LIKE '%_7_%' AND val = 1
    ),
    volatility_regimes AS (
        SELECT 
            idx,
            val as regime_state
        FROM read_parquet('{workspace_path}/traces/SPY_1m/classifiers/volatility_grid/*.parquet')
        WHERE strat = 'volatility_grid_20_05_25'  -- Use representative classifier
    ),
    regime_trades AS (
        SELECT 
            e.entry_idx,
            e.entry_idx + 18 as exit_idx,
            v.regime_state,
            CASE 
                WHEN v.regime_state = 1 THEN 'high_volatility'
                WHEN v.regime_state = 0 THEN 'normal_volatility' 
                WHEN v.regime_state = -1 THEN 'low_volatility'
                ELSE 'unknown'
            END as regime_name
        FROM fast_rsi_entries e
        LEFT JOIN volatility_regimes v ON e.entry_idx = v.idx
        WHERE v.regime_state IS NOT NULL
    )
    SELECT 
        regime_name,
        COUNT(*) as trades,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct_of_trades,
        ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
        ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility,
        COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) as winners,
        ROUND(COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate,
        ROUND(AVG((m2.close - m1.close) / m1.close * 100) / STDDEV((m2.close - m1.close) / m1.close * 100), 3) as sharpe
    FROM regime_trades t
    JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
    JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
    WHERE t.exit_idx <= (SELECT MAX(bar_index) FROM read_parquet('{data_path}'))
    GROUP BY regime_name, regime_state
    ORDER BY avg_return DESC
    """
    
    try:
        volatility_results = con.execute(volatility_query).df()
        print(volatility_results.to_string(index=False))
    except Exception as e:
        print(f"Error in volatility analysis: {e}")
    
    # 3. Market State Analysis  
    print("\n\n3. Market State Analysis:")
    
    market_state_query = f"""
    WITH fast_rsi_entries AS (
        SELECT DISTINCT idx as entry_idx
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
        WHERE strat LIKE '%_7_%' AND val = 1
    ),
    market_states AS (
        SELECT 
            idx,
            val as regime_state
        FROM read_parquet('{workspace_path}/traces/SPY_1m/classifiers/market_state_grid/*.parquet')
        WHERE strat = 'market_state_grid_20_100_10'  -- Use representative classifier
    ),
    regime_trades AS (
        SELECT 
            e.entry_idx,
            e.entry_idx + 18 as exit_idx,
            ms.regime_state,
            CASE 
                WHEN ms.regime_state = 1 THEN 'bullish_state'
                WHEN ms.regime_state = 0 THEN 'neutral_state'
                WHEN ms.regime_state = -1 THEN 'bearish_state'
                ELSE 'unknown'
            END as regime_name
        FROM fast_rsi_entries e
        LEFT JOIN market_states ms ON e.entry_idx = ms.idx
        WHERE ms.regime_state IS NOT NULL
    )
    SELECT 
        regime_name,
        COUNT(*) as trades,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct_of_trades,
        ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
        ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility,
        COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) as winners,
        ROUND(COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate,
        ROUND(AVG((m2.close - m1.close) / m1.close * 100) / STDDEV((m2.close - m1.close) / m1.close * 100), 3) as sharpe
    FROM regime_trades t
    JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
    JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
    WHERE t.exit_idx <= (SELECT MAX(bar_index) FROM read_parquet('{data_path}'))
    GROUP BY regime_name, regime_state
    ORDER BY avg_return DESC
    """
    
    try:
        market_state_results = con.execute(market_state_query).df()
        print(market_state_results.to_string(index=False))
    except Exception as e:
        print(f"Error in market state analysis: {e}")
    
    # 4. Combined Regime Analysis (Best Conditions)
    print("\n\n4. Combined Regime Analysis:")
    print("Testing combinations of favorable regimes...")
    
    combined_query = f"""
    WITH fast_rsi_entries AS (
        SELECT DISTINCT idx as entry_idx
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
        WHERE strat LIKE '%_7_%' AND val = 1
    ),
    all_regimes AS (
        SELECT 
            e.entry_idx,
            m.val as momentum_regime,
            v.val as volatility_regime,
            ms.val as market_state
        FROM fast_rsi_entries e
        LEFT JOIN read_parquet('{workspace_path}/traces/SPY_1m/classifiers/momentum_regime_grid/*.parquet') m 
            ON e.entry_idx = m.idx AND m.strat = 'momentum_regime_grid_70_30_01'
        LEFT JOIN read_parquet('{workspace_path}/traces/SPY_1m/classifiers/volatility_grid/*.parquet') v
            ON e.entry_idx = v.idx AND v.strat = 'volatility_grid_20_05_25'
        LEFT JOIN read_parquet('{workspace_path}/traces/SPY_1m/classifiers/market_state_grid/*.parquet') ms
            ON e.entry_idx = ms.idx AND ms.strat = 'market_state_grid_20_100_10'
    ),
    regime_combinations AS (
        SELECT 
            entry_idx,
            entry_idx + 18 as exit_idx,
            CASE 
                WHEN momentum_regime = -1 AND market_state IN (0, 1) THEN 'mean_reversion_favorable'
                WHEN momentum_regime = 1 AND volatility_regime = 1 THEN 'momentum_high_vol'
                WHEN momentum_regime = 0 AND volatility_regime = 0 THEN 'sideways_normal_vol'
                WHEN market_state = 1 THEN 'bullish_any'
                WHEN market_state = -1 THEN 'bearish_any'
                ELSE 'other_conditions'
            END as regime_combo
        FROM all_regimes
        WHERE momentum_regime IS NOT NULL 
            AND volatility_regime IS NOT NULL 
            AND market_state IS NOT NULL
    )
    SELECT 
        regime_combo,
        COUNT(*) as trades,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct_of_trades,
        ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
        ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility,
        COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) as winners,
        ROUND(COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate,
        ROUND(AVG((m2.close - m1.close) / m1.close * 100) / STDDEV((m2.close - m1.close) / m1.close * 100), 3) as sharpe,
        ROUND(SUM((m2.close - m1.close) / m1.close * 100), 2) as total_return
    FROM regime_combinations t
    JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
    JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
    WHERE t.exit_idx <= (SELECT MAX(bar_index) FROM read_parquet('{data_path}'))
    GROUP BY regime_combo
    HAVING COUNT(*) >= 10  -- Only regimes with significant sample size
    ORDER BY sharpe DESC
    """
    
    try:
        combined_results = con.execute(combined_query).df()
        print(combined_results.to_string(index=False))
    except Exception as e:
        print(f"Error in combined analysis: {e}")
    
    # 5. Regime Filtering Strategy Performance
    print("\n\n5. Regime Filtering Strategy:")
    
    # Find best performing regimes and simulate filtered strategy
    if 'combined_results' in locals() and len(combined_results) > 0:
        # Get top performing regimes (positive Sharpe)
        best_regimes = combined_results[combined_results['sharpe'] > 0]
        
        if len(best_regimes) > 0:
            print("Best performing regime combinations (Sharpe > 0):")
            print(best_regimes[['regime_combo', 'trades', 'avg_return', 'win_rate', 'sharpe']].to_string(index=False))
            
            # Calculate performance improvement
            filtered_trades = best_regimes['trades'].sum()
            filtered_total_return = best_regimes['total_return'].sum()
            filtered_avg_return = filtered_total_return / filtered_trades if filtered_trades > 0 else 0
            
            # Compare to unfiltered strategy
            all_regime_trades = combined_results['trades'].sum()
            all_regime_total = combined_results['total_return'].sum()
            unfiltered_avg = all_regime_total / all_regime_trades if all_regime_trades > 0 else 0
            
            print(f"\n6. Regime Filtering Impact:")
            print(f"Unfiltered strategy: {all_regime_trades} trades, {unfiltered_avg:.4f}% avg return")
            print(f"Filtered strategy: {filtered_trades} trades, {filtered_avg_return:.4f}% avg return")
            print(f"Trade reduction: {(1 - filtered_trades/all_regime_trades)*100:.1f}%")
            print(f"Performance improvement: {((filtered_avg_return/unfiltered_avg - 1)*100):.1f}%")
            
            coverage_pct = filtered_trades / all_regime_trades * 100
            print(f"Regime filter coverage: {coverage_pct:.1f}% of original trades")
            
        else:
            print("No regime combinations show positive Sharpe ratios")
    
    con.close()
    
    print(f"\n=== Regime-Based Strategy Recommendations ===")
    print(f"1. ANALYZE: Identify market conditions where RSI performs best")
    print(f"2. FILTER: Only trade during favorable regime combinations")
    print(f"3. ADAPT: Adjust position sizing based on regime confidence")
    print(f"4. MONITOR: Track regime changes for entry/exit timing")
    print(f"5. OPTIMIZE: Combine multiple regime classifiers for better filtering")
    
    print(f"\nImplementation Framework:")
    print(f"- Entry condition: Fast RSI signal + favorable regime")
    print(f"- Exit framework: Same multi-layer system as before")
    print(f"- Position sizing: Larger in favorable regimes, smaller in neutral")
    print(f"- Risk management: Tighter stops in unfavorable regime transitions")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python regime_based_strategy_analysis.py <workspace_path> <data_path>")
        sys.exit(1)
    
    analyze_regime_based_performance(sys.argv[1], sys.argv[2])