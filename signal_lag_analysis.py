#!/usr/bin/env python3
"""
Signal Lag Analysis for RSI Composite Strategy

Tests the impact of lagging entry and/or exit signals by 1-3 bars
to see if confirmation improves performance or if immediate execution is optimal.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path


def analyze_signal_lag(workspace_path: str, data_path: str):
    """Analyze impact of lagging entry/exit signals."""
    
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print("=== Signal Lag Analysis ===\n")
    
    # Base case: No lag (original analysis)
    print("1. Baseline Performance (No Lag):")
    
    baseline_query = f"""
    WITH composite_trades AS (
        SELECT 
            e.idx as entry_idx,
            MIN(x.idx) as exit_idx,
            MIN(x.idx) - e.idx as holding_period
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet') e
        JOIN read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet') x
            ON x.idx > e.idx 
            AND x.idx <= e.idx + 20
            AND e.val = 1 
            AND x.val = -1
            AND e.strat LIKE '%_7_%'
            AND (x.strat LIKE '%_14_%' OR x.strat LIKE '%_21_%')
        GROUP BY e.idx
    )
    SELECT 
        'No Lag' as scenario,
        COUNT(*) as trades,
        ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
        ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility,
        ROUND(AVG(holding_period), 1) as avg_holding,
        ROUND(COUNT(*) * AVG((m2.close - m1.close) / m1.close * 100), 2) as total_return
    FROM composite_trades t
    JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
    JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
    """
    
    baseline = con.execute(baseline_query).df()
    print(baseline.to_string(index=False))
    
    # Test different lag scenarios
    lag_scenarios = [
        ("Entry +1 bar", 1, 0),
        ("Entry +2 bars", 2, 0), 
        ("Entry +3 bars", 3, 0),
        ("Exit +1 bar", 0, 1),
        ("Exit +2 bars", 0, 2),
        ("Exit +3 bars", 0, 3),
        ("Both +1 bar", 1, 1),
        ("Both +2 bars", 2, 2),
        ("Entry +1, Exit +2", 1, 2),
        ("Entry +2, Exit +1", 2, 1)
    ]
    
    print("\n\n2. Lag Scenario Analysis:")
    
    results = []
    
    for scenario_name, entry_lag, exit_lag in lag_scenarios:
        lag_query = f"""
        WITH composite_trades AS (
            SELECT 
                e.idx + {entry_lag} as entry_idx,  -- Lag entry execution
                MIN(x.idx + {exit_lag}) as exit_idx,  -- Lag exit execution
                MIN(x.idx + {exit_lag}) - (e.idx + {entry_lag}) as holding_period
            FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet') e
            JOIN read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet') x
                ON x.idx > e.idx 
                AND x.idx <= e.idx + 20
                AND e.val = 1 
                AND x.val = -1
                AND e.strat LIKE '%_7_%'
                AND (x.strat LIKE '%_14_%' OR x.strat LIKE '%_21_%')
            GROUP BY e.idx
        ),
        valid_trades AS (
            SELECT * FROM composite_trades 
            WHERE holding_period > 0  -- Ensure positive holding period
        )
        SELECT 
            '{scenario_name}' as scenario,
            COUNT(*) as trades,
            ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
            ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility,
            ROUND(AVG(holding_period), 1) as avg_holding,
            ROUND(COUNT(*) * AVG((m2.close - m1.close) / m1.close * 100), 2) as total_return
        FROM valid_trades t
        JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
        JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
        """
        
        try:
            result = con.execute(lag_query).df()
            if len(result) > 0 and result.iloc[0]['trades'] > 0:
                results.append(result.iloc[0])
        except Exception as e:
            print(f"Error in {scenario_name}: {e}")
    
    # Combine all results
    all_results = pd.concat([baseline] + [pd.DataFrame([r]) for r in results], ignore_index=True)
    print(all_results.to_string(index=False))
    
    # 3. Signal confirmation analysis
    print("\n\n3. Signal Confirmation Analysis:")
    print("Testing if waiting for signal confirmation improves quality...")
    
    confirmation_query = f"""
    WITH signal_persistence AS (
        -- Check if RSI signals persist for multiple bars
        SELECT 
            strat,
            idx,
            val,
            LEAD(val, 1) OVER (PARTITION BY strat ORDER BY idx) as next_val_1,
            LEAD(val, 2) OVER (PARTITION BY strat ORDER BY idx) as next_val_2,
            LEAD(val, 3) OVER (PARTITION BY strat ORDER BY idx) as next_val_3
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
        WHERE strat LIKE '%_7_%' AND val = 1  -- Entry signals only
    ),
    confirmed_signals AS (
        SELECT 
            idx,
            strat,
            CASE 
                WHEN next_val_1 = 1 THEN 'confirmed_1bar'
                WHEN next_val_2 = 1 THEN 'confirmed_2bar' 
                WHEN next_val_3 = 1 THEN 'confirmed_3bar'
                ELSE 'not_confirmed'
            END as confirmation_type
        FROM signal_persistence
    )
    SELECT 
        confirmation_type,
        COUNT(*) as signals,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct
    FROM confirmed_signals
    GROUP BY confirmation_type
    ORDER BY signals DESC
    """
    
    confirmation = con.execute(confirmation_query).df()
    print(confirmation.to_string(index=False))
    
    # 4. Market microstructure analysis
    print("\n\n4. Market Microstructure Analysis:")
    print("Impact of execution delays on fill prices...")
    
    execution_delay_query = f"""
    WITH trade_analysis AS (
        SELECT 
            e.idx as signal_idx,
            -- Test different execution delays
            (m1.close - m0.close) / m0.close * 100 as immediate_fill,
            (m2.close - m0.close) / m0.close * 100 as one_bar_delay,
            (m3.close - m0.close) / m0.close * 100 as two_bar_delay,
            (m4.close - m0.close) / m0.close * 100 as three_bar_delay
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet') e
        JOIN read_parquet('{data_path}') m0 ON e.idx = m0.bar_index
        JOIN read_parquet('{data_path}') m1 ON e.idx + 1 = m1.bar_index  
        JOIN read_parquet('{data_path}') m2 ON e.idx + 2 = m2.bar_index
        JOIN read_parquet('{data_path}') m3 ON e.idx + 3 = m3.bar_index
        JOIN read_parquet('{data_path}') m4 ON e.idx + 4 = m4.bar_index
        WHERE e.strat LIKE '%_7_%' AND e.val = 1
        LIMIT 100  -- Sample for analysis
    )
    SELECT 
        'Immediate' as execution_timing,
        ROUND(AVG(immediate_fill), 4) as avg_fill_impact,
        ROUND(STDDEV(immediate_fill), 4) as fill_volatility,
        COUNT(*) as samples
    FROM trade_analysis
    
    UNION ALL
    
    SELECT 
        '1 Bar Delay' as execution_timing,
        ROUND(AVG(one_bar_delay), 4) as avg_fill_impact,
        ROUND(STDDEV(one_bar_delay), 4) as fill_volatility,
        COUNT(*) as samples
    FROM trade_analysis
    
    UNION ALL
    
    SELECT 
        '2 Bar Delay' as execution_timing,
        ROUND(AVG(two_bar_delay), 4) as avg_fill_impact,
        ROUND(STDDEV(two_bar_delay), 4) as fill_volatility,
        COUNT(*) as samples
    FROM trade_analysis
    
    UNION ALL
    
    SELECT 
        '3 Bar Delay' as execution_timing,
        ROUND(AVG(three_bar_delay), 4) as avg_fill_impact,
        ROUND(STDDEV(three_bar_delay), 4) as fill_volatility,
        COUNT(*) as samples
    FROM trade_analysis
    """
    
    execution = con.execute(execution_delay_query).df()
    print(execution.to_string(index=False))
    
    # 5. Optimal lag recommendation
    print("\n\n5. Performance Ranking by Scenario:")
    
    # Calculate performance metrics
    all_results['sharpe'] = all_results['avg_return'] / all_results['volatility']
    all_results['return_per_trade'] = all_results['avg_return']
    all_results['total_profit'] = all_results['total_return']
    
    # Rank by different metrics
    ranking = all_results.copy()
    ranking['sharpe_rank'] = ranking['sharpe'].rank(ascending=False)
    ranking['return_rank'] = ranking['return_per_trade'].rank(ascending=False)
    ranking['total_rank'] = ranking['total_profit'].rank(ascending=False)
    ranking['avg_rank'] = (ranking['sharpe_rank'] + ranking['return_rank'] + ranking['total_rank']) / 3
    
    best_scenarios = ranking.sort_values('avg_rank')[['scenario', 'trades', 'avg_return', 'sharpe', 'total_return', 'avg_rank']].head(5)
    print(best_scenarios.to_string(index=False))
    
    con.close()
    
    print("\n\n=== Recommendations ===")
    best_scenario = ranking.sort_values('avg_rank').iloc[0]
    print(f"Best overall scenario: {best_scenario['scenario']}")
    print(f"- Average return: {best_scenario['avg_return']:.4f}%")
    print(f"- Sharpe ratio: {best_scenario['sharpe']:.3f}")
    print(f"- Total return: {best_scenario['total_return']:.2f}%")
    print(f"- Number of trades: {int(best_scenario['trades'])}")
    
    print("\nConsiderations:")
    print("1. Entry lag: Reduces false signals but may miss early moves")
    print("2. Exit lag: Allows trend continuation but risks giving back profits")
    print("3. Execution realism: 1-bar delay may be more realistic for live trading")
    print("4. Signal confirmation: Check if waiting improves signal quality")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python signal_lag_analysis.py <workspace_path> <data_path>")
        sys.exit(1)
    
    analyze_signal_lag(sys.argv[1], sys.argv[2])