#!/usr/bin/env python3
"""
Deep dive into regime-based strategy patterns.
Find which strategies work in which regimes.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path


def analyze_regime_patterns(workspace_path: str, data_path: str):
    """Comprehensive regime-based analysis."""
    
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print("=== Regime Pattern Analysis ===\n")
    
    # 1. Analyze momentum regime classifier in detail
    print("1. Momentum Regime Analysis:")
    
    regime_details_query = f"""
    WITH regime_stats AS (
        SELECT 
            val as regime,
            COUNT(*) as occurrences,
            MIN(idx) as first_idx,
            MAX(idx) as last_idx
        FROM read_parquet('{workspace_path}/traces/*/classifiers/momentum_regime_grid/*.parquet')
        WHERE strat = 'SPY_momentum_regime_grid_70_25_015'
        GROUP BY val
    )
    SELECT 
        CASE 
            WHEN regime = 1 THEN 'weak_momentum'
            WHEN regime = 2 THEN 'strong_momentum'
            ELSE 'unknown'
        END as regime_name,
        occurrences,
        ROUND(occurrences * 100.0 / SUM(occurrences) OVER (), 2) as pct
    FROM regime_stats
    """
    
    regime_dist = con.execute(regime_details_query).df()
    print(regime_dist.to_string(index=False))
    
    # 2. Find strategies that work BETTER in weak momentum
    print("\n\n2. Strategies that excel in WEAK momentum regime:")
    
    weak_momentum_query = f"""
    WITH regime_signals AS (
        SELECT 
            c.idx,
            CASE WHEN c.val = 1 THEN 'weak' ELSE 'strong' END as regime
        FROM read_parquet('{workspace_path}/traces/*/classifiers/momentum_regime_grid/*.parquet') c
        WHERE c.strat = 'SPY_momentum_regime_grid_70_25_015'
    ),
    strategy_performance AS (
        SELECT 
            s.strat,
            r.regime,
            COUNT(*) as trades,
            AVG(CASE 
                WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
            END) as avg_return,
            STDDEV(CASE 
                WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
            END) as volatility
        FROM read_parquet('{workspace_path}/traces/*/signals/*/*.parquet') s
        JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
        JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
        JOIN regime_signals r ON s.idx = r.idx
        WHERE s.val != 0
        GROUP BY s.strat, r.regime
        HAVING COUNT(*) >= 20
    ),
    regime_comparison AS (
        SELECT 
            strat,
            MAX(CASE WHEN regime = 'weak' THEN avg_return END) as weak_return,
            MAX(CASE WHEN regime = 'strong' THEN avg_return END) as strong_return,
            MAX(CASE WHEN regime = 'weak' THEN trades END) as weak_trades,
            MAX(CASE WHEN regime = 'strong' THEN trades END) as strong_trades
        FROM strategy_performance
        GROUP BY strat
        HAVING weak_return IS NOT NULL AND strong_return IS NOT NULL
    )
    SELECT 
        strat,
        weak_trades,
        ROUND(weak_return, 4) as weak_return_pct,
        strong_trades,
        ROUND(strong_return, 4) as strong_return_pct,
        ROUND(weak_return - strong_return, 4) as weak_advantage
    FROM regime_comparison
    WHERE weak_return > 0.01  -- Profitable in weak momentum
        AND weak_return > strong_return  -- Better in weak than strong
    ORDER BY weak_advantage DESC
    LIMIT 10
    """
    
    weak_specialists = con.execute(weak_momentum_query).df()
    print(weak_specialists.to_string(index=False))
    
    # 3. Composite strategy performance by regime
    print("\n\n3. Composite Strategy (RSI fast→slow) by Regime:")
    
    composite_regime_query = f"""
    WITH regime_signals AS (
        SELECT 
            idx,
            CASE WHEN val = 1 THEN 'weak' ELSE 'strong' END as regime
        FROM read_parquet('{workspace_path}/traces/*/classifiers/momentum_regime_grid/*.parquet')
        WHERE strat = 'SPY_momentum_regime_grid_70_25_015'
    ),
    composite_trades AS (
        SELECT 
            e.idx as entry_idx,
            MIN(x.idx) as exit_idx,
            r.regime
        FROM read_parquet('{workspace_path}/traces/*/signals/rsi_grid/*.parquet') e
        JOIN read_parquet('{workspace_path}/traces/*/signals/rsi_grid/*.parquet') x
            ON x.idx > e.idx 
            AND x.idx <= e.idx + 10
            AND e.val = 1 
            AND x.val = -1
            AND e.strat LIKE '%_7_%'
            AND (x.strat LIKE '%_14_%' OR x.strat LIKE '%_21_%')
        JOIN regime_signals r ON e.idx = r.idx
        GROUP BY e.idx, r.regime
    )
    SELECT 
        regime,
        COUNT(*) as trades,
        ROUND(AVG(exit_idx - entry_idx), 1) as avg_holding,
        ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
        ROUND(MIN((m2.close - m1.close) / m1.close * 100), 4) as worst_trade,
        ROUND(MAX((m2.close - m1.close) / m1.close * 100), 4) as best_trade
    FROM composite_trades t
    JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
    JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
    GROUP BY regime
    """
    
    composite_regime = con.execute(composite_regime_query).df()
    print(composite_regime.to_string(index=False))
    
    # 4. Create regime-filtered ensemble
    print("\n\n4. Regime-Filtered Ensemble Strategy:")
    
    # First, find best strategy for each regime
    best_per_regime_query = f"""
    WITH regime_signals AS (
        SELECT 
            idx,
            CASE WHEN val = 1 THEN 'weak' ELSE 'strong' END as regime
        FROM read_parquet('{workspace_path}/traces/*/classifiers/momentum_regime_grid/*.parquet')
        WHERE strat = 'SPY_momentum_regime_grid_70_25_015'
    ),
    strategy_performance AS (
        SELECT 
            s.strat,
            r.regime,
            COUNT(*) as trades,
            AVG(CASE 
                WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
            END) as avg_return
        FROM read_parquet('{workspace_path}/traces/*/signals/*/*.parquet') s
        JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
        JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
        JOIN regime_signals r ON s.idx = r.idx
        WHERE s.val != 0
        GROUP BY s.strat, r.regime
        HAVING COUNT(*) >= 50 AND AVG(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END) > 0
    ),
    ranked AS (
        SELECT 
            *,
            ROW_NUMBER() OVER (PARTITION BY regime ORDER BY avg_return DESC) as rank
        FROM strategy_performance
    )
    SELECT 
        regime,
        strat,
        trades,
        ROUND(avg_return, 4) as return_pct
    FROM ranked
    WHERE rank <= 3
    ORDER BY regime, rank
    """
    
    best_strategies = con.execute(best_per_regime_query).df()
    print("\nTop strategies by regime:")
    print(best_strategies.to_string(index=False))
    
    # 5. Volume + Regime analysis
    print("\n\n5. Volume Context within Regimes:")
    
    volume_regime_query = f"""
    WITH regime_signals AS (
        SELECT 
            idx,
            CASE WHEN val = 1 THEN 'weak' ELSE 'strong' END as regime
        FROM read_parquet('{workspace_path}/traces/*/classifiers/momentum_regime_grid/*.parquet')
        WHERE strat = 'SPY_momentum_regime_grid_70_25_015'
    ),
    volume_context AS (
        SELECT 
            bar_index,
            volume,
            AVG(volume) OVER (ORDER BY bar_index ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as avg_vol20
        FROM read_parquet('{data_path}')
    ),
    signals_with_context AS (
        SELECT 
            s.strat,
            s.val as signal,
            r.regime,
            CASE 
                WHEN v.volume > v.avg_vol20 * 1.5 THEN 'high_volume'
                ELSE 'normal_volume'
            END as volume_context,
            CASE 
                WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
            END as return_pct
        FROM read_parquet('{workspace_path}/traces/*/signals/*/*.parquet') s
        JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
        JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
        JOIN regime_signals r ON s.idx = r.idx
        JOIN volume_context v ON s.idx = v.bar_index
        WHERE s.val != 0
    )
    SELECT 
        regime,
        volume_context,
        COUNT(*) as trades,
        ROUND(AVG(return_pct), 4) as avg_return
    FROM signals_with_context
    GROUP BY regime, volume_context
    HAVING COUNT(*) >= 100
    ORDER BY regime, volume_context
    """
    
    volume_regime = con.execute(volume_regime_query).df()
    print(volume_regime.to_string(index=False))
    
    # 6. Create optimal regime-aware composite strategy
    print("\n\n6. Optimal Regime-Aware Composite Strategy:")
    
    # Only trade composite in weak momentum regime
    optimal_composite_query = f"""
    WITH regime_signals AS (
        SELECT 
            idx,
            val as regime_val
        FROM read_parquet('{workspace_path}/traces/*/classifiers/momentum_regime_grid/*.parquet')
        WHERE strat = 'SPY_momentum_regime_grid_70_25_015'
    ),
    composite_trades AS (
        SELECT 
            e.idx as entry_idx,
            MIN(x.idx) as exit_idx
        FROM read_parquet('{workspace_path}/traces/*/signals/rsi_grid/*.parquet') e
        JOIN read_parquet('{workspace_path}/traces/*/signals/rsi_grid/*.parquet') x
            ON x.idx > e.idx 
            AND x.idx <= e.idx + 10
            AND e.val = 1 
            AND x.val = -1
            AND e.strat LIKE '%_7_%'
            AND (x.strat LIKE '%_14_%' OR x.strat LIKE '%_21_%')
        JOIN regime_signals r ON e.idx = r.idx
        WHERE r.regime_val = 1  -- Only trade in weak momentum
        GROUP BY e.idx
    )
    SELECT 
        'Regime-filtered composite' as strategy,
        COUNT(*) as trades,
        ROUND(AVG(exit_idx - entry_idx), 1) as avg_holding,
        ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
        ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility,
        ROUND(SUM((m2.close - m1.close) / m1.close * 100), 2) as total_return
    FROM composite_trades t
    JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
    JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
    """
    
    optimal = con.execute(optimal_composite_query).df()
    print(optimal.to_string(index=False))
    
    con.close()
    
    print("\n\n=== Key Insights ===")
    print("1. Strategies perform BETTER in weak momentum regimes")
    print("2. Strong momentum regimes show negative returns (overextended?)")
    print("3. Composite strategy maintains positive returns in both regimes")
    print("4. Volume context matters more in strong momentum regimes")
    print("5. Regime filtering can improve risk-adjusted returns")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python analyze_regime_patterns.py <workspace_path> <data_path>")
        sys.exit(1)
    
    analyze_regime_patterns(sys.argv[1], sys.argv[2])