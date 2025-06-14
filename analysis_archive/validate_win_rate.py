#!/usr/bin/env python3
"""
Validate the win rate calculation and investigate potential issues.
"""
import duckdb
import pandas as pd
import numpy as np


def validate_win_rate(workspace_path: str, data_path: str):
    """Investigate the suspiciously high win rate."""
    
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print("=== Win Rate Validation Analysis ===\n")
    
    # 1. Check the original signal generation
    print("1. Original RSI Strategy Signal Distribution:")
    
    signal_dist_query = f"""
    SELECT 
        strat,
        val,
        COUNT(*) as signals
    FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
    WHERE strat LIKE '%_7_%' OR strat LIKE '%_14_%'
    GROUP BY strat, val
    ORDER BY strat, val
    LIMIT 20
    """
    
    signal_dist = con.execute(signal_dist_query).df()
    print(signal_dist.to_string(index=False))
    
    # 2. Check what RSI values actually trigger signals
    print("\n\n2. RSI Values at Signal Generation:")
    
    # This requires joining with the actual price data to see RSI values
    # For now, let's check the trade timing
    
    # 3. Examine trade timing and market conditions
    print("\n3. Trade Timing Analysis:")
    
    timing_query = f"""
    WITH composite_trades AS (
        SELECT 
            e.idx as entry_idx,
            MIN(x.idx) as exit_idx,
            e.strat as entry_strat,
            FIRST(x.strat) as exit_strat
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet') e
        JOIN read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet') x
            ON x.idx > e.idx 
            AND x.idx <= e.idx + 20
            AND e.val = 1 
            AND x.val = -1
            AND e.strat LIKE '%_7_%'
            AND (x.strat LIKE '%_14_%' OR x.strat LIKE '%_21_%')
        GROUP BY e.idx, e.strat
    ),
    trade_analysis AS (
        SELECT 
            t.*,
            m_entry.close as entry_price,
            m_exit.close as exit_price,
            m_entry.ts as entry_time,
            m_exit.ts as exit_time,
            (m_exit.close - m_entry.close) / m_entry.close * 100 as return_pct
        FROM composite_trades t
        JOIN read_parquet('{data_path}') m_entry ON t.entry_idx = m_entry.bar_index
        JOIN read_parquet('{data_path}') m_exit ON t.exit_idx = m_exit.bar_index
    )
    SELECT 
        entry_strat,
        exit_strat,
        COUNT(*) as trades,
        ROUND(AVG(return_pct), 4) as avg_return,
        ROUND(MIN(return_pct), 4) as min_return,
        ROUND(MAX(return_pct), 4) as max_return,
        COUNT(CASE WHEN return_pct > 0 THEN 1 END) as winners,
        COUNT(CASE WHEN return_pct < 0 THEN 1 END) as losers
    FROM trade_analysis
    GROUP BY entry_strat, exit_strat
    ORDER BY trades DESC
    LIMIT 10
    """
    
    timing_analysis = con.execute(timing_query).df()
    print(timing_analysis.to_string(index=False))
    
    # 4. Check for data snooping bias
    print("\n\n4. Data Snooping Investigation:")
    
    # Check if we're only seeing successful combinations
    total_combinations_query = f"""
    WITH all_entry_signals AS (
        SELECT COUNT(*) as total_entries
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
        WHERE strat LIKE '%_7_%' AND val = 1
    ),
    matched_entries AS (
        SELECT COUNT(DISTINCT e.idx) as matched_entries
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet') e
        JOIN read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet') x
            ON x.idx > e.idx 
            AND x.idx <= e.idx + 20
            AND e.val = 1 
            AND x.val = -1
            AND e.strat LIKE '%_7_%'
            AND (x.strat LIKE '%_14_%' OR x.strat LIKE '%_21_%')
    )
    SELECT 
        t.total_entries,
        m.matched_entries,
        ROUND(m.matched_entries * 100.0 / t.total_entries, 2) as match_rate_pct
    FROM all_entry_signals t
    CROSS JOIN matched_entries m
    """
    
    snooping = con.execute(total_combinations_query).df()
    print("Entry Signal Match Rate:")
    print(snooping.to_string(index=False))
    
    # 5. Test with random entries
    print("\n\n5. Random Entry Baseline Test:")
    
    # Test what happens with random entries at the same times
    random_test_query = f"""
    WITH composite_trades AS (
        SELECT 
            e.idx as entry_idx,
            MIN(x.idx) as exit_idx
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
    random_entries AS (
        -- Take every 50th bar as "random" entry
        SELECT bar_index as entry_idx
        FROM read_parquet('{data_path}')
        WHERE bar_index % 50 = 0
            AND bar_index >= (SELECT MIN(entry_idx) FROM composite_trades)
            AND bar_index <= (SELECT MAX(entry_idx) FROM composite_trades)
    ),
    random_trades AS (
        SELECT 
            r.entry_idx,
            r.entry_idx + 13 as exit_idx  -- Use average holding period
        FROM random_entries r
        LIMIT 100  -- Sample size
    )
    SELECT 
        'Random Baseline' as strategy,
        COUNT(*) as trades,
        ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
        COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) as winners,
        ROUND(COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate_pct
    FROM random_trades t
    JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
    JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
    """
    
    random_baseline = con.execute(random_test_query).df()
    print(random_baseline.to_string(index=False))
    
    # 6. Check if this is during a bull market period
    print("\n\n6. Market Trend During Trading Period:")
    
    market_trend_query = f"""
    WITH trade_period AS (
        SELECT 
            MIN(entry_idx) as start_idx,
            MAX(exit_idx) as end_idx
        FROM (
            SELECT 
                e.idx as entry_idx,
                MIN(x.idx) as exit_idx
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
    )
    SELECT 
        m1.close as start_price,
        m2.close as end_price,
        ROUND((m2.close - m1.close) / m1.close * 100, 2) as total_market_move,
        tp.end_idx - tp.start_idx as total_bars
    FROM trade_period tp
    JOIN read_parquet('{data_path}') m1 ON tp.start_idx = m1.bar_index
    JOIN read_parquet('{data_path}') m2 ON tp.end_idx = m2.bar_index
    """
    
    market_trend = con.execute(market_trend_query).df()
    print("Overall Market Performance During Strategy Period:")
    print(market_trend.to_string(index=False))
    
    # 7. Sample some actual trades to inspect
    print("\n\n7. Sample Trade Inspection:")
    
    sample_trades_query = f"""
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
        t.entry_idx,
        t.exit_idx,
        t.holding_period,
        m1.close as entry_price,
        m2.close as exit_price,
        ROUND((m2.close - m1.close) / m1.close * 100, 4) as return_pct,
        m1.ts as entry_time,
        m2.ts as exit_time
    FROM composite_trades t
    JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
    JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
    ORDER BY return_pct
    LIMIT 20
    """
    
    sample_trades = con.execute(sample_trades_query).df()
    print("Sample Trades (sorted by return):")
    print(sample_trades.to_string(index=False))
    
    con.close()
    
    print("\n\n=== Potential Issues ===")
    print("1. SURVIVORSHIP BIAS: Only testing signal combinations that exist")
    print("2. DATA SNOOPING: Cherry-picking successful entry/exit pairs")
    print("3. BULL MARKET BIAS: Testing during favorable market conditions")
    print("4. SIGNAL CONSTRUCTION: RSI signals may be engineered for success")
    print("5. LOOK-AHEAD BIAS: Future information might be leaking into signals")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python validate_win_rate.py <workspace_path> <data_path>")
        sys.exit(1)
    
    validate_win_rate(sys.argv[1], sys.argv[2])