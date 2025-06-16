-- Comprehensive Strategy Performance by Regime Analysis
-- Analyze all 1,235 strategies across entire training set (10 months)
-- Highly optimized batched approach

PRAGMA memory_limit='4GB';
SET threads=8;

SELECT 
    '=== COMPREHENSIVE STRATEGY PERFORMANCE BY REGIME ===' as header;

-- First, get overview of available strategies
SELECT 
    '=== STRATEGY UNIVERSE OVERVIEW ===' as section_header;

SELECT 
    strategy_type,
    COUNT(*) as strategy_count,
    COUNT(DISTINCT signal_file_path) as unique_files
FROM analytics.strategies
GROUP BY strategy_type
ORDER BY strategy_count DESC
LIMIT 10;

-- Create a sample analysis for top strategy types
-- We'll process in batches by strategy type to manage memory
SELECT 
    '=== BATCH 1: MACD CROSSOVER STRATEGIES ===' as section_header;

WITH 
-- Define analysis period
analysis_period AS (
    SELECT 
        '2024-03-26 00:00:00'::timestamp as start_time,
        '2025-01-17 20:00:00'::timestamp as end_time
),
-- Get MACD strategies (108 total)
macd_strategies AS (
    SELECT 
        strategy_id,
        strategy_name,
        signal_file_path
    FROM analytics.strategies
    WHERE strategy_type = 'macd_crossover'
    LIMIT 20  -- Sample first 20 for performance
),
-- Get regime timeline with forward-fill (using our best classifier)
regime_sparse AS (
    SELECT 
        c.ts::timestamp as regime_time,
        c.val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c,
         analysis_period ap
    WHERE c.ts::timestamp >= ap.start_time
      AND c.ts::timestamp <= ap.end_time
),
-- Get market data
market_data AS (
    SELECT 
        timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est,  -- Timezone adjustment
        close
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet'),
         analysis_period ap
    WHERE timestamp >= ap.start_time::timestamp with time zone - INTERVAL 4 HOUR
      AND timestamp <= ap.end_time::timestamp with time zone
),
-- Forward-fill regime to all timestamps
regime_timeline AS (
    SELECT 
        m.timestamp_est,
        LAST_VALUE(r.regime_state IGNORE NULLS) OVER (
            ORDER BY m.timestamp_est 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as current_regime
    FROM market_data m
    LEFT JOIN regime_sparse r ON m.timestamp_est = r.regime_time
),
-- Process each strategy and calculate performance
strategy_performance AS (
    SELECT 
        ms.strategy_id,
        ms.strategy_name,
        rt.current_regime as entry_regime,
        COUNT(*) as trade_count,
        AVG(
            CASE 
                WHEN s.val = 1 AND s_next.val = -1 THEN 
                    (m_exit.close - m_entry.close) / m_entry.close
                WHEN s.val = -1 AND s_next.val = 1 THEN 
                    (m_entry.close - m_exit.close) / m_entry.close
                ELSE NULL
            END
        ) as avg_return,
        SUM(
            CASE 
                WHEN s.val = 1 AND s_next.val = -1 THEN 
                    (m_exit.close - m_entry.close) / m_entry.close
                WHEN s.val = -1 AND s_next.val = 1 THEN 
                    (m_entry.close - m_exit.close) / m_entry.close
                ELSE 0
            END
        ) as total_return
    FROM macd_strategies ms,
         LATERAL (
            SELECT 
                ts::timestamp as signal_time,
                val as signal_value,
                LEAD(ts::timestamp) OVER (ORDER BY ts::timestamp) as next_time,
                LEAD(val) OVER (ORDER BY ts::timestamp) as next_value,
                LAG(val) OVER (ORDER BY ts::timestamp) as prev_value
            FROM read_parquet(ms.signal_file_path)
         ) s
    INNER JOIN read_parquet(ms.signal_file_path) s_next 
        ON s_next.ts::timestamp = s.next_time
    LEFT JOIN regime_timeline rt ON s.signal_time = rt.timestamp_est
    LEFT JOIN market_data m_entry ON s.signal_time = m_entry.timestamp_est
    LEFT JOIN market_data m_exit ON s.next_time = m_exit.timestamp_est
    WHERE s.prev_value IS NOT NULL
      AND s.prev_value != s.signal_value  -- Trade signals only
      AND s.next_time IS NOT NULL
      AND rt.current_regime IS NOT NULL
      AND m_entry.close IS NOT NULL
      AND m_exit.close IS NOT NULL
    GROUP BY ms.strategy_id, ms.strategy_name, rt.current_regime
)
SELECT 
    strategy_name,
    entry_regime,
    trade_count,
    ROUND(avg_return * 100, 3) as avg_return_pct,
    ROUND(total_return * 100, 3) as total_return_pct,
    ROUND((total_return - trade_count * 0.0005) * 100, 3) as net_return_pct  -- After costs
FROM strategy_performance
WHERE trade_count >= 10  -- Minimum trades for significance
ORDER BY net_return_pct DESC
LIMIT 20;

-- Summary by regime across all MACD strategies
SELECT 
    '=== MACD STRATEGIES: REGIME PERFORMANCE SUMMARY ===' as section_header;

-- This would be the full analysis query structure
-- For now, let's do a simplified version to test
WITH sample_signals AS (
    -- Get first 5 MACD strategies
    SELECT 
        s.strategy_id,
        s.strategy_name,
        sig.ts::timestamp as signal_time,
        sig.val as signal_value,
        LAG(sig.val) OVER (PARTITION BY s.strategy_id ORDER BY sig.ts::timestamp) as prev_signal
    FROM (
        SELECT strategy_id, strategy_name, signal_file_path 
        FROM analytics.strategies 
        WHERE strategy_type = 'macd_crossover' 
        LIMIT 5
    ) s,
    LATERAL read_parquet(s.signal_file_path) sig
    WHERE sig.ts::timestamp >= '2024-04-01 00:00:00'
      AND sig.ts::timestamp <= '2024-05-01 00:00:00'
),
trades AS (
    SELECT 
        strategy_id,
        strategy_name,
        signal_time,
        signal_value,
        LEAD(signal_time) OVER (PARTITION BY strategy_id ORDER BY signal_time) as exit_time
    FROM sample_signals
    WHERE prev_signal IS NOT NULL
      AND prev_signal != signal_value
),
regime_data AS (
    SELECT 
        c.ts::timestamp as regime_time,
        c.val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
    WHERE c.ts::timestamp >= '2024-04-01 00:00:00'
      AND c.ts::timestamp <= '2024-05-01 00:00:00'
),
trades_with_regime AS (
    SELECT 
        t.*,
        (SELECT r.regime_state 
         FROM regime_data r 
         WHERE r.regime_time <= t.signal_time
         ORDER BY r.regime_time DESC
         LIMIT 1) as entry_regime
    FROM trades t
)
SELECT 
    entry_regime,
    COUNT(DISTINCT strategy_id) as strategies_analyzed,
    COUNT(*) as total_trades,
    ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT strategy_id), 1) as avg_trades_per_strategy
FROM trades_with_regime
WHERE entry_regime IS NOT NULL
  AND exit_time IS NOT NULL
GROUP BY entry_regime
ORDER BY total_trades DESC;