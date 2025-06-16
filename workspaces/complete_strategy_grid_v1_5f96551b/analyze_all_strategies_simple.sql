-- Simplified strategy performance by regime analysis
-- Analyze all strategies across full training set

PRAGMA memory_limit='4GB';
SET threads=8;

SELECT 
    '=== COMPREHENSIVE STRATEGY ANALYSIS BY REGIME ===' as header;

-- First, let's get a count of strategies
SELECT 
    COUNT(DISTINCT strategy_id) as total_strategies,
    COUNT(DISTINCT strategy_type) as strategy_types
FROM analytics.strategies;

-- Analyze top strategy types
WITH strategy_counts AS (
    SELECT 
        strategy_type,
        COUNT(*) as strategy_count
    FROM analytics.strategies
    GROUP BY strategy_type
    ORDER BY strategy_count DESC
    LIMIT 5
)
SELECT * FROM strategy_counts;

-- Analyze MACD strategies as example
SELECT 
    '=== ANALYZING MACD STRATEGIES (SAMPLE) ===' as section_header;

-- Create temp tables for efficient processing
CREATE TEMP TABLE IF NOT EXISTS regime_timeline AS
WITH 
regime_sparse AS (
    SELECT 
        ts::timestamp as regime_time,
        val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet')
    WHERE ts::timestamp >= '2024-03-26 00:00:00'
      AND ts::timestamp <= '2025-01-17 20:00:00'
),
market_times AS (
    SELECT DISTINCT
        timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE timestamp >= '2024-03-26 00:00:00'::timestamp with time zone
      AND timestamp <= '2025-01-17 20:00:00'::timestamp with time zone
)
SELECT 
    mt.timestamp_est,
    LAST_VALUE(rs.regime_state IGNORE NULLS) OVER (
        ORDER BY mt.timestamp_est 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as current_regime
FROM market_times mt
LEFT JOIN regime_sparse rs ON mt.timestamp_est = rs.regime_time;

CREATE INDEX idx_regime_time ON regime_timeline(timestamp_est);

-- Create market prices table
CREATE TEMP TABLE IF NOT EXISTS market_prices AS
SELECT 
    timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est,
    close
FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
WHERE timestamp >= '2024-03-26 00:00:00'::timestamp with time zone
  AND timestamp <= '2025-01-17 20:00:00'::timestamp with time zone;

CREATE INDEX idx_price_time ON market_prices(timestamp_est);

-- Analyze first 5 MACD strategies
WITH macd_strategies AS (
    SELECT 
        strategy_id,
        strategy_name,
        signal_file_path
    FROM analytics.strategies
    WHERE strategy_type = 'macd_crossover'
    ORDER BY strategy_id
    LIMIT 5
),
strategy_performance AS (
    SELECT 
        s.strategy_id,
        s.strategy_name,
        rt.current_regime,
        COUNT(*) as trade_count,
        ROUND(AVG(
            CASE 
                WHEN sig.val = 1 AND sig_next.val = -1 THEN 
                    (mp2.close - mp1.close) / mp1.close
                WHEN sig.val = -1 AND sig_next.val = 1 THEN 
                    (mp1.close - mp2.close) / mp1.close
                ELSE NULL
            END
        ) * 100, 3) as avg_return_pct,
        ROUND(SUM(
            CASE 
                WHEN sig.val = 1 AND sig_next.val = -1 THEN 
                    (mp2.close - mp1.close) / mp1.close - 0.0005
                WHEN sig.val = -1 AND sig_next.val = 1 THEN 
                    (mp1.close - mp2.close) / mp1.close - 0.0005
                ELSE 0
            END
        ) * 100, 3) as net_return_pct
    FROM macd_strategies s
    CROSS JOIN LATERAL (
        SELECT 
            ts::timestamp as signal_time,
            val,
            LAG(val) OVER (ORDER BY ts) as prev_val,
            LEAD(ts::timestamp) OVER (ORDER BY ts) as next_time,
            LEAD(val) OVER (ORDER BY ts) as next_val
        FROM read_parquet(s.signal_file_path)
        WHERE ts::timestamp >= '2024-03-26 00:00:00'
          AND ts::timestamp <= '2025-01-17 20:00:00'
    ) sig
    INNER JOIN read_parquet(s.signal_file_path) sig_next 
        ON sig_next.ts::timestamp = sig.next_time
    LEFT JOIN regime_timeline rt ON sig.signal_time = rt.timestamp_est
    LEFT JOIN market_prices mp1 ON sig.signal_time = mp1.timestamp_est
    LEFT JOIN market_prices mp2 ON sig.next_time = mp2.timestamp_est
    WHERE sig.prev_val IS NOT NULL
      AND sig.prev_val != sig.val
      AND sig.next_time IS NOT NULL
      AND rt.current_regime IS NOT NULL
      AND mp1.close IS NOT NULL
      AND mp2.close IS NOT NULL
    GROUP BY s.strategy_id, s.strategy_name, rt.current_regime
)
SELECT 
    strategy_name,
    current_regime,
    trade_count,
    avg_return_pct,
    net_return_pct
FROM strategy_performance
WHERE trade_count >= 5
ORDER BY net_return_pct DESC
LIMIT 20;

-- Summary across all analyzed strategies
SELECT 
    '=== REGIME PERFORMANCE SUMMARY ===' as section_header;

SELECT 
    current_regime,
    COUNT(*) as minutes_in_regime,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct_of_time
FROM regime_timeline
WHERE current_regime IS NOT NULL
GROUP BY current_regime
ORDER BY pct_of_time DESC;

-- Clean up
DROP TABLE IF EXISTS regime_timeline;
DROP TABLE IF EXISTS market_prices;