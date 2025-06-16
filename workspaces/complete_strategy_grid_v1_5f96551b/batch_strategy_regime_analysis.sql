-- Batch Strategy Performance by Regime Analysis
-- Process strategies one at a time to avoid memory issues

PRAGMA memory_limit='4GB';
SET threads=8;

SELECT 
    '=== BATCH STRATEGY PERFORMANCE BY REGIME ===' as header;

-- Create temporary table for regime timeline (forward-filled)
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
    WHERE timestamp >= '2024-03-26 00:00:00-07'
      AND timestamp <= '2025-01-17 20:00:00-07'
)
SELECT 
    mt.timestamp_est,
    LAST_VALUE(rs.regime_state IGNORE NULLS) OVER (
        ORDER BY mt.timestamp_est 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as current_regime
FROM market_times mt
LEFT JOIN regime_sparse rs ON mt.timestamp_est = rs.regime_time;

-- Create index for performance
CREATE INDEX idx_regime_time ON regime_timeline(timestamp_est);

-- Create temp table for market prices
CREATE TEMP TABLE IF NOT EXISTS market_prices AS
SELECT 
    timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est,
    close
FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
WHERE timestamp >= '2024-03-26 00:00:00-07'
  AND timestamp <= '2025-01-17 20:00:00-07';

CREATE INDEX idx_price_time ON market_prices(timestamp_est);

-- Check data loaded
SELECT 
    '=== DATA LOADED ===' as section_header;

SELECT 
    'Regime Timeline' as data_type,
    COUNT(*) as row_count,
    MIN(timestamp_est) as min_time,
    MAX(timestamp_est) as max_time
FROM regime_timeline
UNION ALL
SELECT 
    'Market Prices' as data_type,
    COUNT(*) as row_count,
    MIN(timestamp_est) as min_time,
    MAX(timestamp_est) as max_time
FROM market_prices;

-- Analyze a sample of strategies
-- Strategy 1: MACD Crossover
SELECT 
    '=== STRATEGY 1: MACD CROSSOVER 12_26_9 ===' as section_header;

WITH 
strategy_signals AS (
    SELECT 
        ts::timestamp as signal_time,
        val as signal_value,
        LAG(val) OVER (ORDER BY ts::timestamp) as prev_signal,
        LEAD(ts::timestamp) OVER (ORDER BY ts::timestamp) as next_signal_time
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet')
    WHERE ts::timestamp >= '2024-03-26 00:00:00'
      AND ts::timestamp <= '2025-01-17 20:00:00'
),
trades AS (
    SELECT 
        signal_time,
        signal_value,
        next_signal_time
    FROM strategy_signals
    WHERE prev_signal IS NOT NULL
      AND prev_signal != signal_value
      AND next_signal_time IS NOT NULL
),
trades_with_data AS (
    SELECT 
        t.*,
        rt.current_regime as entry_regime,
        mp1.close as entry_price,
        mp2.close as exit_price,
        EXTRACT(EPOCH FROM (t.next_signal_time - t.signal_time)) / 60.0 as duration_minutes,
        CASE 
            WHEN t.signal_value = 1 THEN (mp2.close - mp1.close) / mp1.close
            WHEN t.signal_value = -1 THEN (mp1.close - mp2.close) / mp1.close
        END as trade_return
    FROM trades t
    LEFT JOIN regime_timeline rt ON t.signal_time = rt.timestamp_est
    LEFT JOIN market_prices mp1 ON t.signal_time = mp1.timestamp_est
    LEFT JOIN market_prices mp2 ON t.next_signal_time = mp2.timestamp_est
    WHERE rt.current_regime IS NOT NULL
      AND mp1.close IS NOT NULL
      AND mp2.close IS NOT NULL
)
SELECT 
    'MACD_12_26_9' as strategy_name,
    entry_regime,
    COUNT(*) as trade_count,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(SUM(trade_return) * 100, 3) as total_return_pct,
    ROUND(SUM(trade_return - 0.0005) * 100, 3) as net_return_pct,
    ROUND(AVG(trade_return) / NULLIF(STDDEV(trade_return), 0), 3) as sharpe_ratio,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    ROUND(AVG(duration_minutes), 1) as avg_duration_min
FROM trades_with_data
GROUP BY entry_regime
ORDER BY net_return_pct DESC;

-- Strategy 2: EMA Crossover
SELECT 
    '=== STRATEGY 2: EMA CROSSOVER 7_35 ===' as section_header;

WITH 
strategy_signals AS (
    SELECT 
        ts::timestamp as signal_time,
        val as signal_value,
        LAG(val) OVER (ORDER BY ts::timestamp) as prev_signal,
        LEAD(ts::timestamp) OVER (ORDER BY ts::timestamp) as next_signal_time
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/ema_crossover_grid/SPY_ema_crossover_grid_7_35.parquet')
    WHERE ts::timestamp >= '2024-03-26 00:00:00'
      AND ts::timestamp <= '2025-01-17 20:00:00'
),
trades AS (
    SELECT 
        signal_time,
        signal_value,
        next_signal_time
    FROM strategy_signals
    WHERE prev_signal IS NOT NULL
      AND prev_signal != signal_value
      AND next_signal_time IS NOT NULL
),
trades_with_data AS (
    SELECT 
        t.*,
        rt.current_regime as entry_regime,
        mp1.close as entry_price,
        mp2.close as exit_price,
        EXTRACT(EPOCH FROM (t.next_signal_time - t.signal_time)) / 60.0 as duration_minutes,
        CASE 
            WHEN t.signal_value = 1 THEN (mp2.close - mp1.close) / mp1.close
            WHEN t.signal_value = -1 THEN (mp1.close - mp2.close) / mp1.close
        END as trade_return
    FROM trades t
    LEFT JOIN regime_timeline rt ON t.signal_time = rt.timestamp_est
    LEFT JOIN market_prices mp1 ON t.signal_time = mp1.timestamp_est
    LEFT JOIN market_prices mp2 ON t.next_signal_time = mp2.timestamp_est
    WHERE rt.current_regime IS NOT NULL
      AND mp1.close IS NOT NULL
      AND mp2.close IS NOT NULL
)
SELECT 
    'EMA_7_35' as strategy_name,
    entry_regime,
    COUNT(*) as trade_count,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(SUM(trade_return) * 100, 3) as total_return_pct,
    ROUND(SUM(trade_return - 0.0005) * 100, 3) as net_return_pct,
    ROUND(AVG(trade_return) / NULLIF(STDDEV(trade_return), 0), 3) as sharpe_ratio,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    ROUND(AVG(duration_minutes), 1) as avg_duration_min
FROM trades_with_data
GROUP BY entry_regime
ORDER BY net_return_pct DESC;

-- Strategy 3: RSI Threshold
SELECT 
    '=== STRATEGY 3: RSI THRESHOLD 14_30_70 ===' as section_header;

WITH 
strategy_signals AS (
    SELECT 
        ts::timestamp as signal_time,
        val as signal_value,
        LAG(val) OVER (ORDER BY ts::timestamp) as prev_signal,
        LEAD(ts::timestamp) OVER (ORDER BY ts::timestamp) as next_signal_time
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/rsi_threshold_grid/SPY_rsi_threshold_grid_14_30_70.parquet')
    WHERE ts::timestamp >= '2024-03-26 00:00:00'
      AND ts::timestamp <= '2025-01-17 20:00:00'
),
trades AS (
    SELECT 
        signal_time,
        signal_value,
        next_signal_time
    FROM strategy_signals
    WHERE prev_signal IS NOT NULL
      AND prev_signal != signal_value
      AND next_signal_time IS NOT NULL
),
trades_with_data AS (
    SELECT 
        t.*,
        rt.current_regime as entry_regime,
        mp1.close as entry_price,
        mp2.close as exit_price,
        EXTRACT(EPOCH FROM (t.next_signal_time - t.signal_time)) / 60.0 as duration_minutes,
        CASE 
            WHEN t.signal_value = 1 THEN (mp2.close - mp1.close) / mp1.close
            WHEN t.signal_value = -1 THEN (mp1.close - mp2.close) / mp1.close
        END as trade_return
    FROM trades t
    LEFT JOIN regime_timeline rt ON t.signal_time = rt.timestamp_est
    LEFT JOIN market_prices mp1 ON t.signal_time = mp1.timestamp_est
    LEFT JOIN market_prices mp2 ON t.next_signal_time = mp2.timestamp_est
    WHERE rt.current_regime IS NOT NULL
      AND mp1.close IS NOT NULL
      AND mp2.close IS NOT NULL
)
SELECT 
    'RSI_14_30_70' as strategy_name,
    entry_regime,
    COUNT(*) as trade_count,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(SUM(trade_return) * 100, 3) as total_return_pct,
    ROUND(SUM(trade_return - 0.0005) * 100, 3) as net_return_pct,
    ROUND(AVG(trade_return) / NULLIF(STDDEV(trade_return), 0), 3) as sharpe_ratio,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    ROUND(AVG(duration_minutes), 1) as avg_duration_min
FROM trades_with_data
GROUP BY entry_regime
ORDER BY net_return_pct DESC;

-- Overall summary across sampled strategies
SELECT 
    '=== REGIME PERFORMANCE SUMMARY ===' as section_header;

-- Combined view would require UNION of all strategy results
-- For now, showing regime distribution
SELECT 
    current_regime,
    COUNT(*) as minutes_in_regime,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct_of_time,
    MIN(timestamp_est) as first_occurrence,
    MAX(timestamp_est) as last_occurrence
FROM regime_timeline
WHERE current_regime IS NOT NULL
GROUP BY current_regime
ORDER BY pct_of_time DESC;

-- Clean up temp tables
DROP TABLE IF EXISTS regime_timeline;
DROP TABLE IF EXISTS market_prices;