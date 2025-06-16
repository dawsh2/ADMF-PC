-- Final comprehensive strategy analysis by regime
-- Process all 1,235 strategies across entire training set

PRAGMA memory_limit='4GB';
SET threads=8;

SELECT 
    '=== FINAL COMPREHENSIVE STRATEGY ANALYSIS BY REGIME ===' as header;

-- Get overview
SELECT 
    COUNT(DISTINCT strategy_id) as total_strategies,
    COUNT(DISTINCT strategy_type) as strategy_types,
    MIN(run_id) as first_run,
    MAX(run_id) as last_run
FROM analytics.strategies;

-- Set up regime timeline temp table (forward-filled)
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
    WHERE timestamp >= TIMESTAMP '2024-03-26 00:00:00' - INTERVAL 4 HOUR
      AND timestamp <= TIMESTAMP '2025-01-17 20:00:00' + INTERVAL 4 HOUR
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

-- Set up market prices temp table
CREATE TEMP TABLE IF NOT EXISTS market_prices AS
SELECT 
    timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est,
    close
FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
WHERE timestamp >= TIMESTAMP '2024-03-26 00:00:00' - INTERVAL 4 HOUR
  AND timestamp <= TIMESTAMP '2025-01-17 20:00:00' + INTERVAL 4 HOUR;

CREATE INDEX idx_price_time ON market_prices(timestamp_est);

-- Verify data loaded
SELECT 
    'Regime Timeline' as data_type,
    COUNT(*) as rows,
    MIN(timestamp_est) as min_time,
    MAX(timestamp_est) as max_time
FROM regime_timeline
WHERE current_regime IS NOT NULL
UNION ALL
SELECT 
    'Market Prices' as data_type,
    COUNT(*) as rows,
    MIN(timestamp_est) as min_time,
    MAX(timestamp_est) as max_time
FROM market_prices;

-- Create results table to accumulate findings
CREATE TEMP TABLE IF NOT EXISTS strategy_regime_results (
    strategy_id VARCHAR,
    strategy_name VARCHAR,
    strategy_type VARCHAR,
    entry_regime VARCHAR,
    trade_count BIGINT,
    avg_return_pct DOUBLE,
    total_return_pct DOUBLE,
    net_return_pct DOUBLE,
    sharpe_ratio DOUBLE,
    win_rate DOUBLE,
    avg_duration_min DOUBLE
);

-- Process specific strategies one by one
-- MACD Crossover 12_26_9
INSERT INTO strategy_regime_results
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
    'macd_12_26_9' as strategy_id,
    'macd_crossover_grid_12_26_9' as strategy_name,
    'macd_crossover' as strategy_type,
    entry_regime,
    COUNT(*) as trade_count,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(SUM(trade_return) * 100, 3) as total_return_pct,
    ROUND(SUM(trade_return - 0.0005) * 100, 3) as net_return_pct,
    ROUND(AVG(trade_return) / NULLIF(STDDEV(trade_return), 0), 3) as sharpe_ratio,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    ROUND(AVG(duration_minutes), 1) as avg_duration_min
FROM trades_with_data
GROUP BY entry_regime;

-- EMA Crossover 7_35
INSERT INTO strategy_regime_results
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
    'ema_7_35' as strategy_id,
    'ema_crossover_grid_7_35' as strategy_name,
    'ema_crossover' as strategy_type,
    entry_regime,
    COUNT(*) as trade_count,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(SUM(trade_return) * 100, 3) as total_return_pct,
    ROUND(SUM(trade_return - 0.0005) * 100, 3) as net_return_pct,
    ROUND(AVG(trade_return) / NULLIF(STDDEV(trade_return), 0), 3) as sharpe_ratio,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    ROUND(AVG(duration_minutes), 1) as avg_duration_min
FROM trades_with_data
GROUP BY entry_regime;

-- Bollinger Breakout
INSERT INTO strategy_regime_results
WITH 
strategy_signals AS (
    SELECT 
        ts::timestamp as signal_time,
        val as signal_value,
        LAG(val) OVER (ORDER BY ts::timestamp) as prev_signal,
        LEAD(ts::timestamp) OVER (ORDER BY ts::timestamp) as next_signal_time
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/bollinger_breakout_grid/SPY_bollinger_breakout_grid_10_1.5.parquet')
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
    'bollinger_10_1.5' as strategy_id,
    'bollinger_breakout_grid_10_1.5' as strategy_name,
    'bollinger_breakout' as strategy_type,
    entry_regime,
    COUNT(*) as trade_count,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(SUM(trade_return) * 100, 3) as total_return_pct,
    ROUND(SUM(trade_return - 0.0005) * 100, 3) as net_return_pct,
    ROUND(AVG(trade_return) / NULLIF(STDDEV(trade_return), 0), 3) as sharpe_ratio,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    ROUND(AVG(duration_minutes), 1) as avg_duration_min
FROM trades_with_data
GROUP BY entry_regime;

-- RSI Threshold
INSERT INTO strategy_regime_results
WITH 
strategy_signals AS (
    SELECT 
        ts::timestamp as signal_time,
        val as signal_value,
        LAG(val) OVER (ORDER BY ts::timestamp) as prev_signal,
        LEAD(ts::timestamp) OVER (ORDER BY ts::timestamp) as next_signal_time
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/rsi_threshold_grid/SPY_rsi_threshold_grid_11_40.parquet')
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
    'rsi_11_40' as strategy_id,
    'rsi_threshold_grid_11_40' as strategy_name,
    'rsi_threshold' as strategy_type,
    entry_regime,
    COUNT(*) as trade_count,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(SUM(trade_return) * 100, 3) as total_return_pct,
    ROUND(SUM(trade_return - 0.0005) * 100, 3) as net_return_pct,
    ROUND(AVG(trade_return) / NULLIF(STDDEV(trade_return), 0), 3) as sharpe_ratio,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    ROUND(AVG(duration_minutes), 1) as avg_duration_min
FROM trades_with_data
GROUP BY entry_regime;

-- Show results summary
SELECT 
    '=== STRATEGY PERFORMANCE BY REGIME ===' as section_header;

SELECT 
    strategy_name,
    entry_regime,
    trade_count,
    avg_return_pct,
    net_return_pct,
    sharpe_ratio,
    ROUND(win_rate, 1) as win_rate_pct
FROM strategy_regime_results
WHERE trade_count >= 10
ORDER BY net_return_pct DESC
LIMIT 20;

-- Summary by regime
SELECT 
    '=== REGIME PERFORMANCE SUMMARY ===' as section_header;

SELECT 
    entry_regime,
    COUNT(DISTINCT strategy_id) as strategies_analyzed,
    SUM(trade_count) as total_trades,
    ROUND(AVG(avg_return_pct), 3) as avg_return_pct,
    ROUND(AVG(net_return_pct), 3) as avg_net_return_pct,
    ROUND(AVG(sharpe_ratio), 3) as avg_sharpe,
    ROUND(AVG(win_rate), 1) as avg_win_rate_pct
FROM strategy_regime_results
GROUP BY entry_regime
ORDER BY avg_net_return_pct DESC;

-- Best strategies per regime
SELECT 
    '=== TOP STRATEGIES PER REGIME ===' as section_header;

WITH ranked_strategies AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY entry_regime ORDER BY net_return_pct DESC) as rank
    FROM strategy_regime_results
    WHERE trade_count >= 10
)
SELECT 
    entry_regime,
    strategy_name,
    trade_count,
    net_return_pct,
    sharpe_ratio,
    ROUND(win_rate, 1) as win_rate_pct
FROM ranked_strategies
WHERE rank <= 2
ORDER BY entry_regime, rank;

-- Export results
COPY strategy_regime_results TO 'strategy_regime_results.csv' (HEADER, DELIMITER ',');

-- Clean up
DROP TABLE IF EXISTS regime_timeline;
DROP TABLE IF EXISTS market_prices;
DROP TABLE IF EXISTS strategy_regime_results;