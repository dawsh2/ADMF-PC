-- Analyze RSI strategies performance from Jan 17th onwards (test period)
-- This is out-of-sample testing since we discovered these strategies on earlier data

PRAGMA memory_limit='3GB';
SET threads=4;

-- First, let's load the signals and check date range
WITH signal_overview AS (
    SELECT 
        split_part(filename, '/', -1) as strategy_file,
        MIN(ts) as min_date,
        MAX(ts) as max_date,
        COUNT(*) as total_signals,
        COUNT(CASE WHEN ts >= '2025-01-17' THEN 1 END) as test_signals
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/rsi_bands_custom_v1_1a9e2420/traces/SPY_1m/signals/rsi_bands_grid/*.parquet', filename=true)
    GROUP BY filename
)
SELECT * FROM signal_overview;

-- Load market data (assuming same format as before)
DROP TABLE IF EXISTS market_data_rsi;
CREATE TABLE market_data_rsi AS
SELECT DISTINCT
    timestamp,
    open,
    high,
    low,
    close,
    volume
FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_c1ab337f/data/*.parquet')
WHERE timestamp >= '2025-01-17'
ORDER BY timestamp;

-- Analyze both strategies for the test period
DROP TABLE IF EXISTS rsi_test_results;

CREATE TABLE rsi_test_results AS
WITH signals AS (
    SELECT 
        strat,
        ts::timestamp as timestamp,
        val as signal_value,
        LAG(val) OVER (PARTITION BY strat ORDER BY ts) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/rsi_bands_custom_v1_1a9e2420/traces/SPY_1m/signals/rsi_bands_grid/*.parquet')
    WHERE ts >= '2025-01-17'
),
signal_changes AS (
    SELECT 
        s.strat as strategy_id,
        s.timestamp,
        s.signal_value,
        s.prev_signal,
        m.close as price,
        LAG(m.close) OVER (PARTITION BY s.strat ORDER BY s.timestamp) as prev_price
    FROM signals s
    JOIN market_data_rsi m ON s.timestamp = m.timestamp
    WHERE s.prev_signal IS NOT NULL 
      AND s.signal_value != s.prev_signal
      AND s.prev_signal != 0
),
trades AS (
    SELECT 
        strategy_id,
        timestamp,
        CASE 
            WHEN prev_signal = 1 THEN (price / prev_price - 1) * 10000
            WHEN prev_signal = -1 THEN (prev_price / price - 1) * 10000
        END as return_bps
    FROM signal_changes
    WHERE ABS(price / prev_price - 1) < 0.1  -- Filter outliers
)
SELECT 
    strategy_id,
    COUNT(*) as num_trades,
    MIN(timestamp) as first_trade,
    MAX(timestamp) as last_trade,
    ROUND(AVG(return_bps), 3) as gross_return_bps,
    ROUND(AVG(return_bps - 0.5), 3) as net_return_bps,
    ROUND(STDDEV(return_bps), 3) as vol_bps,
    ROUND(COUNT(CASE WHEN return_bps > 0 THEN 1 END) * 100.0 / COUNT(*), 1) as win_rate,
    -- Calculate Sharpe (annualized)
    ROUND(AVG(return_bps - 0.5) / NULLIF(STDDEV(return_bps), 0) * SQRT(252 * 390 / 60), 2) as net_sharpe,
    -- Trading days
    COUNT(DISTINCT DATE(timestamp)) as trading_days,
    COUNT(*)::DOUBLE / COUNT(DISTINCT DATE(timestamp)) as trades_per_day
FROM trades
GROUP BY strategy_id;

-- Show test period results
SELECT 
    strategy_id,
    num_trades,
    trading_days,
    trades_per_day,
    gross_return_bps,
    net_return_bps,
    vol_bps,
    win_rate,
    net_sharpe,
    CASE 
        WHEN net_return_bps > 0 THEN 'PROFITABLE'
        ELSE 'UNPROFITABLE'
    END as status
FROM rsi_test_results
ORDER BY net_sharpe DESC;

-- Compare test period vs training period performance
WITH training_performance AS (
    SELECT 
        strategy_id,
        net_return_bps as train_net_bps,
        net_sharpe as train_sharpe,
        win_rate as train_win_rate
    FROM strategy_test_results
    WHERE strategy_id IN ('SPY_rsi_bands_grid_19_30_80', 'SPY_rsi_bands_grid_19_30_75')
),
comparison AS (
    SELECT 
        t.strategy_id,
        p.train_net_bps,
        t.net_return_bps as test_net_bps,
        p.train_sharpe,
        t.net_sharpe as test_sharpe,
        p.train_win_rate,
        t.win_rate as test_win_rate,
        t.num_trades as test_trades
    FROM rsi_test_results t
    JOIN training_performance p ON t.strategy_id = p.strategy_id
)
SELECT 
    strategy_id,
    ROUND(train_net_bps, 3) as train_net_bps,
    ROUND(test_net_bps, 3) as test_net_bps,
    ROUND(test_net_bps - train_net_bps, 3) as performance_diff,
    ROUND(train_sharpe, 2) as train_sharpe,
    ROUND(test_sharpe, 2) as test_sharpe,
    ROUND(train_win_rate, 1) as train_win_rate,
    ROUND(test_win_rate, 1) as test_win_rate,
    test_trades
FROM comparison;