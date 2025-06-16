-- Analyze RSI strategies performance from Jan 17th onwards (test period)
PRAGMA memory_limit='3GB';
SET threads=4;

-- Use the market data from the analytics database
ATTACH '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_c1ab337f/analytics.duckdb' AS analytics;

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
    JOIN analytics.market_data m ON s.timestamp = m.timestamp
    WHERE s.prev_signal IS NOT NULL 
      AND s.signal_value != s.prev_signal
      AND s.prev_signal != 0
      AND m.timestamp >= '2025-01-17'
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
    MIN(timestamp)::date as first_trade_date,
    MAX(timestamp)::date as last_trade_date,
    ROUND(AVG(return_bps), 3) as gross_return_bps,
    ROUND(AVG(return_bps - 0.5), 3) as net_return_bps,
    ROUND(STDDEV(return_bps), 3) as vol_bps,
    ROUND(COUNT(CASE WHEN return_bps > 0 THEN 1 END) * 100.0 / COUNT(*), 1) as win_rate,
    -- Calculate Sharpe (annualized)
    ROUND(AVG(return_bps - 0.5) / NULLIF(STDDEV(return_bps), 0) * SQRT(252 * 390 / 60), 2) as net_sharpe,
    -- Trading days
    COUNT(DISTINCT DATE(timestamp)) as trading_days,
    ROUND(COUNT(*)::DOUBLE / COUNT(DISTINCT DATE(timestamp)), 2) as trades_per_day
FROM trades
GROUP BY strategy_id;

-- Show test period results
SELECT 
    '=== TEST PERIOD RESULTS (Jan 17 - Apr 2) ===' as header;
    
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

-- Compare with training period performance
SELECT 
    '=== TRAINING vs TEST COMPARISON ===' as header;

WITH training_performance AS (
    SELECT 
        strategy_id,
        net_return_bps,
        net_sharpe,
        win_rate
    FROM analytics.strategy_test_results
    WHERE strategy_id IN ('SPY_rsi_bands_grid_19_30_80', 'SPY_rsi_bands_grid_19_30_75')
)
SELECT 
    t.strategy_id,
    -- Training period (Mar 26 2024 - Jan 16 2025)
    p.net_return_bps as train_net_bps,
    p.net_sharpe as train_sharpe,
    p.win_rate as train_win_rate,
    -- Test period (Jan 17 - Apr 2 2025)
    t.net_return_bps as test_net_bps,
    t.net_sharpe as test_sharpe,
    t.win_rate as test_win_rate,
    -- Performance differences
    ROUND(t.net_return_bps - p.net_return_bps, 3) as net_bps_diff,
    ROUND(t.net_sharpe - p.net_sharpe, 2) as sharpe_diff,
    ROUND(t.win_rate - p.win_rate, 1) as win_rate_diff,
    -- Status
    CASE 
        WHEN t.net_return_bps > 0 AND p.net_return_bps > 0 THEN 'CONSISTENT_PROFIT'
        WHEN t.net_return_bps > 0 AND p.net_return_bps <= 0 THEN 'IMPROVED'
        WHEN t.net_return_bps <= 0 AND p.net_return_bps > 0 THEN 'DEGRADED'
        ELSE 'CONSISTENT_LOSS'
    END as consistency
FROM rsi_test_results t
LEFT JOIN training_performance p ON t.strategy_id = p.strategy_id;

-- Cumulative performance analysis
SELECT 
    '=== CUMULATIVE RETURNS ===' as header;

WITH daily_returns AS (
    SELECT 
        strategy_id,
        DATE(timestamp) as trade_date,
        SUM(CASE 
            WHEN prev_signal = 1 THEN (price / prev_price - 1) * 10000 - 0.5
            WHEN prev_signal = -1 THEN (prev_price / price - 1) * 10000 - 0.5
        END) as daily_net_bps
    FROM (
        SELECT 
            s.strat as strategy_id,
            s.timestamp,
            s.val as signal_value,
            LAG(s.val) OVER (PARTITION BY s.strat ORDER BY s.ts) as prev_signal,
            m.close as price,
            LAG(m.close) OVER (PARTITION BY s.strat ORDER BY s.timestamp) as prev_price
        FROM read_parquet('/Users/daws/ADMF-PC/workspaces/rsi_bands_custom_v1_1a9e2420/traces/SPY_1m/signals/rsi_bands_grid/*.parquet') s
        JOIN analytics.market_data m ON s.ts::timestamp = m.timestamp
        WHERE s.ts >= '2025-01-17'
    ) t
    WHERE prev_signal IS NOT NULL 
      AND signal_value != prev_signal
      AND prev_signal != 0
    GROUP BY strategy_id, trade_date
)
SELECT 
    strategy_id,
    COUNT(*) as trading_days,
    ROUND(SUM(daily_net_bps), 2) as total_net_bps,
    ROUND(AVG(daily_net_bps), 2) as avg_daily_bps,
    ROUND(MAX(daily_net_bps), 2) as best_day_bps,
    ROUND(MIN(daily_net_bps), 2) as worst_day_bps
FROM daily_returns
GROUP BY strategy_id;