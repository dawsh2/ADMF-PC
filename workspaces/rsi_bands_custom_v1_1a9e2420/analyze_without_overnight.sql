-- Analyze RSI strategies excluding overnight trades
PRAGMA memory_limit='3GB';
SET threads=4;

ATTACH '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_c1ab337f/analytics.duckdb' AS analytics;

-- First, let's identify the worst trades and their timing
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
        m.close as price
    FROM signals s
    JOIN analytics.market_data m ON s.timestamp = m.timestamp
    WHERE s.prev_signal IS NOT NULL 
      AND s.signal_value != s.prev_signal
),
trades AS (
    SELECT 
        sc1.strategy_id,
        sc1.timestamp as entry_time,
        sc1.price as entry_price,
        sc2.timestamp as exit_time,
        sc2.price as exit_price,
        sc1.signal_value as position,
        CASE 
            WHEN sc1.signal_value = 1 THEN (sc2.price / sc1.price - 1) * 10000
            WHEN sc1.signal_value = -1 THEN (sc1.price / sc2.price - 1) * 10000
        END as return_bps,
        -- Check if trade spans overnight
        CASE 
            WHEN DATE(sc1.timestamp) != DATE(sc2.timestamp) THEN 'OVERNIGHT'
            WHEN EXTRACT(hour FROM sc1.timestamp) >= 16 AND EXTRACT(hour FROM sc2.timestamp) < 9 THEN 'OVERNIGHT'
            ELSE 'INTRADAY'
        END as trade_type
    FROM signal_changes sc1
    JOIN signal_changes sc2 ON sc1.strategy_id = sc2.strategy_id
        AND sc2.timestamp > sc1.timestamp
        AND sc1.signal_value != 0
        AND sc2.signal_value = 0
        AND sc1.prev_signal = 0
        AND sc2.prev_signal = sc1.signal_value
    WHERE NOT EXISTS (
        SELECT 1 FROM signal_changes sc3
        WHERE sc3.strategy_id = sc1.strategy_id
          AND sc3.timestamp > sc1.timestamp
          AND sc3.timestamp < sc2.timestamp
          AND sc3.signal_value = 0
          AND sc3.prev_signal = sc1.signal_value
    )
)
-- Show worst trades with timing
SELECT 
    strategy_id,
    entry_time,
    exit_time,
    ROUND(return_bps, 2) as return_bps,
    trade_type,
    EXTRACT(hour FROM entry_time) as entry_hour,
    EXTRACT(hour FROM exit_time) as exit_hour
FROM trades
WHERE return_bps < -50
ORDER BY return_bps;

-- Now calculate performance excluding overnight trades
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
        m.close as price
    FROM signals s
    JOIN analytics.market_data m ON s.timestamp = m.timestamp
    WHERE s.prev_signal IS NOT NULL 
      AND s.signal_value != s.prev_signal
),
trades AS (
    SELECT 
        sc1.strategy_id,
        sc1.timestamp as entry_time,
        sc1.price as entry_price,
        sc2.timestamp as exit_time,
        sc2.price as exit_price,
        sc1.signal_value as position,
        CASE 
            WHEN sc1.signal_value = 1 THEN (sc2.price / sc1.price - 1) * 10000
            WHEN sc1.signal_value = -1 THEN (sc1.price / sc2.price - 1) * 10000
        END as return_bps,
        CASE 
            WHEN DATE(sc1.timestamp) != DATE(sc2.timestamp) THEN 'OVERNIGHT'
            WHEN EXTRACT(hour FROM sc1.timestamp) >= 16 AND EXTRACT(hour FROM sc2.timestamp) < 9 THEN 'OVERNIGHT'
            ELSE 'INTRADAY'
        END as trade_type
    FROM signal_changes sc1
    JOIN signal_changes sc2 ON sc1.strategy_id = sc2.strategy_id
        AND sc2.timestamp > sc1.timestamp
        AND sc1.signal_value != 0
        AND sc2.signal_value = 0
        AND sc1.prev_signal = 0
        AND sc2.prev_signal = sc1.signal_value
    WHERE NOT EXISTS (
        SELECT 1 FROM signal_changes sc3
        WHERE sc3.strategy_id = sc1.strategy_id
          AND sc3.timestamp > sc1.timestamp
          AND sc3.timestamp < sc2.timestamp
          AND sc3.signal_value = 0
          AND sc3.prev_signal = sc1.signal_value
    )
)
-- Compare results with and without overnight trades
SELECT 
    strategy_id,
    '=== ALL TRADES ===' as category,
    COUNT(*) as num_trades,
    ROUND(AVG(return_bps), 2) as avg_gross_return,
    ROUND(AVG(return_bps - 0.5), 2) as avg_net_return,
    ROUND(STDDEV(return_bps), 2) as volatility,
    ROUND(MIN(return_bps), 2) as worst_trade,
    ROUND(MAX(return_bps), 2) as best_trade,
    ROUND(COUNT(CASE WHEN return_bps > 0 THEN 1 END) * 100.0 / COUNT(*), 1) as win_rate
FROM trades
GROUP BY strategy_id

UNION ALL

SELECT 
    strategy_id,
    '=== INTRADAY ONLY ===' as category,
    COUNT(*) as num_trades,
    ROUND(AVG(return_bps), 2) as avg_gross_return,
    ROUND(AVG(return_bps - 0.5), 2) as avg_net_return,
    ROUND(STDDEV(return_bps), 2) as volatility,
    ROUND(MIN(return_bps), 2) as worst_trade,
    ROUND(MAX(return_bps), 2) as best_trade,
    ROUND(COUNT(CASE WHEN return_bps > 0 THEN 1 END) * 100.0 / COUNT(*), 1) as win_rate
FROM trades
WHERE trade_type = 'INTRADAY'
GROUP BY strategy_id
ORDER BY strategy_id, category;

-- Calculate Sharpe ratios for intraday-only trades
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
        m.close as price
    FROM signals s
    JOIN analytics.market_data m ON s.timestamp = m.timestamp
    WHERE s.prev_signal IS NOT NULL 
      AND s.signal_value != s.prev_signal
),
trades AS (
    SELECT 
        sc1.strategy_id,
        sc1.timestamp as entry_time,
        sc2.timestamp as exit_time,
        EXTRACT(EPOCH FROM (sc2.timestamp - sc1.timestamp)) / 60 as duration_minutes,
        CASE 
            WHEN sc1.signal_value = 1 THEN (sc2.price / sc1.price - 1) * 10000
            WHEN sc1.signal_value = -1 THEN (sc1.price / sc2.price - 1) * 10000
        END as return_bps,
        CASE 
            WHEN DATE(sc1.timestamp) != DATE(sc2.timestamp) THEN 'OVERNIGHT'
            ELSE 'INTRADAY'
        END as trade_type
    FROM signal_changes sc1
    JOIN signal_changes sc2 ON sc1.strategy_id = sc2.strategy_id
        AND sc2.timestamp > sc1.timestamp
        AND sc1.signal_value != 0
        AND sc2.signal_value = 0
        AND sc1.prev_signal = 0
        AND sc2.prev_signal = sc1.signal_value
    WHERE NOT EXISTS (
        SELECT 1 FROM signal_changes sc3
        WHERE sc3.strategy_id = sc1.strategy_id
          AND sc3.timestamp > sc1.timestamp
          AND sc3.timestamp < sc2.timestamp
          AND sc3.signal_value = 0
          AND sc3.prev_signal = sc1.signal_value
    )
)
SELECT 
    strategy_id,
    COUNT(*) as intraday_trades,
    ROUND(AVG(return_bps - 0.5), 2) as net_return_bps,
    ROUND(AVG(return_bps - 0.5) / NULLIF(STDDEV(return_bps), 0) * SQRT(252 * 390 / NULLIF(AVG(duration_minutes), 1)), 2) as net_sharpe,
    CASE 
        WHEN AVG(return_bps - 0.5) > 0 THEN 'PROFITABLE'
        ELSE 'UNPROFITABLE'
    END as status
FROM trades
WHERE trade_type = 'INTRADAY'
GROUP BY strategy_id;