-- Analyze impact of limiting trade duration on RSI strategy profitability
PRAGMA memory_limit='3GB';
SET threads=4;

ATTACH '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_c1ab337f/analytics.duckdb' AS analytics;

-- Analyze trades with different duration limits
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
      AND s.signal_value <> s.prev_signal
),
trades AS (
    SELECT 
        sc1.strategy_id,
        sc1.timestamp as entry_time,
        sc1.price as entry_price,
        sc2.timestamp as exit_time,
        sc2.price as exit_price,
        sc1.signal_value as position,
        EXTRACT(EPOCH FROM (sc2.timestamp - sc1.timestamp)) / 60 as duration_minutes,
        CASE 
            WHEN sc1.signal_value = 1 THEN (sc2.price / sc1.price - 1) * 10000
            WHEN sc1.signal_value = -1 THEN (sc1.price / sc2.price - 1) * 10000
        END as return_bps,
        -- Check if trade spans overnight
        CASE 
            WHEN DATE(sc1.timestamp) <> DATE(sc2.timestamp) THEN 'OVERNIGHT'
            WHEN EXTRACT(hour FROM sc1.timestamp) >= 16 AND EXTRACT(hour FROM sc2.timestamp) < 9 THEN 'OVERNIGHT'
            ELSE 'INTRADAY'
        END as trade_type
    FROM signal_changes sc1
    JOIN signal_changes sc2 ON sc1.strategy_id = sc2.strategy_id
        AND sc2.timestamp > sc1.timestamp
        AND sc1.signal_value <> 0
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
),
-- Analyze performance at different duration thresholds
duration_analysis AS (
    SELECT 
        strategy_id,
        duration_limit,
        COUNT(*) as num_trades,
        ROUND(AVG(return_bps), 2) as avg_gross_return,
        ROUND(AVG(return_bps - 0.5), 2) as avg_net_return,
        ROUND(STDDEV(return_bps), 2) as volatility,
        ROUND(MIN(return_bps), 2) as worst_trade,
        ROUND(MAX(return_bps), 2) as best_trade,
        ROUND(COUNT(CASE WHEN return_bps > 0.5 THEN 1 END) * 100.0 / COUNT(*), 1) as profitable_pct,
        ROUND(AVG(return_bps - 0.5) / NULLIF(STDDEV(return_bps), 0) * SQRT(252 * 390 / NULLIF(AVG(duration_minutes), 1)), 2) as net_sharpe
    FROM (
        SELECT t.*, d.duration_limit
        FROM trades t
        CROSS JOIN (VALUES (30), (60), (120), (240), (390), (9999)) AS d(duration_limit)
        WHERE t.trade_type = 'INTRADAY'  -- Exclude overnight trades
          AND t.duration_minutes <= d.duration_limit
    ) filtered_trades
    GROUP BY strategy_id, duration_limit
)
-- Show results for different duration limits
SELECT 
    strategy_id,
    duration_limit as max_minutes,
    num_trades,
    avg_net_return as net_bps,
    net_sharpe,
    profitable_pct,
    CASE 
        WHEN avg_net_return > 0 THEN 'PROFITABLE'
        ELSE 'UNPROFITABLE'
    END as status
FROM duration_analysis
ORDER BY strategy_id, duration_limit;

-- Detailed analysis for the most promising duration limit
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
      AND s.signal_value <> s.prev_signal
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
            WHEN DATE(sc1.timestamp) <> DATE(sc2.timestamp) THEN 'OVERNIGHT'
            ELSE 'INTRADAY'
        END as trade_type
    FROM signal_changes sc1
    JOIN signal_changes sc2 ON sc1.strategy_id = sc2.strategy_id
        AND sc2.timestamp > sc1.timestamp
        AND sc1.signal_value <> 0
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
-- Show distribution of trade durations
SELECT 
    strategy_id,
    '=== TRADE DURATION DISTRIBUTION ===' as category,
    COUNT(CASE WHEN duration_minutes <= 30 THEN 1 END) as trades_under_30min,
    COUNT(CASE WHEN duration_minutes > 30 AND duration_minutes <= 60 THEN 1 END) as trades_30_60min,
    COUNT(CASE WHEN duration_minutes > 60 AND duration_minutes <= 120 THEN 1 END) as trades_60_120min,
    COUNT(CASE WHEN duration_minutes > 120 AND duration_minutes <= 240 THEN 1 END) as trades_120_240min,
    COUNT(CASE WHEN duration_minutes > 240 THEN 1 END) as trades_over_240min,
    COUNT(*) as total_intraday_trades
FROM trades
WHERE trade_type = 'INTRADAY'
GROUP BY strategy_id;

-- Analyze returns by duration bucket
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
      AND s.signal_value <> s.prev_signal
),
trades AS (
    SELECT 
        sc1.strategy_id,
        EXTRACT(EPOCH FROM (sc2.timestamp - sc1.timestamp)) / 60 as duration_minutes,
        CASE 
            WHEN sc1.signal_value = 1 THEN (sc2.price / sc1.price - 1) * 10000
            WHEN sc1.signal_value = -1 THEN (sc1.price / sc2.price - 1) * 10000
        END as return_bps,
        CASE 
            WHEN DATE(sc1.timestamp) <> DATE(sc2.timestamp) THEN 'OVERNIGHT'
            ELSE 'INTRADAY'
        END as trade_type
    FROM signal_changes sc1
    JOIN signal_changes sc2 ON sc1.strategy_id = sc2.strategy_id
        AND sc2.timestamp > sc1.timestamp
        AND sc1.signal_value <> 0
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
    CASE 
        WHEN duration_minutes <= 30 THEN '0-30 min'
        WHEN duration_minutes <= 60 THEN '30-60 min'
        WHEN duration_minutes <= 120 THEN '60-120 min'
        WHEN duration_minutes <= 240 THEN '120-240 min'
        ELSE '240+ min'
    END as duration_bucket,
    COUNT(*) as num_trades,
    ROUND(AVG(return_bps), 2) as avg_gross_bps,
    ROUND(AVG(return_bps - 0.5), 2) as avg_net_bps,
    ROUND(MIN(return_bps), 2) as worst_trade,
    ROUND(MAX(return_bps), 2) as best_trade
FROM trades
WHERE trade_type = 'INTRADAY'
GROUP BY strategy_id, duration_bucket
ORDER BY strategy_id, 
    CASE duration_bucket
        WHEN '0-30 min' THEN 1
        WHEN '30-60 min' THEN 2
        WHEN '60-120 min' THEN 3
        WHEN '120-240 min' THEN 4
        ELSE 5
    END;