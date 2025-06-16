-- Analyze training period performance with duration limits
PRAGMA memory_limit='3GB';
SET threads=4;

ATTACH '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_c1ab337f/analytics.duckdb' AS analytics;

-- Training period analysis (before Jan 17, 2025)
WITH signals AS (
    SELECT 
        strat,
        ts::timestamp as timestamp,
        val as signal_value,
        LAG(val) OVER (PARTITION BY strat ORDER BY ts) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/rsi_bands_custom_v1_1a9e2420/traces/SPY_1m/signals/rsi_bands_grid/*.parquet')
    WHERE ts < '2025-01-17'
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
-- Training period performance with 30-minute duration limit
SELECT 
    '=== TRAINING PERIOD (Mar 26, 2024 - Jan 16, 2025) ===' as period,
    strategy_id,
    -- 30-minute limit performance
    COUNT(CASE WHEN trade_type = 'INTRADAY' AND duration_minutes <= 30 THEN 1 END) as trades_30min_limit,
    COUNT(DISTINCT DATE(entry_time)) as trading_days,
    ROUND(COUNT(CASE WHEN trade_type = 'INTRADAY' AND duration_minutes <= 30 THEN 1 END)::DOUBLE / COUNT(DISTINCT DATE(entry_time)), 2) as trades_per_day_30min,
    ROUND(AVG(CASE WHEN trade_type = 'INTRADAY' AND duration_minutes <= 30 THEN return_bps - 0.5 END), 2) as net_return_bps_30min,
    ROUND(AVG(CASE WHEN trade_type = 'INTRADAY' AND duration_minutes <= 30 THEN return_bps - 0.5 END) / 
          NULLIF(STDDEV(CASE WHEN trade_type = 'INTRADAY' AND duration_minutes <= 30 THEN return_bps END), 0) * 
          SQRT(252 * 390 / 30), 2) as net_sharpe_30min,
    -- All intraday trades performance  
    COUNT(CASE WHEN trade_type = 'INTRADAY' THEN 1 END) as all_intraday_trades,
    ROUND(AVG(CASE WHEN trade_type = 'INTRADAY' THEN return_bps - 0.5 END), 2) as net_return_bps_all,
    ROUND(AVG(CASE WHEN trade_type = 'INTRADAY' THEN return_bps - 0.5 END) / 
          NULLIF(STDDEV(CASE WHEN trade_type = 'INTRADAY' THEN return_bps END), 0) * 
          SQRT(252 * 390 / NULLIF(AVG(CASE WHEN trade_type = 'INTRADAY' THEN duration_minutes END), 1)), 2) as net_sharpe_all
FROM trades
GROUP BY strategy_id
ORDER BY strategy_id;

-- Duration distribution in training period
WITH signals AS (
    SELECT 
        strat,
        ts::timestamp as timestamp,
        val as signal_value,
        LAG(val) OVER (PARTITION BY strat ORDER BY ts) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/rsi_bands_custom_v1_1a9e2420/traces/SPY_1m/signals/rsi_bands_grid/*.parquet')
    WHERE ts < '2025-01-17'
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
    '=== TRAINING PERIOD DURATION BREAKDOWN ===' as header,
    strategy_id,
    CASE 
        WHEN duration_minutes <= 30 THEN '0-30 min'
        WHEN duration_minutes <= 60 THEN '30-60 min'
        WHEN duration_minutes <= 120 THEN '60-120 min'
        ELSE '120+ min'
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
        ELSE 4
    END;