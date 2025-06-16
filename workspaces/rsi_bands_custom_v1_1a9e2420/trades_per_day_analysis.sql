-- Calculate trades per day for both test and training periods
PRAGMA memory_limit='3GB';
SET threads=4;

ATTACH '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_c1ab337f/analytics.duckdb' AS analytics;

-- Test period trades per day (30-minute limit)
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
        EXTRACT(EPOCH FROM (sc2.timestamp - sc1.timestamp)) / 60 as duration_minutes,
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
    '=== TEST PERIOD (Jan 17 - Apr 2, 2025) ===' as period,
    strategy_id,
    COUNT(CASE WHEN trade_type = 'INTRADAY' AND duration_minutes <= 30 THEN 1 END) as trades_30min_limit,
    COUNT(DISTINCT DATE(entry_time)) as trading_days,
    ROUND(COUNT(CASE WHEN trade_type = 'INTRADAY' AND duration_minutes <= 30 THEN 1 END)::DOUBLE / COUNT(DISTINCT DATE(entry_time)), 2) as trades_per_day_30min
FROM trades
GROUP BY strategy_id
ORDER BY strategy_id;