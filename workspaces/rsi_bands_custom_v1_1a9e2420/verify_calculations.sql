-- Verify RSI calculations with detailed breakdown
PRAGMA memory_limit='3GB';
SET threads=4;

ATTACH '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_c1ab337f/analytics.duckdb' AS analytics;

-- Calculate returns properly by matching signal changes
WITH signals AS (
    SELECT 
        strat,
        ts::timestamp as timestamp,
        val as signal_value,
        LAG(val) OVER (PARTITION BY strat ORDER BY ts) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/rsi_bands_custom_v1_1a9e2420/traces/SPY_1m/signals/rsi_bands_grid/SPY_rsi_bands_grid_19_30_80.parquet')
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
    -- Match entries with exits
    SELECT 
        sc1.strategy_id,
        sc1.timestamp as entry_time,
        sc1.price as entry_price,
        sc2.timestamp as exit_time,
        sc2.price as exit_price,
        sc1.signal_value as position,
        -- Calculate returns
        CASE 
            WHEN sc1.signal_value = 1 THEN (sc2.price / sc1.price - 1) * 10000
            WHEN sc1.signal_value = -1 THEN (sc1.price / sc2.price - 1) * 10000
        END as return_bps
    FROM signal_changes sc1
    JOIN signal_changes sc2 ON sc1.strategy_id = sc2.strategy_id
        AND sc2.timestamp > sc1.timestamp
        AND sc1.signal_value != 0  -- Entry
        AND sc2.signal_value = 0   -- Exit
        AND sc1.prev_signal = 0    -- Was neutral before entry
        AND sc2.prev_signal = sc1.signal_value  -- Exit from the position we entered
    WHERE NOT EXISTS (
        -- Make sure this is the next exit after entry
        SELECT 1 FROM signal_changes sc3
        WHERE sc3.strategy_id = sc1.strategy_id
          AND sc3.timestamp > sc1.timestamp
          AND sc3.timestamp < sc2.timestamp
          AND sc3.signal_value = 0
          AND sc3.prev_signal = sc1.signal_value
    )
)
-- Show sample of trades with details
SELECT 
    strategy_id,
    entry_time,
    entry_price,
    exit_time,
    exit_price,
    CASE WHEN position = 1 THEN 'LONG' ELSE 'SHORT' END as direction,
    ROUND(return_bps, 2) as return_bps,
    ROUND(return_bps - 0.5, 2) as net_return_bps
FROM trades
ORDER BY entry_time
LIMIT 20;

-- Calculate summary statistics
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
        END as return_bps
    FROM signal_changes sc1
    JOIN signal_changes sc2 ON sc1.strategy_id = sc2.strategy_id
        AND sc2.timestamp > sc1.timestamp
        AND sc1.signal_value != 0  -- Entry
        AND sc2.signal_value = 0   -- Exit
        AND sc1.prev_signal = 0    -- Was neutral before entry
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
    COUNT(*) as num_trades,
    ROUND(AVG(return_bps), 2) as avg_gross_return,
    ROUND(AVG(return_bps - 0.5), 2) as avg_net_return,
    ROUND(STDDEV(return_bps), 2) as volatility,
    ROUND(MIN(return_bps), 2) as worst_trade,
    ROUND(MAX(return_bps), 2) as best_trade,
    ROUND(COUNT(CASE WHEN return_bps > 0 THEN 1 END) * 100.0 / COUNT(*), 1) as win_rate,
    ROUND(AVG(CASE WHEN return_bps > 0 THEN return_bps END), 2) as avg_win,
    ROUND(AVG(CASE WHEN return_bps < 0 THEN return_bps END), 2) as avg_loss
FROM trades
GROUP BY strategy_id
ORDER BY strategy_id;