-- Debug RSI calculations step by step
PRAGMA memory_limit='3GB';
SET threads=4;

ATTACH '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_c1ab337f/analytics.duckdb' AS analytics;

-- First, let's look at a sample of signals to understand the pattern
SELECT 
    strat,
    ts,
    val,
    LAG(val) OVER (PARTITION BY strat ORDER BY ts) as prev_val,
    LEAD(val) OVER (PARTITION BY strat ORDER BY ts) as next_val
FROM read_parquet('/Users/daws/ADMF-PC/workspaces/rsi_bands_custom_v1_1a9e2420/traces/SPY_1m/signals/rsi_bands_grid/SPY_rsi_bands_grid_19_30_80.parquet')
WHERE ts >= '2025-01-17'
ORDER BY ts
LIMIT 30;

-- Now let's trace through the calculation for a few trades
WITH signals AS (
    SELECT 
        strat,
        ts::timestamp as timestamp,
        val as signal_value,
        LAG(val) OVER (PARTITION BY strat ORDER BY ts) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/rsi_bands_custom_v1_1a9e2420/traces/SPY_1m/signals/rsi_bands_grid/SPY_rsi_bands_grid_19_30_80.parquet')
    WHERE ts >= '2025-01-17'
),
signal_with_prices AS (
    SELECT 
        s.strat as strategy_id,
        s.timestamp,
        s.signal_value,
        s.prev_signal,
        m.close as current_price,
        LAG(m.close) OVER (PARTITION BY s.strat ORDER BY s.timestamp) as prev_price,
        -- Show what type of signal change this is
        CASE 
            WHEN prev_signal = 0 AND signal_value = 1 THEN 'LONG_ENTRY'
            WHEN prev_signal = 0 AND signal_value = -1 THEN 'SHORT_ENTRY'
            WHEN prev_signal = 1 AND signal_value = 0 THEN 'LONG_EXIT'
            WHEN prev_signal = -1 AND signal_value = 0 THEN 'SHORT_EXIT'
            WHEN prev_signal = 1 AND signal_value = -1 THEN 'LONG_TO_SHORT'
            WHEN prev_signal = -1 AND signal_value = 1 THEN 'SHORT_TO_LONG'
            WHEN prev_signal = signal_value THEN 'NO_CHANGE'
            ELSE 'OTHER'
        END as signal_type
    FROM signals s
    JOIN analytics.market_data m ON s.timestamp = m.timestamp
    WHERE s.timestamp >= '2025-01-17 00:00:00'
)
SELECT 
    strategy_id,
    timestamp,
    signal_value,
    prev_signal,
    current_price,
    prev_price,
    signal_type,
    -- Calculate return for exits
    CASE 
        WHEN signal_type = 'LONG_EXIT' THEN ROUND((current_price / prev_price - 1) * 10000, 2)
        WHEN signal_type = 'SHORT_EXIT' THEN ROUND((prev_price / current_price - 1) * 10000, 2)
        ELSE NULL
    END as return_bps
FROM signal_with_prices
WHERE signal_type != 'NO_CHANGE'
ORDER BY timestamp
LIMIT 50;

-- Let me also check if we're properly matching entry/exit pairs
WITH signals AS (
    SELECT 
        strat,
        ts::timestamp as timestamp,
        val as signal_value
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/rsi_bands_custom_v1_1a9e2420/traces/SPY_1m/signals/rsi_bands_grid/SPY_rsi_bands_grid_19_30_80.parquet')
    WHERE ts >= '2025-01-17'
),
entries AS (
    SELECT 
        s.strat as strategy_id,
        s.timestamp as entry_time,
        s.signal_value as position_type,
        m.close as entry_price,
        ROW_NUMBER() OVER (PARTITION BY s.strat ORDER BY s.timestamp) as entry_num
    FROM signals s
    JOIN analytics.market_data m ON s.timestamp = m.timestamp
    WHERE s.signal_value != 0
      AND LAG(s.signal_value) OVER (PARTITION BY s.strat ORDER BY s.timestamp) = 0
),
exits AS (
    SELECT 
        s.strat as strategy_id,
        s.timestamp as exit_time,
        LAG(s.signal_value) OVER (PARTITION BY s.strat ORDER BY s.timestamp) as position_type,
        m.close as exit_price,
        ROW_NUMBER() OVER (PARTITION BY s.strat ORDER BY s.timestamp) as exit_num
    FROM signals s
    JOIN analytics.market_data m ON s.timestamp = m.timestamp
    WHERE s.signal_value = 0
      AND LAG(s.signal_value) OVER (PARTITION BY s.strat ORDER BY s.timestamp) != 0
)
SELECT 
    e.strategy_id,
    e.entry_time,
    e.entry_price,
    x.exit_time,
    x.exit_price,
    CASE 
        WHEN e.position_type = 1 THEN 'LONG'
        ELSE 'SHORT'
    END as direction,
    EXTRACT(EPOCH FROM (x.exit_time - e.entry_time)) / 60 as duration_minutes,
    CASE 
        WHEN e.position_type = 1 THEN ROUND((x.exit_price / e.entry_price - 1) * 10000, 2)
        ELSE ROUND((e.entry_price / x.exit_price - 1) * 10000, 2)
    END as return_bps
FROM entries e
LEFT JOIN exits x ON e.strategy_id = x.strategy_id
    AND e.position_type = x.position_type
    AND x.exit_time > e.entry_time
    AND NOT EXISTS (
        SELECT 1 FROM exits x2
        WHERE x2.strategy_id = e.strategy_id
          AND x2.position_type = e.position_type
          AND x2.exit_time > e.entry_time
          AND x2.exit_time < x.exit_time
    )
ORDER BY e.entry_time
LIMIT 20;