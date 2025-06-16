
-- Analyze signal patterns for different strategy types
WITH macd_signals AS (
    SELECT 
        ts,
        CAST(val AS INTEGER) as signal_val,
        LAG(CAST(val AS INTEGER), 1) OVER (ORDER BY ts) as prev_signal,
        ROW_NUMBER() OVER (ORDER BY ts) as seq
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet')
    ORDER BY ts
    LIMIT 30
),

williams_signals AS (
    SELECT 
        ts,
        CAST(val AS INTEGER) as signal_val,
        LAG(CAST(val AS INTEGER), 1) OVER (ORDER BY ts) as prev_signal,
        ROW_NUMBER() OVER (ORDER BY ts) as seq
    FROM read_parquet('traces/SPY_1m/signals/williams_r_grid/SPY_williams_r_grid_7_-80_-20.parquet')
    ORDER BY ts
    LIMIT 30
),

rsi_signals AS (
    SELECT 
        ts,
        CAST(val AS INTEGER) as signal_val,
        LAG(CAST(val AS INTEGER), 1) OVER (ORDER BY ts) as prev_signal,
        ROW_NUMBER() OVER (ORDER BY ts) as seq
    FROM read_parquet('traces/SPY_1m/signals/rsi_threshold_grid/SPY_rsi_threshold_grid_11_50.parquet')
    ORDER BY ts
    LIMIT 30
)

-- Show patterns for each strategy type
SELECT 'MACD_CROSSOVER' as strategy, seq, ts, signal_val, prev_signal,
       CASE WHEN prev_signal IS NOT NULL AND signal_val = prev_signal THEN 'DUPLICATE' ELSE 'CHANGE' END as pattern_type
FROM macd_signals

UNION ALL

SELECT 'WILLIAMS_R' as strategy, seq, ts, signal_val, prev_signal,
       CASE WHEN prev_signal IS NOT NULL AND signal_val = prev_signal THEN 'DUPLICATE' ELSE 'CHANGE' END as pattern_type
FROM williams_signals

UNION ALL

SELECT 'RSI_THRESHOLD' as strategy, seq, ts, signal_val, prev_signal,
       CASE WHEN prev_signal IS NOT NULL AND signal_val = prev_signal THEN 'DUPLICATE' ELSE 'CHANGE' END as pattern_type
FROM rsi_signals

ORDER BY strategy, seq;
