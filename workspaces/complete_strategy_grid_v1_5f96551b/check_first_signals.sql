-- Check if first signals are transitions TO zero (which would break trade sequencing)
-- This would cause us to count "exits" without corresponding "entries"

PRAGMA memory_limit='3GB';
SET threads=4;

-- Check first signal values across multiple strategy files
WITH first_signals AS (
    -- MACD Crossover first signals
    SELECT 
        'macd_crossover' as strategy_type,
        strat as strategy_id,
        MIN(ts::timestamp) as first_timestamp,
        FIRST_VALUE(val) OVER (PARTITION BY strat ORDER BY ts::timestamp) as first_signal_value
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/*.parquet')
    GROUP BY strat
    
    UNION ALL
    
    -- EMA Crossover first signals  
    SELECT 
        'ema_crossover' as strategy_type,
        strat as strategy_id,
        MIN(ts::timestamp) as first_timestamp,
        FIRST_VALUE(val) OVER (PARTITION BY strat ORDER BY ts::timestamp) as first_signal_value
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/ema_crossover_grid/*.parquet')
    GROUP BY strat
    
    UNION ALL
    
    -- RSI Threshold first signals
    SELECT 
        'rsi_threshold' as strategy_type,
        strat as strategy_id,
        MIN(ts::timestamp) as first_timestamp,
        FIRST_VALUE(val) OVER (PARTITION BY strat ORDER BY ts::timestamp) as first_signal_value
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/rsi_threshold_grid/*.parquet')
    GROUP BY strat
    
    UNION ALL
    
    -- Bollinger Breakout first signals
    SELECT 
        'bollinger_breakout' as strategy_type,
        strat as strategy_id,
        MIN(ts::timestamp) as first_timestamp,
        FIRST_VALUE(val) OVER (PARTITION BY strat ORDER BY ts::timestamp) as first_signal_value
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/bollinger_breakout_grid/*.parquet')
    GROUP BY strat
)
SELECT 
    '=== FIRST SIGNAL VALUE ANALYSIS ===' as header;

SELECT 
    strategy_type,
    first_signal_value,
    COUNT(*) as strategy_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY strategy_type), 1) as pct_of_strategies,
    CASE 
        WHEN first_signal_value = 0 THEN 'NEUTRAL_START (Could break trade logic)'
        WHEN first_signal_value = 1 THEN 'LONG_START'  
        WHEN first_signal_value = -1 THEN 'SHORT_START'
        ELSE 'OTHER'
    END as signal_interpretation
FROM first_signals
GROUP BY strategy_type, first_signal_value
ORDER BY strategy_type, first_signal_value;

-- Check for problematic sequences: first signal = 0, then next signal = 1 or -1
WITH signal_sequences AS (
    SELECT 
        strat as strategy_id,
        ts::timestamp as timestamp,
        val as signal_value,
        ROW_NUMBER() OVER (PARTITION BY strat ORDER BY ts::timestamp) as signal_order,
        LAG(val) OVER (PARTITION BY strat ORDER BY ts::timestamp) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet')
    ORDER BY ts::timestamp
),
problematic_starts AS (
    SELECT 
        strategy_id,
        signal_order,
        prev_signal,
        signal_value,
        CASE 
            WHEN signal_order = 1 AND signal_value = 0 THEN 'STARTS_WITH_ZERO'
            WHEN signal_order = 2 AND prev_signal = 0 AND signal_value IN (1, -1) THEN 'ZERO_TO_POSITION (Fake Entry)'
            WHEN signal_order = 2 AND prev_signal IN (1, -1) AND signal_value = 0 THEN 'POSITION_TO_ZERO (Fake Exit)'
            ELSE 'NORMAL'
        END as sequence_type
    FROM signal_sequences
    WHERE signal_order <= 3  -- Focus on first few signals
)
SELECT 
    '=== PROBLEMATIC SIGNAL SEQUENCES ===' as header;

SELECT 
    sequence_type,
    COUNT(*) as occurrence_count,
    STRING_AGG(DISTINCT strategy_id, ', ') as example_strategies
FROM problematic_starts
WHERE sequence_type != 'NORMAL'
GROUP BY sequence_type;

-- Sample actual signal sequences for a few strategies
SELECT 
    '=== SAMPLE SIGNAL SEQUENCES (First 10 signals) ===' as header;

WITH sample_sequences AS (
    SELECT 
        strat as strategy_id,
        ts::timestamp as timestamp,
        val as signal_value,
        LAG(val) OVER (PARTITION BY strat ORDER BY ts::timestamp) as prev_signal,
        ROW_NUMBER() OVER (PARTITION BY strat ORDER BY ts::timestamp) as signal_order
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet')
    ORDER BY ts::timestamp
)
SELECT 
    strategy_id,
    signal_order,
    timestamp,
    prev_signal,
    signal_value,
    CASE 
        WHEN prev_signal IS NULL THEN 'FIRST_SIGNAL'
        WHEN prev_signal = 0 AND signal_value IN (1, -1) THEN 'NEUTRAL_TO_POSITION'
        WHEN prev_signal IN (1, -1) AND signal_value = 0 THEN 'POSITION_TO_NEUTRAL'
        WHEN prev_signal = 1 AND signal_value = -1 THEN 'LONG_TO_SHORT'
        WHEN prev_signal = -1 AND signal_value = 1 THEN 'SHORT_TO_LONG'
        ELSE 'OTHER'
    END as transition_type
FROM sample_sequences
WHERE signal_order <= 10
ORDER BY strategy_id, signal_order;