-- Check if signal values include leading zeros that could affect trade sequencing
-- E.g., stored as "01", "-01" instead of "1", "-1"

PRAGMA memory_limit='3GB';
SET threads=4;

-- Check actual signal values in classifier files
SELECT 
    '=== CLASSIFIER SIGNAL VALUES ===' as header;

SELECT DISTINCT 
    val as signal_value,
    typeof(val) as value_type,
    LENGTH(val) as value_length,
    CASE 
        WHEN val LIKE '0%' AND val != '0' THEN 'HAS_LEADING_ZERO'
        WHEN val LIKE '-0%' AND val != '0' AND val != '-0' THEN 'HAS_LEADING_ZERO_NEGATIVE'
        ELSE 'NORMAL'
    END as zero_padding_status
FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_55_35.parquet')
ORDER BY val;

-- Check strategy signal values too
SELECT 
    '=== STRATEGY SIGNAL VALUES ===' as header;

SELECT DISTINCT 
    val as signal_value,
    typeof(val) as value_type,
    LENGTH(val) as value_length,
    CASE 
        WHEN val LIKE '0%' AND val != '0' THEN 'HAS_LEADING_ZERO'
        WHEN val LIKE '-0%' AND val != '0' AND val != '-0' THEN 'HAS_LEADING_ZERO_NEGATIVE'
        ELSE 'NORMAL'
    END as zero_padding_status
FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet')
ORDER BY val;

-- Test signal sequencing logic with potential leading zeros
WITH sample_signals AS (
    SELECT 
        ts::timestamp as timestamp,
        val as signal_value,
        LAG(val) OVER (ORDER BY ts::timestamp) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet')
    ORDER BY ts::timestamp
    LIMIT 100
)
SELECT 
    '=== SIGNAL SEQUENCING TEST ===' as header;

SELECT 
    timestamp,
    signal_value,
    prev_signal,
    -- Test different comparison methods
    CASE WHEN signal_value != prev_signal THEN 'DIFFERENT_STRING' ELSE 'SAME_STRING' END as string_comparison,
    CASE WHEN CAST(signal_value AS INTEGER) != CAST(prev_signal AS INTEGER) THEN 'DIFFERENT_INT' ELSE 'SAME_INT' END as int_comparison,
    -- Check for entry/exit patterns
    CASE 
        WHEN prev_signal = '0' AND signal_value = '1' THEN 'LONG_ENTRY'
        WHEN prev_signal = '0' AND signal_value = '-1' THEN 'SHORT_ENTRY'  
        WHEN prev_signal = '1' AND signal_value = '0' THEN 'LONG_EXIT'
        WHEN prev_signal = '-1' AND signal_value = '0' THEN 'SHORT_EXIT'
        WHEN prev_signal = '0' AND signal_value = '01' THEN 'LONG_ENTRY_LEADING_ZERO'
        WHEN prev_signal = '01' AND signal_value = '0' THEN 'LONG_EXIT_LEADING_ZERO'
        ELSE 'OTHER'
    END as trade_signal_type
FROM sample_signals
WHERE prev_signal IS NOT NULL
  AND (signal_value != prev_signal OR CAST(signal_value AS INTEGER) != CAST(prev_signal AS INTEGER))
ORDER BY timestamp
LIMIT 20;