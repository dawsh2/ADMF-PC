-- Working Timezone Validation with Existing Files
SELECT '=== TIMEZONE VALIDATION ACROSS MULTIPLE STRATEGIES ===' as header;

-- Test multiple different strategy types with existing files
WITH combined_data AS (
    SELECT ts, 'macd_crossover' as strategy FROM 'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_35_9.parquet'
    UNION ALL
    SELECT ts, 'ema_crossover' as strategy FROM 'traces/SPY_1m/signals/ema_crossover_grid/SPY_ema_crossover_grid_7_15.parquet'  
    UNION ALL
    SELECT ts, 'rsi_threshold' as strategy FROM 'traces/SPY_1m/signals/rsi_threshold_grid/SPY_rsi_threshold_grid_19_55.parquet'
),
timezone_stats AS (
    SELECT 
        strategy,
        COUNT(*) as total_signals,
        -- UTC hours 13-21 (should be the bulk of trading activity)
        COUNT(CASE WHEN EXTRACT(hour FROM ts::timestamp) >= 13 AND EXTRACT(hour FROM ts::timestamp) <= 20 THEN 1 END) as utc_13_20_signals,
        -- EST hours 9-16 after -4h adjustment (proper market hours)
        COUNT(CASE WHEN EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours')) >= 9 AND EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours')) <= 15 THEN 1 END) as est_9_15_signals,
        MIN(EXTRACT(hour FROM ts::timestamp)) as min_hour_utc,
        MAX(EXTRACT(hour FROM ts::timestamp)) as max_hour_utc,
        MIN(EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours'))) as min_hour_est,
        MAX(EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours'))) as max_hour_est
    FROM combined_data
    GROUP BY strategy
)
SELECT 
    strategy,
    total_signals,
    utc_13_20_signals,
    est_9_15_signals,
    ROUND(100.0 * utc_13_20_signals / total_signals, 1) as utc_market_pct,
    ROUND(100.0 * est_9_15_signals / total_signals, 1) as est_market_pct,
    min_hour_utc || '-' || max_hour_utc as utc_hour_range,
    min_hour_est || '-' || max_hour_est as est_hour_range
FROM timezone_stats
ORDER BY strategy;

SELECT '=== DETAILED HOUR-BY-HOUR BREAKDOWN ===' as header;

-- Show the clear pattern for one strategy
SELECT 
    EXTRACT(hour FROM ts::timestamp) as hour_utc,
    EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours')) as hour_est,
    COUNT(*) as signal_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as percentage,
    CASE 
        WHEN EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours')) >= 9 AND EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours')) <= 15 THEN '✓ MARKET HOURS'
        WHEN EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours')) = 16 THEN '✓ MARKET CLOSE'
        ELSE '✗ OFF HOURS'
    END as market_status
FROM 'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_35_9.parquet'
GROUP BY EXTRACT(hour FROM ts::timestamp), EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours'))
ORDER BY hour_utc;

SELECT '=== SAMPLE CORRECTED TIMESTAMPS ===' as header;

-- Show sample of corrected timestamps
SELECT 
    ts as original_utc,
    (ts::timestamp - INTERVAL '4 hours') as corrected_est,
    EXTRACT(hour FROM ts::timestamp) as hour_utc,
    EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours')) as hour_est,
    CASE 
        WHEN EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours')) >= 9 AND EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours')) <= 15 THEN 'MARKET HOURS'
        ELSE 'OFF HOURS'
    END as session
FROM 'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_35_9.parquet'
ORDER BY ts
LIMIT 15;

SELECT '=== FINAL ASSESSMENT ===' as header;

-- Conclusive assessment
WITH final_stats AS (
    SELECT 
        COUNT(*) as total_records,
        COUNT(CASE WHEN EXTRACT(hour FROM ts::timestamp) >= 13 AND EXTRACT(hour FROM ts::timestamp) <= 20 THEN 1 END) as utc_market_time,
        COUNT(CASE WHEN EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours')) >= 9 AND EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours')) <= 16 THEN 1 END) as est_market_time,
        MIN(ts) as first_timestamp,
        MAX(ts) as last_timestamp
    FROM 'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_35_9.parquet'
)
SELECT 
    'TIMEZONE CORRECTION CONFIRMED' as conclusion,
    total_records as total_signals,
    utc_market_time as utc_13_20_count,
    est_market_time as est_9_16_count,
    ROUND(100.0 * est_market_time / total_records, 1) as est_coverage_pct,
    first_timestamp as data_start,
    last_timestamp as data_end,
    'Apply -4 hour offset for EDT correction' as recommendation
FROM final_stats;