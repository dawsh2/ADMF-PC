-- Final Comprehensive Timezone Validation
SELECT '=== TIMEZONE VALIDATION ACROSS MULTIPLE STRATEGIES ===' as header;

-- Test multiple different strategy types
WITH combined_data AS (
    SELECT ts, 'macd_crossover' as strategy FROM 'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_35_9.parquet'
    UNION ALL
    SELECT ts, 'ema_crossover' as strategy FROM 'traces/SPY_1m/signals/ema_crossover_grid/SPY_ema_crossover_grid_11_15.parquet'  
    UNION ALL
    SELECT ts, 'rsi_threshold' as strategy FROM 'traces/SPY_1m/signals/rsi_threshold_grid/SPY_rsi_threshold_grid_14_25.parquet'
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

-- Show the clear pattern: UTC 13-20 maps to EST 9-16
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

SELECT '=== TIMEZONE CORRECTION RECOMMENDATION ===' as header;

-- Final recommendation
SELECT 
    'CONFIRMED TIMEZONE MISALIGNMENT' as status,
    'UTC timestamps representing EST market times' as issue,
    'Apply -4 hours offset (EDT) or -5 hours (EST)' as solution,
    'Data spans Mar 2024 - Jan 2025 (primarily EDT period)' as note,
    '13:00-21:00 UTC → 9:00-17:00 EST' as mapping;