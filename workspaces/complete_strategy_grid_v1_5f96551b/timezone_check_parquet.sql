-- Timezone Alignment Check for Parquet Files
-- This script tests the hypothesis that timestamps are in UTC but represent EST market hours
-- Expected: 13:00-21:00 UTC = 9:00-17:00 EST (during daylight saving time)

SELECT '=== TIMEZONE ALIGNMENT ANALYSIS ===' as header;

-- 1. Current hour distribution in UTC
SELECT 
    'Current Hour Distribution (UTC)' as analysis_type,
    EXTRACT(hour FROM ts::timestamp) as hour_utc,
    COUNT(*) as signal_count
FROM 'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_35_9.parquet'
GROUP BY EXTRACT(hour FROM ts::timestamp)
ORDER BY hour_utc;

SELECT '=== TIMEZONE SHIFT TEST ===' as header;

-- 2. Hour distribution after -4 hour shift (UTC to EST)
SELECT 
    'Hour Distribution After -4H Shift (EST)' as analysis_type,
    EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours')) as hour_est,
    COUNT(*) as signal_count
FROM 'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_35_9.parquet'
GROUP BY EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours'))
ORDER BY hour_est;

SELECT '=== SAMPLE TIMESTAMP COMPARISON ===' as header;

-- 3. Sample timestamps showing before/after conversion
SELECT 
    'Sample Timestamps' as analysis_type,
    ts as original_utc,
    (ts::timestamp - INTERVAL '4 hours') as adjusted_est,
    EXTRACT(hour FROM ts::timestamp) as hour_utc,
    EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours')) as hour_est
FROM 'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_35_9.parquet'
ORDER BY ts
LIMIT 10;

SELECT '=== MARKET HOURS VALIDATION ===' as header;

-- 4. Check if adjusted times fall within market hours (9:00-16:00 EST)
WITH adjusted_times AS (
    SELECT 
        ts as original_utc,
        (ts::timestamp - INTERVAL '4 hours') as adjusted_est,
        EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours')) as hour_est
    FROM 'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_35_9.parquet'
),
market_hours_check AS (
    SELECT 
        COUNT(*) as total_signals,
        COUNT(CASE WHEN hour_est >= 9 AND hour_est < 16 THEN 1 END) as market_hours_signals,
        COUNT(CASE WHEN hour_est < 9 OR hour_est >= 16 THEN 1 END) as non_market_hours_signals
    FROM adjusted_times
)
SELECT 
    'Market Hours Coverage After -4H Adjustment' as analysis_type,
    total_signals,
    market_hours_signals,
    non_market_hours_signals,
    ROUND(100.0 * market_hours_signals / total_signals, 2) as market_hours_percentage
FROM market_hours_check;

SELECT '=== TIME RANGE ANALYSIS ===' as header;

-- 5. Overall time range analysis
SELECT 
    'Time Range Analysis' as analysis_type,
    MIN(ts) as earliest_utc,
    MAX(ts) as latest_utc,
    MIN(ts::timestamp - INTERVAL '4 hours') as earliest_est,
    MAX(ts::timestamp - INTERVAL '4 hours') as latest_est,
    COUNT(DISTINCT DATE(ts)) as trading_days
FROM 'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_35_9.parquet';

SELECT '=== CONCLUSION ===' as header;

-- 6. Timezone hypothesis validation
WITH hourly_analysis AS (
    SELECT 
        EXTRACT(hour FROM ts::timestamp) as hour_utc,
        EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours')) as hour_est,
        COUNT(*) as signal_count
    FROM 'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_35_9.parquet'
    GROUP BY EXTRACT(hour FROM ts::timestamp), EXTRACT(hour FROM (ts::timestamp - INTERVAL '4 hours'))
),
market_hours_validation AS (
    SELECT 
        COUNT(CASE WHEN hour_utc >= 13 AND hour_utc < 21 THEN 1 END) as utc_13_21_count,
        COUNT(CASE WHEN hour_est >= 9 AND hour_est < 17 THEN 1 END) as est_9_17_count,
        COUNT(*) as total_count
    FROM hourly_analysis
)
SELECT 
    'Timezone Hypothesis Validation' as conclusion,
    CASE 
        WHEN utc_13_21_count > 0 AND est_9_17_count > 0 AND utc_13_21_count = est_9_17_count 
        THEN 'CONFIRMED: Data is UTC timestamps representing EST market hours'
        ELSE 'INCONCLUSIVE: Further investigation needed'
    END as hypothesis_result,
    utc_13_21_count as signals_in_utc_13_21,
    est_9_17_count as signals_in_est_9_17,
    total_count as total_signals
FROM market_hours_validation;