-- Check Timezone Alignment Issue
-- Data showing 13:00-21:00 suggests 8-hour shift (likely UTC vs EST misalignment)

PRAGMA memory_limit='3GB';
SET threads=4;

-- Check timezone interpretation in market data
SELECT 
    '=== TIMEZONE ALIGNMENT CHECK ===' as header;

SELECT 
    'Market Data Timestamps' as source,
    MIN(timestamp) as earliest,
    MAX(timestamp) as latest,
    EXTRACT(hour FROM MIN(timestamp)) as earliest_hour,
    EXTRACT(hour FROM MAX(timestamp)) as latest_hour,
    'Hours ' || EXTRACT(hour FROM MIN(timestamp)) || '-' || EXTRACT(hour FROM MAX(timestamp)) as time_range
FROM analytics.market_data
WHERE timestamp >= '2024-03-26' AND timestamp <= '2025-01-17'

UNION ALL

SELECT 
    'Classifier Data Timestamps' as source,
    MIN(ts::timestamp) as earliest,
    MAX(ts::timestamp) as latest,
    EXTRACT(hour FROM MIN(ts::timestamp)) as earliest_hour,
    EXTRACT(hour FROM MAX(ts::timestamp)) as latest_hour,
    'Hours ' || EXTRACT(hour FROM MIN(ts::timestamp)) || '-' || EXTRACT(hour FROM MAX(ts::timestamp)) as time_range
FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_55_35.parquet')
WHERE strat = 'SPY_volatility_momentum_grid_05_55_35';

-- Test timezone shift hypothesis
-- If data is 4 hours ahead (UTC showing EST times), shift back 4 hours
-- If data is 4 hours behind (EST showing UTC times), shift forward 4 hours
SELECT 
    '=== TIMEZONE SHIFT TEST ===' as header;

WITH timezone_test AS (
    SELECT 
        timestamp as original_time,
        timestamp - INTERVAL '4 hours' as shifted_minus_4h,
        timestamp + INTERVAL '4 hours' as shifted_plus_4h,
        EXTRACT(hour FROM timestamp) as original_hour,
        EXTRACT(hour FROM timestamp - INTERVAL '4 hours') as hour_minus_4,
        EXTRACT(hour FROM timestamp + INTERVAL '4 hours') as hour_plus_4
    FROM analytics.market_data
    WHERE timestamp >= '2024-03-26' AND timestamp <= '2024-03-27'
    ORDER BY timestamp
    LIMIT 20
)
SELECT 
    original_time,
    original_hour,
    hour_minus_4,
    hour_plus_4,
    CASE 
        WHEN hour_minus_4 BETWEEN 9 AND 16 THEN 'MINUS_4H_GIVES_MARKET_HOURS'
        WHEN hour_plus_4 BETWEEN 9 AND 16 THEN 'PLUS_4H_GIVES_MARKET_HOURS'
        WHEN original_hour BETWEEN 9 AND 16 THEN 'ORIGINAL_IS_MARKET_HOURS'
        ELSE 'NO_MATCH'
    END as timezone_hypothesis
FROM timezone_test;

-- Check hourly distribution with timezone shift
SELECT 
    '=== HOURLY DISTRIBUTION WITH -4H SHIFT ===' as header;

SELECT 
    EXTRACT(hour FROM timestamp - INTERVAL '4 hours') as adjusted_hour,
    COUNT(*) as records,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct_of_total
FROM analytics.market_data
WHERE timestamp >= '2024-03-26' AND timestamp <= '2025-01-17'
GROUP BY EXTRACT(hour FROM timestamp - INTERVAL '4 hours')
ORDER BY adjusted_hour;

-- Sample of corrected timestamps
SELECT 
    '=== SAMPLE CORRECTED TIMESTAMPS ===' as header;

SELECT 
    timestamp as stored_timestamp,
    timestamp - INTERVAL '4 hours' as corrected_est_time,
    EXTRACT(hour FROM timestamp) as stored_hour,
    EXTRACT(hour FROM timestamp - INTERVAL '4 hours') as corrected_hour,
    CASE 
        WHEN EXTRACT(hour FROM timestamp - INTERVAL '4 hours') BETWEEN 9 AND 16 THEN 'MARKET_HOURS'
        WHEN EXTRACT(hour FROM timestamp - INTERVAL '4 hours') BETWEEN 16 AND 20 THEN 'AFTER_HOURS'
        ELSE 'PRE_MARKET'
    END as session_type
FROM analytics.market_data
WHERE timestamp >= '2024-03-26 13:00:00' AND timestamp <= '2024-03-26 21:00:00'
ORDER BY timestamp
LIMIT 20;