-- Investigate Time Coverage Issue
-- Why are classifiers only showing 13:00-15:00 data?
-- Compare classifier timestamps vs market data timestamps

PRAGMA memory_limit='3GB';
SET threads=4;

-- First, check what time range we have in market data
SELECT 
    '=== MARKET DATA TIME COVERAGE ===' as header;

SELECT 
    MIN(timestamp) as earliest_market_data,
    MAX(timestamp) as latest_market_data,
    COUNT(*) as total_market_records,
    COUNT(DISTINCT DATE(timestamp)) as trading_days,
    MIN(EXTRACT(hour FROM timestamp)) as earliest_hour,
    MAX(EXTRACT(hour FROM timestamp)) as latest_hour
FROM analytics.market_data;

-- Check hourly distribution of market data
SELECT 
    '=== MARKET DATA HOURLY DISTRIBUTION ===' as header;

SELECT 
    EXTRACT(hour FROM timestamp) as hour,
    COUNT(*) as records,
    COUNT(DISTINCT DATE(timestamp)) as days_with_data,
    MIN(DATE(timestamp)) as first_day,
    MAX(DATE(timestamp)) as last_day
FROM analytics.market_data
WHERE timestamp >= '2024-03-26' AND timestamp <= '2025-01-17'
GROUP BY EXTRACT(hour FROM timestamp)
ORDER BY hour;

-- Now check classifier time coverage
SELECT 
    '=== CLASSIFIER TIME COVERAGE COMPARISON ===' as header;

-- Hidden Markov time coverage
WITH hm_times AS (
    SELECT 
        'hidden_markov' as classifier_type,
        ts::timestamp as timestamp
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet')
    WHERE strat = 'SPY_hidden_markov_grid_11_0001_05'
),
mr_times AS (
    SELECT 
        'market_regime' as classifier_type,
        ts::timestamp as timestamp
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/market_regime_grid/SPY_market_regime_grid_00015_03.parquet')
    WHERE strat = 'SPY_market_regime_grid_00015_03'
),
vm_times AS (
    SELECT 
        'volatility_momentum' as classifier_type,
        ts::timestamp as timestamp
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_55_35.parquet')
    WHERE strat = 'SPY_volatility_momentum_grid_05_55_35'
),
all_classifier_times AS (
    SELECT * FROM hm_times
    UNION ALL
    SELECT * FROM mr_times
    UNION ALL
    SELECT * FROM vm_times
)
SELECT 
    classifier_type,
    MIN(timestamp) as earliest_classifier_data,
    MAX(timestamp) as latest_classifier_data,
    COUNT(*) as classifier_records,
    MIN(EXTRACT(hour FROM timestamp)) as earliest_hour,
    MAX(EXTRACT(hour FROM timestamp)) as latest_hour,
    COUNT(DISTINCT EXTRACT(hour FROM timestamp)) as unique_hours
FROM all_classifier_times
GROUP BY classifier_type;

-- Detailed hourly breakdown for classifiers
SELECT 
    '=== CLASSIFIER HOURLY DISTRIBUTION ===' as header;

WITH all_classifier_times AS (
    SELECT 
        'hidden_markov' as classifier_type,
        ts::timestamp as timestamp
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet')
    WHERE strat = 'SPY_hidden_markov_grid_11_0001_05'
    
    UNION ALL
    
    SELECT 
        'market_regime' as classifier_type,
        ts::timestamp as timestamp
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/market_regime_grid/SPY_market_regime_grid_00015_03.parquet')
    WHERE strat = 'SPY_market_regime_grid_00015_03'
    
    UNION ALL
    
    SELECT 
        'volatility_momentum' as classifier_type,
        ts::timestamp as timestamp
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_55_35.parquet')
    WHERE strat = 'SPY_volatility_momentum_grid_05_55_35'
)
SELECT 
    classifier_type,
    EXTRACT(hour FROM timestamp) as hour,
    COUNT(*) as records,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY classifier_type), 1) as pct_of_classifier_data
FROM all_classifier_times
GROUP BY classifier_type, EXTRACT(hour FROM timestamp)
ORDER BY classifier_type, hour;

-- Check if market data has full coverage but classifiers are missing morning hours
SELECT 
    '=== MISSING CLASSIFIER COVERAGE ANALYSIS ===' as header;

WITH market_hours AS (
    SELECT DISTINCT 
        EXTRACT(hour FROM timestamp) as hour
    FROM analytics.market_data
    WHERE timestamp >= '2024-03-26' AND timestamp <= '2025-01-17'
      AND EXTRACT(hour FROM timestamp) BETWEEN 9 AND 15
),
classifier_hours AS (
    SELECT DISTINCT 
        EXTRACT(hour FROM ts::timestamp) as hour
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_55_35.parquet')
    WHERE strat = 'SPY_volatility_momentum_grid_05_55_35'
)
SELECT 
    m.hour,
    CASE WHEN c.hour IS NOT NULL THEN 'HAS_CLASSIFIER_DATA' ELSE 'MISSING_CLASSIFIER_DATA' END as classifier_coverage
FROM market_hours m
LEFT JOIN classifier_hours c ON m.hour = c.hour
ORDER BY m.hour;

-- Sample timestamps to see the exact pattern
SELECT 
    '=== SAMPLE TIMESTAMPS COMPARISON ===' as header;

-- Show first 20 market data timestamps vs first 20 classifier timestamps
WITH market_sample AS (
    SELECT 
        'market_data' as source,
        timestamp,
        ROW_NUMBER() OVER (ORDER BY timestamp) as rn
    FROM analytics.market_data
    WHERE timestamp >= '2024-03-26'
    ORDER BY timestamp
    LIMIT 20
),
classifier_sample AS (
    SELECT 
        'classifier_data' as source,
        ts::timestamp as timestamp,
        ROW_NUMBER() OVER (ORDER BY ts::timestamp) as rn
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_55_35.parquet')
    WHERE strat = 'SPY_volatility_momentum_grid_05_55_35'
    ORDER BY ts::timestamp
    LIMIT 20
)
SELECT source, timestamp FROM market_sample
UNION ALL
SELECT source, timestamp FROM classifier_sample
ORDER BY timestamp;