-- Phase 1: Classifier Validation - CORRECT for sparse storage
-- Classifiers use sparse storage - only state changes are recorded
-- Must union with market data to get full time series

PRAGMA memory_limit='3GB';
SET threads=4;

-- Step 1.1: Basic Classifier Overview
SELECT 
    '=== CLASSIFIER DATA OVERVIEW (Sparse Storage) ===' as header;

-- Count classifier state changes (stored records) vs total time coverage
WITH classifier_changes AS (
    SELECT 
        'hidden_markov' as classifier_type,
        COUNT(DISTINCT strat) as num_classifiers,
        COUNT(*) as total_state_changes,
        MIN(ts::timestamp) as earliest_data,
        MAX(ts::timestamp) as latest_data
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/hidden_markov_grid/*.parquet')
    
    UNION ALL
    
    SELECT 
        'market_regime' as classifier_type,
        COUNT(DISTINCT strat) as num_classifiers,
        COUNT(*) as total_state_changes,
        MIN(ts::timestamp) as earliest_data,
        MAX(ts::timestamp) as latest_data
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/market_regime_grid/*.parquet')
    
    UNION ALL
    
    SELECT 
        'microstructure' as classifier_type,
        COUNT(DISTINCT strat) as num_classifiers,
        COUNT(*) as total_state_changes,
        MIN(ts::timestamp) as earliest_data,
        MAX(ts::timestamp) as latest_data
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/microstructure_grid/*.parquet')
    
    UNION ALL
    
    SELECT 
        'multi_timeframe_trend' as classifier_type,
        COUNT(DISTINCT strat) as num_classifiers,
        COUNT(*) as total_state_changes,
        MIN(ts::timestamp) as earliest_data,
        MAX(ts::timestamp) as latest_data
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/multi_timeframe_trend_grid/*.parquet')
    
    UNION ALL
    
    SELECT 
        'volatility_momentum' as classifier_type,
        COUNT(DISTINCT strat) as num_classifiers,
        COUNT(*) as total_state_changes,
        MIN(ts::timestamp) as earliest_data,
        MAX(ts::timestamp) as latest_data
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet')
)
SELECT 
    classifier_type,
    num_classifiers,
    total_state_changes,
    earliest_data,
    latest_data,
    -- Estimate change frequency (changes per day)
    ROUND(total_state_changes::DOUBLE / EXTRACT(DAYS FROM (latest_data - earliest_data + INTERVAL '1 day')), 1) as avg_changes_per_day
FROM classifier_changes
ORDER BY classifier_type;

-- Step 1.2: Regime Persistence Analysis (Key Test!)
-- Check if new 20-30m averaged classifiers have longer-lasting regimes

WITH sample_classifier_full_series AS (
    -- Take one representative classifier and create full time series
    -- Use Hidden Markov as example
    SELECT 
        m.timestamp,
        COALESCE(c.val, LAG(c.val) OVER (ORDER BY m.timestamp)) as current_regime,
        CASE WHEN c.val IS NOT NULL THEN 'STATE_CHANGE' ELSE 'CONTINUATION' END as event_type
    FROM analytics.market_data m
    LEFT JOIN (
        SELECT ts::timestamp as timestamp, val
        FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet')
        WHERE strat = 'SPY_hidden_markov_grid_11_0001_05'
    ) c ON m.timestamp = c.timestamp
    WHERE m.timestamp >= '2024-03-26'  -- Start of classifier data
      AND m.timestamp <= '2025-01-17'  -- End of classifier data
    ORDER BY m.timestamp
),
regime_transitions AS (
    SELECT 
        timestamp,
        current_regime,
        LAG(current_regime) OVER (ORDER BY timestamp) as prev_regime,
        LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp
    FROM sample_classifier_full_series
    WHERE current_regime IS NOT NULL
),
regime_durations AS (
    SELECT 
        current_regime,
        timestamp as change_timestamp,
        EXTRACT(EPOCH FROM (timestamp - prev_timestamp)) / 60 as duration_minutes
    FROM regime_transitions
    WHERE prev_regime IS NOT NULL 
      AND prev_regime != current_regime  -- Only regime changes
)
SELECT 
    '=== REGIME PERSISTENCE TEST (Hidden Markov Sample) ===' as header;

SELECT 
    'All Regime Changes' as analysis_type,
    COUNT(*) as num_regime_changes,
    ROUND(AVG(duration_minutes), 1) as avg_duration_minutes,
    ROUND(MEDIAN(duration_minutes), 1) as median_duration_minutes,
    ROUND(MIN(duration_minutes), 1) as min_duration,
    ROUND(MAX(duration_minutes), 1) as max_duration,
    ROUND(COUNT(CASE WHEN duration_minutes < 10 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_under_10min,
    ROUND(COUNT(CASE WHEN duration_minutes >= 20 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_over_20min,
    ROUND(COUNT(CASE WHEN duration_minutes >= 30 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_over_30min,
    CASE 
        WHEN AVG(duration_minutes) >= 20 THEN 'EXCELLENT (20+ min avg - Target MET!)'
        WHEN AVG(duration_minutes) >= 15 THEN 'GOOD (15+ min avg)'
        WHEN AVG(duration_minutes) >= 10 THEN 'MODERATE (10+ min avg)'
        ELSE 'POOR (< 10 min avg - Too frequent)'
    END as regime_stability_rating
FROM regime_durations
WHERE duration_minutes < 500  -- Filter extreme outliers

UNION ALL

-- Show breakdown by regime type
SELECT 
    'By Regime: ' || current_regime as analysis_type,
    COUNT(*) as num_regime_changes,
    ROUND(AVG(duration_minutes), 1) as avg_duration_minutes,
    ROUND(MEDIAN(duration_minutes), 1) as median_duration_minutes,
    ROUND(MIN(duration_minutes), 1) as min_duration,
    ROUND(MAX(duration_minutes), 1) as max_duration,
    ROUND(COUNT(CASE WHEN duration_minutes < 10 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_under_10min,
    ROUND(COUNT(CASE WHEN duration_minutes >= 20 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_over_20min,
    ROUND(COUNT(CASE WHEN duration_minutes >= 30 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_over_30min,
    'Individual regime analysis' as regime_stability_rating
FROM regime_durations
WHERE duration_minutes < 500
GROUP BY current_regime
ORDER BY avg_duration_minutes DESC;

-- Step 1.3: State Distribution Analysis (After Full Time Series)
WITH sample_market_regime_full AS (
    -- Test Market Regime classifier full time series
    SELECT 
        m.timestamp,
        COALESCE(c.val, LAG(c.val) OVER (ORDER BY m.timestamp)) as current_regime
    FROM analytics.market_data m
    LEFT JOIN (
        SELECT ts::timestamp as timestamp, val
        FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/market_regime_grid/SPY_market_regime_grid_00015_03.parquet')
        WHERE strat = 'SPY_market_regime_grid_00015_03'
    ) c ON m.timestamp = c.timestamp
    WHERE m.timestamp >= '2024-03-26'
      AND m.timestamp <= '2025-01-17'
      AND current_regime IS NOT NULL
),
state_distribution AS (
    SELECT 
        current_regime,
        COUNT(*) as time_periods,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as pct_time
    FROM sample_market_regime_full
    GROUP BY current_regime
)
SELECT 
    '=== STATE DISTRIBUTION ANALYSIS (Market Regime Sample) ===' as header;

SELECT 
    current_regime,
    time_periods,
    ROUND(pct_time, 1) as pct_time,
    CASE 
        WHEN pct_time < 5 THEN 'RARE_STATE (< 5%)'
        WHEN pct_time > 60 THEN 'DOMINANT_STATE (> 60%)'
        ELSE 'BALANCED'
    END as distribution_quality
FROM state_distribution
ORDER BY pct_time DESC;

-- Step 1.4: Compare with Expected Target (20-30 minute regimes)
SELECT 
    '=== REGIME STABILITY ASSESSMENT vs TARGET ===' as header;

WITH stability_summary AS (
    SELECT 
        ROUND(AVG(duration_minutes), 1) as actual_avg_duration,
        ROUND(COUNT(CASE WHEN duration_minutes >= 20 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_meeting_target,
        CASE 
            WHEN AVG(duration_minutes) >= 20 THEN 'TARGET MET: 20+ min average'
            WHEN COUNT(CASE WHEN duration_minutes >= 20 THEN 1 END) * 100.0 / COUNT(*) >= 50 THEN 'CLOSE: 50%+ regimes are 20+ min'
            ELSE 'TARGET MISSED: Need more stable regimes'
        END as target_assessment
    FROM regime_durations
    WHERE duration_minutes < 500
)
SELECT 
    'Target: 20-30 minute regimes' as target,
    actual_avg_duration || ' minutes' as actual_result,
    pct_meeting_target || '% meet target' as success_rate,
    target_assessment as assessment
FROM stability_summary;