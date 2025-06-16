-- Phase 1: Classifier Validation - FINAL VERSION
-- Direct analysis of classifier sparse data
-- Focus on regime duration analysis

PRAGMA memory_limit='3GB';
SET threads=4;

-- Step 1: Classifier Overview (Already worked)
SELECT 
    '=== CLASSIFIER DATA OVERVIEW (Sparse Storage) ===' as header;

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
    ROUND(total_state_changes::DOUBLE / EXTRACT(DAYS FROM (latest_data - earliest_data + INTERVAL '1 day')), 1) as avg_changes_per_day
FROM classifier_changes
ORDER BY classifier_type;

-- Step 2: CRITICAL REGIME DURATION ANALYSIS
-- This is the key test - do we have 20-30 minute regimes?
SELECT 
    '=== REGIME PERSISTENCE TEST (Direct from Classifier Data) ===' as header;

-- Analyze regime durations from state changes in classifier data
WITH sample_classifier_data AS (
    -- Use Hidden Markov classifier sample
    SELECT 
        ts::timestamp as timestamp,
        val as regime_state,
        strat
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet')
    WHERE strat = 'SPY_hidden_markov_grid_11_0001_05'
    ORDER BY ts
),
regime_durations AS (
    SELECT 
        regime_state,
        timestamp,
        LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
        LAG(regime_state) OVER (ORDER BY timestamp) as prev_state,
        EXTRACT(EPOCH FROM (timestamp - LAG(timestamp) OVER (ORDER BY timestamp))) / 60 as duration_minutes
    FROM sample_classifier_data
)
SELECT 
    'Direct State Change Analysis' as analysis_type,
    COUNT(*) as num_regime_changes,
    ROUND(AVG(duration_minutes), 1) as avg_duration_minutes,
    ROUND(MEDIAN(duration_minutes), 1) as median_duration_minutes,
    ROUND(MIN(duration_minutes), 1) as min_duration,
    ROUND(MAX(duration_minutes), 1) as max_duration,
    ROUND(COUNT(CASE WHEN duration_minutes < 5 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_under_5min,
    ROUND(COUNT(CASE WHEN duration_minutes < 10 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_under_10min,
    ROUND(COUNT(CASE WHEN duration_minutes >= 20 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_over_20min,
    ROUND(COUNT(CASE WHEN duration_minutes >= 30 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_over_30min,
    CASE 
        WHEN AVG(duration_minutes) >= 20 THEN 'EXCELLENT (20+ min avg - TARGET MET!)'
        WHEN AVG(duration_minutes) >= 15 THEN 'GOOD (15+ min avg)'
        WHEN AVG(duration_minutes) >= 10 THEN 'MODERATE (10+ min avg)'
        WHEN AVG(duration_minutes) >= 5 THEN 'POOR (5-10 min avg)'
        ELSE 'TERRIBLE (< 5 min avg - Unusable!)'
    END as regime_stability_rating
FROM regime_durations
WHERE duration_minutes IS NOT NULL 
  AND duration_minutes > 0 
  AND duration_minutes < 500  -- Filter extreme outliers

UNION ALL

-- Show breakdown by regime type  
SELECT 
    'By Regime: ' || regime_state as analysis_type,
    COUNT(*) as num_regime_changes,
    ROUND(AVG(duration_minutes), 1) as avg_duration_minutes,
    ROUND(MEDIAN(duration_minutes), 1) as median_duration_minutes,
    ROUND(MIN(duration_minutes), 1) as min_duration,
    ROUND(MAX(duration_minutes), 1) as max_duration,
    ROUND(COUNT(CASE WHEN duration_minutes < 5 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_under_5min,
    ROUND(COUNT(CASE WHEN duration_minutes < 10 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_under_10min,
    ROUND(COUNT(CASE WHEN duration_minutes >= 20 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_over_20min,
    ROUND(COUNT(CASE WHEN duration_minutes >= 30 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_over_30min,
    'Individual regime analysis' as regime_stability_rating
FROM regime_durations
WHERE duration_minutes IS NOT NULL 
  AND duration_minutes > 0 
  AND duration_minutes < 500
GROUP BY regime_state
ORDER BY avg_duration_minutes DESC;

-- Step 3: Compare Multiple Classifier Types
SELECT 
    '=== CROSS-CLASSIFIER REGIME STABILITY COMPARISON ===' as header;

-- Compare different classifier types
WITH all_classifier_durations AS (
    -- Hidden Markov
    SELECT 
        'Hidden Markov' as classifier_type,
        EXTRACT(EPOCH FROM (ts::timestamp - LAG(ts::timestamp) OVER (ORDER BY ts))) / 60 as duration_minutes
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet')
    WHERE strat = 'SPY_hidden_markov_grid_11_0001_05'
    
    UNION ALL
    
    -- Market Regime
    SELECT 
        'Market Regime' as classifier_type,
        EXTRACT(EPOCH FROM (ts::timestamp - LAG(ts::timestamp) OVER (ORDER BY ts))) / 60 as duration_minutes
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/market_regime_grid/SPY_market_regime_grid_00015_03.parquet')
    WHERE strat = 'SPY_market_regime_grid_00015_03'
    
    UNION ALL
    
    -- Volatility Momentum
    SELECT 
        'Volatility Momentum' as classifier_type,
        EXTRACT(EPOCH FROM (ts::timestamp - LAG(ts::timestamp) OVER (ORDER BY ts))) / 60 as duration_minutes
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_55_35.parquet')
    WHERE strat = 'SPY_volatility_momentum_grid_05_55_35'
)
SELECT 
    classifier_type,
    COUNT(*) as state_changes,
    ROUND(AVG(duration_minutes), 1) as avg_duration_minutes,
    ROUND(MEDIAN(duration_minutes), 1) as median_duration_minutes,
    ROUND(COUNT(CASE WHEN duration_minutes >= 20 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_meeting_target,
    CASE 
        WHEN AVG(duration_minutes) >= 20 THEN 'TARGET MET ✓'
        WHEN AVG(duration_minutes) >= 15 THEN 'CLOSE'
        WHEN AVG(duration_minutes) >= 10 THEN 'MODERATE'
        ELSE 'POOR'
    END as target_assessment
FROM all_classifier_durations
WHERE duration_minutes IS NOT NULL 
  AND duration_minutes > 0 
  AND duration_minutes < 500
GROUP BY classifier_type
ORDER BY avg_duration_minutes DESC;

-- Step 4: Final Assessment
SELECT 
    '=== FINAL ASSESSMENT: Did New Averaging Fix Regime Instability? ===' as header;

WITH final_assessment AS (
    SELECT 
        ROUND(AVG(EXTRACT(EPOCH FROM (ts::timestamp - LAG(ts::timestamp) OVER (ORDER BY ts))) / 60), 1) as avg_regime_duration,
        ROUND(COUNT(CASE WHEN EXTRACT(EPOCH FROM (ts::timestamp - LAG(ts::timestamp) OVER (ORDER BY ts))) / 60 >= 20 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_meeting_20min_target,
        COUNT(*) as total_regime_changes
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet')
    WHERE strat = 'SPY_hidden_markov_grid_11_0001_05'
      AND EXTRACT(EPOCH FROM (ts::timestamp - LAG(ts::timestamp) OVER (ORDER BY ts))) / 60 BETWEEN 0.5 AND 500
)
SELECT 
    'Previous Target: 20-30 minute regimes' as goal,
    avg_regime_duration || ' minutes' as actual_avg,
    pct_meeting_20min_target || '%' as success_rate,
    total_regime_changes as total_changes,
    CASE 
        WHEN avg_regime_duration >= 20 THEN 'SUCCESS: New averaging FIXED the instability!'
        WHEN avg_regime_duration >= 15 THEN 'PROGRESS: Better than before, but still needs work'
        WHEN avg_regime_duration >= 10 THEN 'INSUFFICIENT: Still too chaotic for trading'
        ELSE 'FAILED: Regimes still changing too frequently'
    END as conclusion
FROM final_assessment;