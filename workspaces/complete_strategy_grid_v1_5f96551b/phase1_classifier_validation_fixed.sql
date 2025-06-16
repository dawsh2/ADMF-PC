-- Phase 1: Classifier Validation - FIXED for actual data structure
-- Execute the full analytics workflow with proper classifier data

PRAGMA memory_limit='3GB';
SET threads=4;

-- Step 1.1: State Distribution Analysis
-- Check if classifiers have reasonable distribution across states

WITH classifier_overview AS (
    -- Get overview of all classifier types and their data
    SELECT 
        'hidden_markov' as classifier_type,
        strat as classifier_id,
        COUNT(*) as total_records,
        MIN(ts::timestamp) as min_timestamp,
        MAX(ts::timestamp) as max_timestamp
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/hidden_markov_grid/*.parquet')
    GROUP BY strat
    
    UNION ALL
    
    SELECT 
        'market_regime' as classifier_type,
        strat as classifier_id,
        COUNT(*) as total_records,
        MIN(ts::timestamp) as min_timestamp,
        MAX(ts::timestamp) as max_timestamp
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/market_regime_grid/*.parquet')
    GROUP BY strat
    
    UNION ALL
    
    SELECT 
        'microstructure' as classifier_type,
        strat as classifier_id,
        COUNT(*) as total_records,
        MIN(ts::timestamp) as min_timestamp,
        MAX(ts::timestamp) as max_timestamp
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/microstructure_grid/*.parquet')
    GROUP BY strat
    
    UNION ALL
    
    SELECT 
        'multi_timeframe_trend' as classifier_type,
        strat as classifier_id,
        COUNT(*) as total_records,
        MIN(ts::timestamp) as min_timestamp,
        MAX(ts::timestamp) as max_timestamp
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/multi_timeframe_trend_grid/*.parquet')
    GROUP BY strat
    
    UNION ALL
    
    SELECT 
        'volatility_momentum' as classifier_type,
        strat as classifier_id,
        COUNT(*) as total_records,
        MIN(ts::timestamp) as min_timestamp,
        MAX(ts::timestamp) as max_timestamp
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet')
    GROUP BY strat
)
SELECT 
    '=== CLASSIFIER DATA OVERVIEW ===' as header;

SELECT 
    classifier_type,
    COUNT(*) as num_classifiers,
    ROUND(AVG(total_records), 0) as avg_records_per_classifier,
    MIN(min_timestamp) as earliest_data,
    MAX(max_timestamp) as latest_data
FROM classifier_overview
GROUP BY classifier_type
ORDER BY classifier_type;

-- Analyze state distributions for each classifier type
WITH state_distributions AS (
    -- Hidden Markov state distribution
    SELECT 
        'hidden_markov' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as occurrences,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY strat) as pct_time
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/hidden_markov_grid/*.parquet')
    GROUP BY strat, val
    
    UNION ALL
    
    -- Market Regime state distribution
    SELECT 
        'market_regime' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as occurrences,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY strat) as pct_time
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/market_regime_grid/*.parquet')
    GROUP BY strat, val
    
    UNION ALL
    
    -- Microstructure state distribution
    SELECT 
        'microstructure' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as occurrences,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY strat) as pct_time
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/microstructure_grid/*.parquet')
    GROUP BY strat, val
    
    UNION ALL
    
    -- Multi-timeframe trend state distribution
    SELECT 
        'multi_timeframe_trend' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as occurrences,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY strat) as pct_time
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/multi_timeframe_trend_grid/*.parquet')
    GROUP BY strat, val
    
    UNION ALL
    
    -- Volatility momentum state distribution
    SELECT 
        'volatility_momentum' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as occurrences,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY strat) as pct_time
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet')
    GROUP BY strat, val
),
distribution_quality AS (
    SELECT 
        classifier_type,
        classifier_id,
        regime_state,
        pct_time,
        CASE 
            WHEN pct_time < 5 THEN 'RARE_STATE'
            WHEN pct_time > 60 THEN 'DOMINANT_STATE'
            ELSE 'BALANCED'
        END as distribution_flag
    FROM state_distributions
)
SELECT 
    '=== STATE DISTRIBUTION ANALYSIS ===' as header;

-- Summary by classifier type
SELECT 
    classifier_type,
    COUNT(DISTINCT classifier_id) as num_classifiers,
    COUNT(DISTINCT regime_state) as unique_states_across_all,
    ROUND(AVG(pct_time), 1) as avg_state_percentage,
    ROUND(STDDEV(pct_time), 1) as state_percentage_std,
    COUNT(CASE WHEN distribution_flag = 'RARE_STATE' THEN 1 END) as rare_states_count,
    COUNT(CASE WHEN distribution_flag = 'DOMINANT_STATE' THEN 1 END) as dominant_states_count,
    COUNT(CASE WHEN distribution_flag = 'BALANCED' THEN 1 END) as balanced_states_count
FROM distribution_quality
GROUP BY classifier_type
ORDER BY state_percentage_std ASC;

-- Best balanced classifiers per type (for regime persistence analysis)
WITH balanced_classifiers AS (
    SELECT 
        classifier_type,
        classifier_id,
        ROUND(STDDEV(pct_time), 2) as balance_score,
        COUNT(DISTINCT regime_state) as num_states
    FROM distribution_quality
    GROUP BY classifier_type, classifier_id
    HAVING COUNT(DISTINCT regime_state) >= 2  -- At least 2 states
)
SELECT 
    '=== BEST BALANCED CLASSIFIERS BY TYPE ===' as header;

SELECT 
    classifier_type,
    classifier_id,
    balance_score,
    num_states,
    CASE 
        WHEN balance_score < 5 THEN 'EXCELLENT'
        WHEN balance_score < 10 THEN 'GOOD'
        WHEN balance_score < 20 THEN 'FAIR'
        ELSE 'POOR'
    END as balance_quality
FROM balanced_classifiers
WHERE balance_score < 30  -- Filter out severely unbalanced
ORDER BY classifier_type, balance_score ASC;

-- Check for expected regime persistence improvement (20-30m averages)
-- Compare regime durations to see if they're more stable than the original 5-6 min changes
WITH regime_transitions AS (
    -- Sample from one classifier of each type to check regime persistence
    SELECT 
        'hidden_markov' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        ts::timestamp as timestamp,
        LAG(val) OVER (PARTITION BY strat ORDER BY ts::timestamp) as prev_regime,
        LAG(ts::timestamp) OVER (PARTITION BY strat ORDER BY ts::timestamp) as prev_timestamp
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet')
    WHERE strat = 'SPY_hidden_markov_grid_11_0001_05'
    
    UNION ALL
    
    SELECT 
        'market_regime' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        ts::timestamp as timestamp,
        LAG(val) OVER (PARTITION BY strat ORDER BY ts::timestamp) as prev_regime,
        LAG(ts::timestamp) OVER (PARTITION BY strat ORDER BY ts::timestamp) as prev_timestamp
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/market_regime_grid/SPY_market_regime_grid_00015_03.parquet')
    WHERE strat = 'SPY_market_regime_grid_00015_03'
),
regime_durations AS (
    SELECT 
        classifier_type,
        classifier_id,
        regime_state,
        timestamp,
        EXTRACT(EPOCH FROM (timestamp - prev_timestamp)) / 60 as duration_minutes
    FROM regime_transitions
    WHERE prev_regime IS NOT NULL 
      AND prev_regime != regime_state  -- Regime change points
)
SELECT 
    '=== REGIME PERSISTENCE ANALYSIS (Sample) ===' as header;

SELECT 
    classifier_type,
    COUNT(*) as regime_changes,
    ROUND(AVG(duration_minutes), 1) as avg_regime_duration_min,
    ROUND(MEDIAN(duration_minutes), 1) as median_regime_duration_min,
    ROUND(MIN(duration_minutes), 1) as min_duration_min,
    ROUND(MAX(duration_minutes), 1) as max_duration_min,
    ROUND(COUNT(CASE WHEN duration_minutes < 10 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_short_regimes_under_10min,
    ROUND(COUNT(CASE WHEN duration_minutes > 30 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_long_regimes_over_30min,
    CASE 
        WHEN AVG(duration_minutes) > 20 THEN 'STABLE (Target met!)'
        WHEN AVG(duration_minutes) > 10 THEN 'MODERATE'
        ELSE 'UNSTABLE (Too frequent changes)'
    END as stability_assessment
FROM regime_durations
WHERE duration_minutes < 1000  -- Filter outliers
GROUP BY classifier_type
ORDER BY avg_regime_duration_min DESC;