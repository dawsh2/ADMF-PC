-- Phase 1: Classifier Validation - New Regime Data Analysis
-- Execute the full analytics workflow with proper classifier data

PRAGMA memory_limit='3GB';
SET threads=4;

-- Step 1.1: State Distribution Analysis
-- Check if classifiers have reasonable distribution across states

WITH classifier_overview AS (
    -- Get overview of all classifier types and their data
    SELECT 
        'hidden_markov' as classifier_type,
        split_part(filename, '/', -1) as classifier_file,
        COUNT(*) as total_records,
        MIN(ts) as min_timestamp,
        MAX(ts) as max_timestamp
    FROM read_parquet('traces/SPY_1m/classifiers/hidden_markov_grid/*.parquet', filename=true)
    GROUP BY filename
    
    UNION ALL
    
    SELECT 
        'market_regime' as classifier_type,
        split_part(filename, '/', -1) as classifier_file,
        COUNT(*) as total_records,
        MIN(ts) as min_timestamp,
        MAX(ts) as max_timestamp
    FROM read_parquet('traces/SPY_1m/classifiers/market_regime_grid/*.parquet', filename=true)
    GROUP BY filename
    
    UNION ALL
    
    SELECT 
        'microstructure' as classifier_type,
        split_part(filename, '/', -1) as classifier_file,
        COUNT(*) as total_records,
        MIN(ts) as min_timestamp,
        MAX(ts) as max_timestamp
    FROM read_parquet('traces/SPY_1m/classifiers/microstructure_grid/*.parquet', filename=true)
    GROUP BY filename
    
    UNION ALL
    
    SELECT 
        'multi_timeframe_trend' as classifier_type,
        split_part(filename, '/', -1) as classifier_file,
        COUNT(*) as total_records,
        MIN(ts) as min_timestamp,
        MAX(ts) as max_timestamp
    FROM read_parquet('traces/SPY_1m/classifiers/multi_timeframe_trend_grid/*.parquet', filename=true)
    GROUP BY filename
    
    UNION ALL
    
    SELECT 
        'volatility_momentum' as classifier_type,
        split_part(filename, '/', -1) as classifier_file,
        COUNT(*) as total_records,
        MIN(ts) as min_timestamp,
        MAX(ts) as max_timestamp
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet', filename=true)
    GROUP BY filename
)
SELECT 
    '=== CLASSIFIER DATA OVERVIEW ===' as header;

SELECT 
    classifier_type,
    COUNT(*) as num_classifiers,
    AVG(total_records) as avg_records_per_classifier,
    MIN(min_timestamp) as earliest_data,
    MAX(max_timestamp) as latest_data
FROM classifier_overview
GROUP BY classifier_type
ORDER BY classifier_type;

-- Sample one classifier from each type to check data structure
SELECT 
    '=== SAMPLE CLASSIFIER DATA STRUCTURE ===' as header;

-- Sample Hidden Markov data
WITH sample_hm AS (
    SELECT 'hidden_markov' as type, ts, val, conf
    FROM read_parquet('traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet')
    ORDER BY ts LIMIT 10
)
SELECT * FROM sample_hm

UNION ALL

-- Sample Market Regime data  
SELECT 'market_regime' as type, ts, val, conf
FROM read_parquet('traces/SPY_1m/classifiers/market_regime_grid/SPY_market_regime_grid_00015_03.parquet')
ORDER BY ts LIMIT 10

UNION ALL

-- Sample Microstructure data
SELECT 'microstructure' as type, ts, val, conf
FROM read_parquet('traces/SPY_1m/classifiers/microstructure_grid/SPY_microstructure_grid_00015_00003.parquet')
ORDER BY ts LIMIT 10;

-- Analyze state distributions for each classifier type
WITH state_distributions AS (
    -- Hidden Markov state distribution
    SELECT 
        'hidden_markov' as classifier_type,
        split_part(filename, '/', -1) as classifier_id,
        val as regime_state,
        COUNT(*) as occurrences,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY filename) as pct_time
    FROM read_parquet('traces/SPY_1m/classifiers/hidden_markov_grid/*.parquet', filename=true)
    GROUP BY filename, val
    
    UNION ALL
    
    -- Market Regime state distribution
    SELECT 
        'market_regime' as classifier_type,
        split_part(filename, '/', -1) as classifier_id,
        val as regime_state,
        COUNT(*) as occurrences,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY filename) as pct_time
    FROM read_parquet('traces/SPY_1m/classifiers/market_regime_grid/*.parquet', filename=true)
    GROUP BY filename, val
    
    UNION ALL
    
    -- Microstructure state distribution
    SELECT 
        'microstructure' as classifier_type,
        split_part(filename, '/', -1) as classifier_id,
        val as regime_state,
        COUNT(*) as occurrences,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY filename) as pct_time
    FROM read_parquet('traces/SPY_1m/classifiers/microstructure_grid/*.parquet', filename=true)
    GROUP BY filename, val
    
    UNION ALL
    
    -- Multi-timeframe trend state distribution
    SELECT 
        'multi_timeframe_trend' as classifier_type,
        split_part(filename, '/', -1) as classifier_id,
        val as regime_state,
        COUNT(*) as occurrences,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY filename) as pct_time
    FROM read_parquet('traces/SPY_1m/classifiers/multi_timeframe_trend_grid/*.parquet', filename=true)
    GROUP BY filename, val
    
    UNION ALL
    
    -- Volatility momentum state distribution
    SELECT 
        'volatility_momentum' as classifier_type,
        split_part(filename, '/', -1) as classifier_id,
        val as regime_state,
        COUNT(*) as occurrences,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY filename) as pct_time
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet', filename=true)
    GROUP BY filename, val
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
    AVG(pct_time) as avg_state_percentage,
    STDDEV(pct_time) as state_percentage_std,
    COUNT(CASE WHEN distribution_flag = 'RARE_STATE' THEN 1 END) as rare_states_count,
    COUNT(CASE WHEN distribution_flag = 'DOMINANT_STATE' THEN 1 END) as dominant_states_count,
    COUNT(CASE WHEN distribution_flag = 'BALANCED' THEN 1 END) as balanced_states_count
FROM distribution_quality
GROUP BY classifier_type
ORDER BY state_percentage_std ASC;

-- Detailed view of problematic classifiers
SELECT 
    '=== PROBLEMATIC CLASSIFIERS (Unbalanced States) ===' as header;

SELECT 
    classifier_type,
    classifier_id,
    regime_state,
    ROUND(pct_time, 1) as pct_time,
    distribution_flag
FROM distribution_quality
WHERE distribution_flag IN ('RARE_STATE', 'DOMINANT_STATE')
ORDER BY classifier_type, classifier_id, pct_time DESC;

-- Best balanced classifiers per type
WITH balanced_classifiers AS (
    SELECT 
        classifier_type,
        classifier_id,
        STDDEV(pct_time) as balance_score
    FROM distribution_quality
    GROUP BY classifier_type, classifier_id
    HAVING COUNT(DISTINCT regime_state) >= 2  -- At least 2 states
)
SELECT 
    '=== BEST BALANCED CLASSIFIERS BY TYPE ===' as header;

SELECT 
    classifier_type,
    classifier_id,
    ROUND(balance_score, 2) as balance_score,
    CASE 
        WHEN balance_score < 5 THEN 'EXCELLENT'
        WHEN balance_score < 10 THEN 'GOOD'
        WHEN balance_score < 20 THEN 'FAIR'
        ELSE 'POOR'
    END as balance_quality
FROM balanced_classifiers
WHERE balance_score < 25  -- Filter out severely unbalanced
ORDER BY classifier_type, balance_score ASC;