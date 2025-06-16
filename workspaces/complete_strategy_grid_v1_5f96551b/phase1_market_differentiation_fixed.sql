-- Phase 1.3: Market Differentiation Test - Fixed Version
-- Test if ~10 minute regime changes predict different market conditions
-- This is the KEY test - do regimes actually matter for trading?

PRAGMA memory_limit='3GB';
SET threads=4;

-- First, let's examine what classifier data we have
SELECT 
    '=== AVAILABLE CLASSIFIER DATA ===' as header;

-- Get summary of classifier data
WITH classifier_summary AS (
    SELECT 
        'hidden_markov' as classifier_type,
        'SPY_hidden_markov_grid_11_0001_05' as classifier_id,
        COUNT(*) as total_points,
        MIN(ts::timestamp) as start_date,
        MAX(ts::timestamp) as end_date,
        COUNT(DISTINCT val) as unique_states
    FROM read_parquet('traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet')
    WHERE strat = 'SPY_hidden_markov_grid_11_0001_05'
    
    UNION ALL
    
    SELECT 
        'market_regime' as classifier_type,
        'SPY_market_regime_grid_00015_03' as classifier_id,
        COUNT(*) as total_points,
        MIN(ts::timestamp) as start_date,
        MAX(ts::timestamp) as end_date,
        COUNT(DISTINCT val) as unique_states
    FROM read_parquet('traces/SPY_1m/classifiers/market_regime_grid/SPY_market_regime_grid_00015_03.parquet')
    WHERE strat = 'SPY_market_regime_grid_00015_03'
    
    UNION ALL
    
    SELECT 
        'volatility_momentum' as classifier_type,
        'SPY_volatility_momentum_grid_05_55_35' as classifier_id,
        COUNT(*) as total_points,
        MIN(ts::timestamp) as start_date,
        MAX(ts::timestamp) as end_date,
        COUNT(DISTINCT val) as unique_states
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_55_35.parquet')
    WHERE strat = 'SPY_volatility_momentum_grid_05_55_35'
)
SELECT * FROM classifier_summary;

-- Since we don't have market data with prices, let's analyze the regime patterns themselves
SELECT 
    '=== REGIME STATE ANALYSIS ===' as header;

-- Analyze regime states for each classifier
WITH regime_analysis AS (
    SELECT 
        'hidden_markov' as classifier_type,
        val as regime_state,
        COUNT(*) as frequency,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
    FROM read_parquet('traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet')
    WHERE strat = 'SPY_hidden_markov_grid_11_0001_05'
    GROUP BY val
    
    UNION ALL
    
    SELECT 
        'market_regime' as classifier_type,
        val as regime_state,
        COUNT(*) as frequency,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
    FROM read_parquet('traces/SPY_1m/classifiers/market_regime_grid/SPY_market_regime_grid_00015_03.parquet')
    WHERE strat = 'SPY_market_regime_grid_00015_03'
    GROUP BY val
    
    UNION ALL
    
    SELECT 
        'volatility_momentum' as classifier_type,
        val as regime_state,
        COUNT(*) as frequency,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_55_35.parquet')
    WHERE strat = 'SPY_volatility_momentum_grid_05_55_35'
    GROUP BY val
)
SELECT 
    classifier_type,
    regime_state,
    frequency,
    percentage
FROM regime_analysis
ORDER BY classifier_type, frequency DESC;

-- Analyze regime transitions and persistence
SELECT 
    '=== REGIME TRANSITION ANALYSIS ===' as header;

-- Hidden Markov transitions
WITH regime_transitions AS (
    SELECT 
        'hidden_markov' as classifier_type,
        ts::timestamp as timestamp,
        val as regime_state,
        LAG(val) OVER (ORDER BY ts::timestamp) as prev_regime,
        -- Calculate time between regime changes
        EXTRACT(EPOCH FROM (ts::timestamp - LAG(ts::timestamp) OVER (ORDER BY ts::timestamp))) / 60 as minutes_since_last
    FROM read_parquet('traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet')
    WHERE strat = 'SPY_hidden_markov_grid_11_0001_05'
    
    UNION ALL
    
    SELECT 
        'market_regime' as classifier_type,
        ts::timestamp as timestamp,
        val as regime_state,
        LAG(val) OVER (ORDER BY ts::timestamp) as prev_regime,
        EXTRACT(EPOCH FROM (ts::timestamp - LAG(ts::timestamp) OVER (ORDER BY ts::timestamp))) / 60 as minutes_since_last
    FROM read_parquet('traces/SPY_1m/classifiers/market_regime_grid/SPY_market_regime_grid_00015_03.parquet')
    WHERE strat = 'SPY_market_regime_grid_00015_03'
    
    UNION ALL
    
    SELECT 
        'volatility_momentum' as classifier_type,
        ts::timestamp as timestamp,
        val as regime_state,
        LAG(val) OVER (ORDER BY ts::timestamp) as prev_regime,
        EXTRACT(EPOCH FROM (ts::timestamp - LAG(ts::timestamp) OVER (ORDER BY ts::timestamp))) / 60 as minutes_since_last
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_55_35.parquet')
    WHERE strat = 'SPY_volatility_momentum_grid_05_55_35'
),
transition_stats AS (
    SELECT 
        classifier_type,
        prev_regime || ' → ' || regime_state as transition_type,
        COUNT(*) as num_transitions,
        ROUND(AVG(minutes_since_last), 1) as avg_minutes_between,
        ROUND(MIN(minutes_since_last), 1) as min_minutes_between,
        ROUND(MAX(minutes_since_last), 1) as max_minutes_between
    FROM regime_transitions
    WHERE prev_regime IS NOT NULL 
      AND prev_regime != regime_state  -- Only actual transitions
      AND minutes_since_last IS NOT NULL
      AND minutes_since_last > 0
    GROUP BY classifier_type, transition_type
    HAVING COUNT(*) > 5  -- Sufficient transitions
)
SELECT 
    classifier_type,
    transition_type,
    num_transitions,
    avg_minutes_between,
    min_minutes_between,
    max_minutes_between,
    CASE 
        WHEN avg_minutes_between < 10 THEN 'HIGH_FREQUENCY'
        WHEN avg_minutes_between < 30 THEN 'MODERATE_FREQUENCY'
        ELSE 'LOW_FREQUENCY'
    END as transition_frequency_class
FROM transition_stats
ORDER BY classifier_type, num_transitions DESC;

-- Analyze regime persistence (how long each regime lasts)
SELECT 
    '=== REGIME PERSISTENCE ANALYSIS ===' as header;

WITH regime_persistence AS (
    SELECT 
        'hidden_markov' as classifier_type,
        val as regime_state,
        ts::timestamp as timestamp,
        -- Calculate duration of each regime episode
        LEAD(ts::timestamp) OVER (ORDER BY ts::timestamp) as next_timestamp
    FROM read_parquet('traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet')
    WHERE strat = 'SPY_hidden_markov_grid_11_0001_05'
    
    UNION ALL
    
    SELECT 
        'market_regime' as classifier_type,
        val as regime_state,
        ts::timestamp as timestamp,
        LEAD(ts::timestamp) OVER (ORDER BY ts::timestamp) as next_timestamp
    FROM read_parquet('traces/SPY_1m/classifiers/market_regime_grid/SPY_market_regime_grid_00015_03.parquet')
    WHERE strat = 'SPY_market_regime_grid_00015_03'
    
    UNION ALL
    
    SELECT 
        'volatility_momentum' as classifier_type,
        val as regime_state,
        ts::timestamp as timestamp,
        LEAD(ts::timestamp) OVER (ORDER BY ts::timestamp) as next_timestamp
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_55_35.parquet')
    WHERE strat = 'SPY_volatility_momentum_grid_05_55_35'
),
persistence_stats AS (
    SELECT 
        classifier_type,
        regime_state,
        COUNT(*) as total_occurrences,
        ROUND(AVG(EXTRACT(EPOCH FROM (next_timestamp - timestamp)) / 60), 1) as avg_duration_minutes,
        ROUND(MIN(EXTRACT(EPOCH FROM (next_timestamp - timestamp)) / 60), 1) as min_duration_minutes,
        ROUND(MAX(EXTRACT(EPOCH FROM (next_timestamp - timestamp)) / 60), 1) as max_duration_minutes
    FROM regime_persistence
    WHERE next_timestamp IS NOT NULL
    GROUP BY classifier_type, regime_state
)
SELECT 
    classifier_type,
    regime_state,
    total_occurrences,
    avg_duration_minutes,
    min_duration_minutes,
    max_duration_minutes,
    CASE 
        WHEN avg_duration_minutes < 10 THEN 'SHORT_DURATION'
        WHEN avg_duration_minutes < 30 THEN 'MEDIUM_DURATION'
        ELSE 'LONG_DURATION'
    END as duration_class
FROM persistence_stats
ORDER BY classifier_type, total_occurrences DESC;

-- Time-of-day analysis to see if regimes are intraday patterns
SELECT 
    '=== INTRADAY REGIME PATTERNS ===' as header;

WITH intraday_patterns AS (
    SELECT 
        'hidden_markov' as classifier_type,
        val as regime_state,
        EXTRACT(hour FROM ts::timestamp) as hour_of_day,
        COUNT(*) as frequency
    FROM read_parquet('traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet')
    WHERE strat = 'SPY_hidden_markov_grid_11_0001_05'
      AND EXTRACT(hour FROM ts::timestamp) BETWEEN 9 AND 15  -- Market hours
    GROUP BY val, EXTRACT(hour FROM ts::timestamp)
    
    UNION ALL
    
    SELECT 
        'market_regime' as classifier_type,
        val as regime_state,
        EXTRACT(hour FROM ts::timestamp) as hour_of_day,
        COUNT(*) as frequency
    FROM read_parquet('traces/SPY_1m/classifiers/market_regime_grid/SPY_market_regime_grid_00015_03.parquet')
    WHERE strat = 'SPY_market_regime_grid_00015_03'
      AND EXTRACT(hour FROM ts::timestamp) BETWEEN 9 AND 15
    GROUP BY val, EXTRACT(hour FROM ts::timestamp)
    
    UNION ALL
    
    SELECT 
        'volatility_momentum' as classifier_type,
        val as regime_state,
        EXTRACT(hour FROM ts::timestamp) as hour_of_day,
        COUNT(*) as frequency
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_55_35.parquet')
    WHERE strat = 'SPY_volatility_momentum_grid_05_55_35'
      AND EXTRACT(hour FROM ts::timestamp) BETWEEN 9 AND 15
    GROUP BY val, EXTRACT(hour FROM ts::timestamp)
)
SELECT 
    classifier_type,
    regime_state,
    hour_of_day,
    frequency,
    ROUND(frequency * 100.0 / SUM(frequency) OVER (PARTITION BY classifier_type, regime_state), 2) as pct_of_regime,
    CASE 
        WHEN hour_of_day IN (9, 10) THEN 'OPEN'
        WHEN hour_of_day IN (11, 12) THEN 'MORNING'
        WHEN hour_of_day IN (13, 14) THEN 'MIDDAY'
        WHEN hour_of_day = 15 THEN 'CLOSE'
    END as session_period
FROM intraday_patterns
ORDER BY classifier_type, regime_state, hour_of_day;

-- Final assessment: Do these classifiers show meaningful regime differentiation?
SELECT 
    '=== CLASSIFIER DIFFERENTIATION ASSESSMENT ===' as header;

WITH final_assessment AS (
    SELECT 
        'hidden_markov' as classifier_type,
        COUNT(DISTINCT val) as num_distinct_regimes,
        COUNT(*) as total_signals,
        ROUND(COUNT(*) / COUNT(DISTINCT val), 0) as avg_signals_per_regime
    FROM read_parquet('traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet')
    WHERE strat = 'SPY_hidden_markov_grid_11_0001_05'
    
    UNION ALL
    
    SELECT 
        'market_regime' as classifier_type,
        COUNT(DISTINCT val) as num_distinct_regimes,
        COUNT(*) as total_signals,
        ROUND(COUNT(*) / COUNT(DISTINCT val), 0) as avg_signals_per_regime
    FROM read_parquet('traces/SPY_1m/classifiers/market_regime_grid/SPY_market_regime_grid_00015_03.parquet')
    WHERE strat = 'SPY_market_regime_grid_00015_03'
    
    UNION ALL
    
    SELECT 
        'volatility_momentum' as classifier_type,
        COUNT(DISTINCT val) as num_distinct_regimes,
        COUNT(*) as total_signals,
        ROUND(COUNT(*) / COUNT(DISTINCT val), 0) as avg_signals_per_regime
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_55_35.parquet')
    WHERE strat = 'SPY_volatility_momentum_grid_05_55_35'
)
SELECT 
    classifier_type,
    num_distinct_regimes,
    total_signals,
    avg_signals_per_regime,
    CASE 
        WHEN num_distinct_regimes > 5 THEN 'HIGH_DIFFERENTIATION'
        WHEN num_distinct_regimes > 3 THEN 'MODERATE_DIFFERENTIATION'
        WHEN num_distinct_regimes > 2 THEN 'LOW_DIFFERENTIATION'
        ELSE 'NO_DIFFERENTIATION'
    END as differentiation_level,
    CASE 
        WHEN avg_signals_per_regime > 100 THEN 'SUFFICIENT_SAMPLE'
        WHEN avg_signals_per_regime > 50 THEN 'MODERATE_SAMPLE'
        ELSE 'INSUFFICIENT_SAMPLE'
    END as sample_adequacy
FROM final_assessment
ORDER BY num_distinct_regimes DESC;