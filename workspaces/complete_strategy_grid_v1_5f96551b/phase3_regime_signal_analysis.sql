-- Phase 3: Regime-based signal analysis using available data
-- Working with classifier regimes and strategy signals directly

PRAGMA memory_limit='3GB';
SET threads=4;

SELECT 
    '=== PHASE 3: REGIME-BASED SIGNAL ANALYSIS ===' as header;

-- Key insight from our classifier analysis
SELECT 
    '=== KEY FINDING: ALL 3 TOP CLASSIFIERS ARE IDENTICAL ===' as finding;

SELECT 
    'Our top 3 classifiers produce exactly the same regime classifications.' as insight_1;
SELECT 
    'This means we can focus on just ONE classifier for Phase 3 analysis.' as insight_2;
SELECT 
    'Using SPY_volatility_momentum_grid_05_65_40 as our representative classifier.' as insight_3;

-- Regime distribution in our analysis period
SELECT 
    '=== REGIME DISTRIBUTION (MAR 26 - APR 2, 2024) ===' as section_header;

SELECT 
    c.val as regime_state,
    COUNT(*) as minutes_in_regime,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage,
    MIN(c.ts::timestamp) as first_occurrence,
    MAX(c.ts::timestamp) as last_occurrence,
    ROUND(COUNT(*) / 60.0, 1) as hours_in_regime
FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
WHERE c.ts::timestamp >= '2024-03-26 13:30:00'
  AND c.ts::timestamp <= '2024-04-02 20:00:00'
GROUP BY c.val
ORDER BY COUNT(*) DESC;

-- Check what strategy signal files we have available
SELECT 
    '=== AVAILABLE STRATEGY SIGNAL FILES ===' as section_header;

SELECT 
    strategy_type,
    COUNT(*) as strategy_count,
    MIN(created_at) as earliest_strategy,
    MAX(created_at) as latest_strategy
FROM strategies
GROUP BY strategy_type
ORDER BY strategy_count DESC;

-- Sample a few strategies for regime analysis
SELECT 
    '=== SAMPLE STRATEGIES FOR REGIME ANALYSIS ===' as section_header;

SELECT 
    strategy_id,
    strategy_type,
    strategy_name,
    signal_file_path
FROM strategies
WHERE strategy_type IN ('macd_crossover', 'ema_crossover', 'rsi_threshold')
ORDER BY strategy_type, strategy_id
LIMIT 15;

-- For one sample strategy, let's analyze signal timing vs regimes
SELECT 
    '=== SIGNAL TIMING ANALYSIS: SAMPLE MACD STRATEGY ===' as section_header;

WITH sample_strategy AS (
    SELECT signal_file_path, strategy_id, strategy_name
    FROM strategies 
    WHERE strategy_type = 'macd_crossover'
    LIMIT 1
),
strategy_signals AS (
    SELECT 
        s.ts::timestamp as signal_time,
        s.val as signal_value,
        ss.strategy_id,
        ss.strategy_name
    FROM sample_strategy ss,
         read_parquet(ss.signal_file_path) s
    WHERE s.ts::timestamp >= '2024-03-26 13:30:00'
      AND s.ts::timestamp <= '2024-04-02 20:00:00'
),
regime_at_signal_time AS (
    SELECT 
        ss.*,
        c.val as regime_state
    FROM strategy_signals ss
    LEFT JOIN read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
        ON ss.signal_time = c.ts::timestamp
)
SELECT 
    strategy_id,
    strategy_name,
    regime_state,
    signal_value,
    COUNT(*) as signal_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY strategy_id), 1) as signal_percentage
FROM regime_at_signal_time
WHERE regime_state IS NOT NULL
GROUP BY strategy_id, strategy_name, regime_state, signal_value
ORDER BY regime_state, signal_value;

-- Time-of-day analysis for regimes
SELECT 
    '=== REGIME TIMING PATTERNS ===' as section_header;

SELECT 
    c.val as regime_state,
    EXTRACT(HOUR FROM (c.ts::timestamp - INTERVAL 4 HOUR)) as hour_est,  -- Apply timezone correction
    COUNT(*) as occurrences,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY c.val), 1) as hourly_percentage
FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
WHERE c.ts::timestamp >= '2024-03-26 13:30:00'
  AND c.ts::timestamp <= '2024-04-02 20:00:00'
  AND EXTRACT(HOUR FROM (c.ts::timestamp - INTERVAL 4 HOUR)) BETWEEN 9 AND 16  -- Market hours EST
GROUP BY c.val, EXTRACT(HOUR FROM (c.ts::timestamp - INTERVAL 4 HOUR))
ORDER BY regime_state, hour_est;

-- Regime transition analysis
SELECT 
    '=== REGIME TRANSITION PATTERNS ===' as section_header;

WITH regime_transitions AS (
    SELECT 
        c.ts::timestamp as timestamp,
        c.val as current_regime,
        LAG(c.val) OVER (ORDER BY c.ts::timestamp) as previous_regime,
        EXTRACT(HOUR FROM (c.ts::timestamp - INTERVAL 4 HOUR)) as hour_est
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
    WHERE c.ts::timestamp >= '2024-03-26 13:30:00'
      AND c.ts::timestamp <= '2024-04-02 20:00:00'
)
SELECT 
    previous_regime,
    current_regime,
    COUNT(*) as transition_count,
    ROUND(AVG(hour_est), 1) as avg_transition_hour_est,
    MIN(timestamp) as first_transition,
    MAX(timestamp) as last_transition
FROM regime_transitions
WHERE previous_regime IS NOT NULL 
  AND previous_regime != current_regime  -- Only actual transitions
GROUP BY previous_regime, current_regime
ORDER BY transition_count DESC;