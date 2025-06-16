-- Phase 2: Rank top 18 Volatility Momentum classifiers based on comprehensive quality scores
-- Focus on our excellent classifiers identified in Phase 1

PRAGMA memory_limit='3GB';
SET threads=4;

-- Define our top 18 excellent classifiers
WITH excellent_classifiers AS (
    SELECT classifier_id FROM (VALUES 
        ('SPY_volatility_momentum_grid_05_55_35'),
        ('SPY_volatility_momentum_grid_05_55_40'), 
        ('SPY_volatility_momentum_grid_05_55_45'),
        ('SPY_volatility_momentum_grid_05_60_35'),
        ('SPY_volatility_momentum_grid_05_60_40'),
        ('SPY_volatility_momentum_grid_05_60_45'),
        ('SPY_volatility_momentum_grid_05_65_35'),
        ('SPY_volatility_momentum_grid_05_65_40'),
        ('SPY_volatility_momentum_grid_05_65_45'),
        ('SPY_volatility_momentum_grid_08_55_35'),
        ('SPY_volatility_momentum_grid_08_55_40'),
        ('SPY_volatility_momentum_grid_08_55_45'),
        ('SPY_volatility_momentum_grid_08_60_35'),
        ('SPY_volatility_momentum_grid_08_60_40'),
        ('SPY_volatility_momentum_grid_08_60_45'),
        ('SPY_volatility_momentum_grid_08_65_35'),
        ('SPY_volatility_momentum_grid_08_65_40'),
        ('SPY_volatility_momentum_grid_08_65_45')
    ) AS t(classifier_id)
),

-- Get classifier sparse data with forward-fill
classifier_sparse AS (
    SELECT 
        c.ts::timestamp as timestamp,
        c.strat as classifier_id,
        c.val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet') c
    INNER JOIN excellent_classifiers ec ON c.strat = ec.classifier_id
),

-- Forward-fill classifier states over full market data timeline
classifier_full_series AS (
    SELECT 
        m.timestamp,
        ec.classifier_id,
        -- Forward-fill the regime state using LAST_VALUE with proper window
        LAST_VALUE(cs.regime_state IGNORE NULLS) OVER (
            PARTITION BY ec.classifier_id 
            ORDER BY m.timestamp 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as current_regime
    FROM analytics.market_data m
    CROSS JOIN excellent_classifiers ec
    LEFT JOIN classifier_sparse cs ON m.timestamp = cs.timestamp AND ec.classifier_id = cs.classifier_id
    WHERE m.timestamp >= '2024-03-26 13:30:00'  -- Start of data availability
      AND m.timestamp <= '2024-04-02 20:00:00'  -- End of data 
),

-- Calculate regime transitions and persistence
regime_analysis AS (
    SELECT 
        classifier_id,
        current_regime,
        timestamp,
        LAG(current_regime) OVER (PARTITION BY classifier_id ORDER BY timestamp) as prev_regime,
        LAG(timestamp) OVER (PARTITION BY classifier_id ORDER BY timestamp) as prev_timestamp,
        -- Identify regime changes
        CASE WHEN current_regime != LAG(current_regime) OVER (PARTITION BY classifier_id ORDER BY timestamp) 
             THEN 1 ELSE 0 END as regime_change
    FROM classifier_full_series
    WHERE current_regime IS NOT NULL
),

-- Calculate regime durations
regime_durations AS (
    SELECT 
        classifier_id,
        current_regime,
        timestamp,
        prev_timestamp,
        CASE WHEN regime_change = 1 
             THEN EXTRACT(EPOCH FROM (timestamp - prev_timestamp))/60.0  -- Minutes
             ELSE NULL END as regime_duration_minutes
    FROM regime_analysis
    WHERE regime_change = 1
),

-- Calculate state balance and persistence metrics
classifier_quality_metrics AS (
    SELECT 
        cfs.classifier_id,
        
        -- State distribution balance
        COUNT(*) as total_observations,
        COUNT(DISTINCT current_regime) as num_states,
        
        -- Calculate state percentages
        COUNT(CASE WHEN current_regime = 'high_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) as high_vol_bearish_pct,
        COUNT(CASE WHEN current_regime = 'high_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) as high_vol_bullish_pct,
        COUNT(CASE WHEN current_regime = 'low_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) as low_vol_bearish_pct,
        COUNT(CASE WHEN current_regime = 'low_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) as low_vol_bullish_pct,
        COUNT(CASE WHEN current_regime = 'neutral' THEN 1 END) * 100.0 / COUNT(*) as neutral_pct,
        
        -- State balance standard deviation (lower = more balanced)
        STDDEV(CASE 
            WHEN current_regime = 'high_vol_bearish' THEN COUNT(CASE WHEN current_regime = 'high_vol_bearish' THEN 1 END) * 100.0 / COUNT(*)
            WHEN current_regime = 'high_vol_bullish' THEN COUNT(CASE WHEN current_regime = 'high_vol_bullish' THEN 1 END) * 100.0 / COUNT(*)
            WHEN current_regime = 'low_vol_bearish' THEN COUNT(CASE WHEN current_regime = 'low_vol_bearish' THEN 1 END) * 100.0 / COUNT(*)
            WHEN current_regime = 'low_vol_bullish' THEN COUNT(CASE WHEN current_regime = 'low_vol_bullish' THEN 1 END) * 100.0 / COUNT(*)
            WHEN current_regime = 'neutral' THEN COUNT(CASE WHEN current_regime = 'neutral' THEN 1 END) * 100.0 / COUNT(*)
        END) as state_balance_std,
        
        -- Regime persistence from durations
        (SELECT AVG(regime_duration_minutes) FROM regime_durations rd WHERE rd.classifier_id = cfs.classifier_id) as avg_regime_duration_minutes,
        (SELECT STDDEV(regime_duration_minutes) FROM regime_durations rd WHERE rd.classifier_id = cfs.classifier_id) as regime_duration_std,
        (SELECT COUNT(*) FROM regime_durations rd WHERE rd.classifier_id = cfs.classifier_id) as total_regime_changes,
        
        -- Stability score: fewer changes = more stable
        (SELECT COUNT(*) FROM regime_durations rd WHERE rd.classifier_id = cfs.classifier_id) * 1.0 / COUNT(*) as regime_change_frequency
        
    FROM classifier_full_series cfs
    WHERE current_regime IS NOT NULL
    GROUP BY classifier_id
),

-- Extract parameter values for additional scoring
parameter_analysis AS (
    SELECT 
        classifier_id,
        -- Extract the three parameters from the ID
        CAST(REGEXP_EXTRACT(classifier_id, '_(\\d+)_(\\d+)_(\\d+)$', 1) AS INTEGER) as vol_lookback,
        CAST(REGEXP_EXTRACT(classifier_id, '_(\\d+)_(\\d+)_(\\d+)$', 2) AS INTEGER) as momentum_fast,
        CAST(REGEXP_EXTRACT(classifier_id, '_(\\d+)_(\\d+)_(\\d+)$', 3) AS INTEGER) as momentum_slow
    FROM excellent_classifiers
),

-- Calculate comprehensive quality score
final_ranking AS (
    SELECT 
        cqm.*,
        pa.vol_lookback,
        pa.momentum_fast,
        pa.momentum_slow,
        
        -- Quality scoring components
        CASE 
            WHEN state_balance_std IS NULL THEN 0
            WHEN state_balance_std < 3 THEN 100
            WHEN state_balance_std < 5 THEN 90
            WHEN state_balance_std < 8 THEN 70
            ELSE 50
        END as balance_score,
        
        CASE 
            WHEN avg_regime_duration_minutes IS NULL THEN 0
            WHEN avg_regime_duration_minutes >= 20 THEN 100
            WHEN avg_regime_duration_minutes >= 15 THEN 90
            WHEN avg_regime_duration_minutes >= 10 THEN 80
            WHEN avg_regime_duration_minutes >= 8 THEN 70
            ELSE 50
        END as persistence_score,
        
        CASE 
            WHEN regime_change_frequency IS NULL THEN 0
            WHEN regime_change_frequency < 0.01 THEN 100  -- Very stable
            WHEN regime_change_frequency < 0.02 THEN 90
            WHEN regime_change_frequency < 0.03 THEN 80
            WHEN regime_change_frequency < 0.05 THEN 70
            ELSE 50
        END as stability_score,
        
        -- Parameter preference (prefer middle values)
        CASE 
            WHEN vol_lookback IN (8, 5) THEN 90  -- Good lookback periods
            ELSE 80
        END as param_score
        
    FROM classifier_quality_metrics cqm
    JOIN parameter_analysis pa ON cqm.classifier_id = pa.classifier_id
)

SELECT 
    '=== TOP 18 VOLATILITY MOMENTUM CLASSIFIERS RANKED ===' as header;

SELECT 
    ROW_NUMBER() OVER (ORDER BY (balance_score + persistence_score + stability_score + param_score) DESC) as rank,
    classifier_id,
    
    -- Key parameters
    vol_lookback,
    momentum_fast, 
    momentum_slow,
    
    -- Quality metrics
    ROUND(state_balance_std, 2) as state_balance_std,
    ROUND(avg_regime_duration_minutes, 1) as avg_duration_min,
    ROUND(regime_change_frequency * 100, 3) as change_freq_pct,
    
    -- Individual scores
    balance_score,
    persistence_score,
    stability_score,
    param_score,
    
    -- Overall quality score
    (balance_score + persistence_score + stability_score + param_score) as total_score,
    
    -- State distribution
    ROUND(high_vol_bearish_pct, 1) as hv_bear_pct,
    ROUND(high_vol_bullish_pct, 1) as hv_bull_pct,
    ROUND(low_vol_bearish_pct, 1) as lv_bear_pct,
    ROUND(low_vol_bullish_pct, 1) as lv_bull_pct,
    ROUND(neutral_pct, 1) as neutral_pct,
    
    total_regime_changes,
    total_observations

FROM final_ranking
ORDER BY (balance_score + persistence_score + stability_score + param_score) DESC
LIMIT 15;

-- Show parameter distribution of top performers
SELECT 
    '=== PARAMETER PATTERNS IN TOP PERFORMERS ===' as header;

WITH top_10 AS (
    SELECT *,
        ROW_NUMBER() OVER (ORDER BY (balance_score + persistence_score + stability_score + param_score) DESC) as rank
    FROM final_ranking
)
SELECT 
    vol_lookback,
    momentum_fast,
    momentum_slow,
    COUNT(*) as classifier_count,
    AVG(total_score) as avg_score,
    STRING_AGG(classifier_id, ', ') as classifier_ids
FROM top_10 
WHERE rank <= 10
GROUP BY vol_lookback, momentum_fast, momentum_slow
ORDER BY avg_score DESC;