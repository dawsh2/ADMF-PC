-- Phase 2: Rank top 18 Volatility Momentum classifiers based on comprehensive quality scores
-- Simplified version focusing on core metrics

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

-- Get classifier data and calculate basic metrics
classifier_metrics AS (
    SELECT 
        c.strat as classifier_id,
        COUNT(*) as total_state_changes,
        COUNT(DISTINCT c.val) as num_unique_states,
        
        -- State distribution percentages
        COUNT(CASE WHEN c.val = 'high_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) as high_vol_bearish_pct,
        COUNT(CASE WHEN c.val = 'high_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) as high_vol_bullish_pct,
        COUNT(CASE WHEN c.val = 'low_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) as low_vol_bearish_pct,
        COUNT(CASE WHEN c.val = 'low_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) as low_vol_bullish_pct,
        COUNT(CASE WHEN c.val = 'neutral' THEN 1 END) * 100.0 / COUNT(*) as neutral_pct,
        
        -- Calculate state balance score (closer to 20% each = better)
        ABS(COUNT(CASE WHEN c.val = 'high_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
        ABS(COUNT(CASE WHEN c.val = 'high_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
        ABS(COUNT(CASE WHEN c.val = 'low_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
        ABS(COUNT(CASE WHEN c.val = 'low_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
        ABS(COUNT(CASE WHEN c.val = 'neutral' THEN 1 END) * 100.0 / COUNT(*) - 20) as balance_deviation,
        
        -- Time span analysis
        MIN(c.ts::timestamp) as first_timestamp,
        MAX(c.ts::timestamp) as last_timestamp,
        EXTRACT(EPOCH FROM (MAX(c.ts::timestamp) - MIN(c.ts::timestamp))) / 3600.0 as total_hours,
        
        -- Calculate average time between state changes (stability)
        EXTRACT(EPOCH FROM (MAX(c.ts::timestamp) - MIN(c.ts::timestamp))) / (COUNT(*) - 1) / 60.0 as avg_minutes_between_changes
        
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet') c
    INNER JOIN excellent_classifiers ec ON c.strat = ec.classifier_id
    GROUP BY c.strat
),

-- Extract parameter values for scoring
parameter_analysis AS (
    SELECT 
        classifier_id,
        -- Extract the three parameters from the ID
        CAST(REGEXP_EXTRACT(classifier_id, '_(\\d+)_(\\d+)_(\\d+)$', 1) AS INTEGER) as vol_lookback,
        CAST(REGEXP_EXTRACT(classifier_id, '_(\\d+)_(\\d+)_(\\d+)$', 2) AS INTEGER) as momentum_fast,
        CAST(REGEXP_EXTRACT(classifier_id, '_(\\d+)_(\\d+)_(\\d+)$', 3) AS INTEGER) as momentum_slow
    FROM excellent_classifiers
),

-- Calculate comprehensive quality scores
final_ranking AS (
    SELECT 
        cm.*,
        pa.vol_lookback,
        pa.momentum_fast,
        pa.momentum_slow,
        
        -- Quality scoring components (0-100 scale)
        CASE 
            WHEN cm.balance_deviation <= 10 THEN 100  -- Very balanced
            WHEN cm.balance_deviation <= 20 THEN 90   -- Good balance
            WHEN cm.balance_deviation <= 30 THEN 80   -- Acceptable balance
            WHEN cm.balance_deviation <= 40 THEN 70   -- Poor balance
            ELSE 50                                   -- Very poor balance
        END as balance_score,
        
        CASE 
            WHEN cm.avg_minutes_between_changes >= 15 THEN 100  -- Very stable (15+ min between changes)
            WHEN cm.avg_minutes_between_changes >= 12 THEN 90   -- Stable (12+ min)
            WHEN cm.avg_minutes_between_changes >= 10 THEN 80   -- Moderately stable (10+ min)
            WHEN cm.avg_minutes_between_changes >= 8 THEN 70    -- Less stable (8+ min)
            ELSE 50                                            -- Unstable (< 8 min)
        END as stability_score,
        
        CASE 
            WHEN cm.num_unique_states = 5 THEN 100              -- All 5 states present
            WHEN cm.num_unique_states = 4 THEN 90               -- 4 states
            WHEN cm.num_unique_states = 3 THEN 70               -- 3 states
            ELSE 50                                             -- Fewer states
        END as diversity_score,
        
        -- Parameter preference scores
        CASE 
            WHEN pa.vol_lookback = 8 THEN 95    -- Optimal volatility lookback
            WHEN pa.vol_lookback = 5 THEN 90    -- Good volatility lookback
            WHEN pa.vol_lookback = 12 THEN 85   -- Acceptable volatility lookback
            ELSE 80
        END as vol_param_score,
        
        CASE 
            WHEN pa.momentum_fast = 60 THEN 95   -- Optimal fast momentum
            WHEN pa.momentum_fast IN (55, 65) THEN 90  -- Good fast momentum
            ELSE 85
        END as momentum_param_score
        
    FROM classifier_metrics cm
    JOIN parameter_analysis pa ON cm.classifier_id = pa.classifier_id
)

SELECT 
    '=== TOP 15 VOLATILITY MOMENTUM CLASSIFIERS RANKED ===' as header;

SELECT 
    ROW_NUMBER() OVER (ORDER BY (balance_score + stability_score + diversity_score + vol_param_score + momentum_param_score) DESC) as rank,
    classifier_id,
    
    -- Key parameters
    vol_lookback,
    momentum_fast, 
    momentum_slow,
    
    -- Quality metrics
    ROUND(balance_deviation, 1) as balance_dev,
    ROUND(avg_minutes_between_changes, 1) as avg_min_between,
    num_unique_states,
    total_state_changes,
    
    -- Individual scores
    balance_score,
    stability_score,
    diversity_score,
    vol_param_score,
    momentum_param_score,
    
    -- Overall quality score
    (balance_score + stability_score + diversity_score + vol_param_score + momentum_param_score) as total_score,
    
    -- State distribution
    ROUND(high_vol_bearish_pct, 1) as hv_bear_pct,
    ROUND(high_vol_bullish_pct, 1) as hv_bull_pct,
    ROUND(low_vol_bearish_pct, 1) as lv_bear_pct,
    ROUND(low_vol_bullish_pct, 1) as lv_bull_pct,
    ROUND(neutral_pct, 1) as neutral_pct

FROM final_ranking
ORDER BY (balance_score + stability_score + diversity_score + vol_param_score + momentum_param_score) DESC
LIMIT 15;

-- Show parameter distribution of top performers
SELECT 
    '=== PARAMETER PATTERNS IN TOP 10 PERFORMERS ===' as header;

WITH top_performers AS (
    SELECT *,
        ROW_NUMBER() OVER (ORDER BY (balance_score + stability_score + diversity_score + vol_param_score + momentum_param_score) DESC) as rank
    FROM final_ranking
)
SELECT 
    vol_lookback,
    momentum_fast,
    momentum_slow,
    COUNT(*) as classifier_count,
    ROUND(AVG(balance_score + stability_score + diversity_score + vol_param_score + momentum_param_score), 1) as avg_total_score,
    STRING_AGG(classifier_id, ', ') as classifier_list
FROM top_performers 
WHERE rank <= 10
GROUP BY vol_lookback, momentum_fast, momentum_slow
ORDER BY avg_total_score DESC;