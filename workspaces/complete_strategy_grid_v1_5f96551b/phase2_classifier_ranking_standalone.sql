-- Phase 2: Standalone ranking of top 18 Volatility Momentum classifiers
-- Fixed regex and simplified approach

PRAGMA memory_limit='3GB';
SET threads=4;

SELECT 
    '=== PHASE 2: TOP VOLATILITY MOMENTUM CLASSIFIERS ===' as header;

-- Analysis with manual parameter extraction for our 18 excellent classifiers
SELECT 
    c.strat as classifier_id,
    
    -- Basic metrics
    COUNT(*) as total_state_changes,
    COUNT(DISTINCT c.val) as num_unique_states,
    
    -- State distribution percentages
    ROUND(COUNT(CASE WHEN c.val = 'high_vol_bearish' THEN 1 END) * 100.0 / COUNT(*), 1) as hv_bear_pct,
    ROUND(COUNT(CASE WHEN c.val = 'high_vol_bullish' THEN 1 END) * 100.0 / COUNT(*), 1) as hv_bull_pct,
    ROUND(COUNT(CASE WHEN c.val = 'low_vol_bearish' THEN 1 END) * 100.0 / COUNT(*), 1) as lv_bear_pct,
    ROUND(COUNT(CASE WHEN c.val = 'low_vol_bullish' THEN 1 END) * 100.0 / COUNT(*), 1) as lv_bull_pct,
    ROUND(COUNT(CASE WHEN c.val = 'neutral' THEN 1 END) * 100.0 / COUNT(*), 1) as neutral_pct,
    
    -- Balance deviation (perfect balance = 20% each = lower deviation is better)
    ROUND(
        ABS(COUNT(CASE WHEN c.val = 'high_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
        ABS(COUNT(CASE WHEN c.val = 'high_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
        ABS(COUNT(CASE WHEN c.val = 'low_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
        ABS(COUNT(CASE WHEN c.val = 'low_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
        ABS(COUNT(CASE WHEN c.val = 'neutral' THEN 1 END) * 100.0 / COUNT(*) - 20), 
        1
    ) as balance_deviation,
    
    -- Stability: average minutes between state changes (higher is more stable)
    ROUND(
        EXTRACT(EPOCH FROM (MAX(c.ts::timestamp) - MIN(c.ts::timestamp))) / (COUNT(*) - 1) / 60.0,
        1
    ) as avg_minutes_between_changes,
    
    -- Time coverage
    MIN(c.ts::timestamp) as first_change,
    MAX(c.ts::timestamp) as last_change,
    ROUND(EXTRACT(EPOCH FROM (MAX(c.ts::timestamp) - MIN(c.ts::timestamp))) / 3600.0, 1) as total_hours,
    
    -- Quality score (lower balance deviation + higher stability = better)
    ROUND(
        (100 - (ABS(COUNT(CASE WHEN c.val = 'high_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
                ABS(COUNT(CASE WHEN c.val = 'high_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
                ABS(COUNT(CASE WHEN c.val = 'low_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
                ABS(COUNT(CASE WHEN c.val = 'low_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
                ABS(COUNT(CASE WHEN c.val = 'neutral' THEN 1 END) * 100.0 / COUNT(*) - 20))) +
        (EXTRACT(EPOCH FROM (MAX(c.ts::timestamp) - MIN(c.ts::timestamp))) / (COUNT(*) - 1) / 60.0 * 2),
        1
    ) as quality_score

FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet') c
WHERE c.strat IN (
    'SPY_volatility_momentum_grid_05_55_35',
    'SPY_volatility_momentum_grid_05_55_40', 
    'SPY_volatility_momentum_grid_05_55_45',
    'SPY_volatility_momentum_grid_05_60_35',
    'SPY_volatility_momentum_grid_05_60_40',
    'SPY_volatility_momentum_grid_05_60_45',
    'SPY_volatility_momentum_grid_05_65_35',
    'SPY_volatility_momentum_grid_05_65_40',
    'SPY_volatility_momentum_grid_05_65_45',
    'SPY_volatility_momentum_grid_08_55_35',
    'SPY_volatility_momentum_grid_08_55_40',
    'SPY_volatility_momentum_grid_08_55_45',
    'SPY_volatility_momentum_grid_08_60_35',
    'SPY_volatility_momentum_grid_08_60_40',
    'SPY_volatility_momentum_grid_08_60_45',
    'SPY_volatility_momentum_grid_08_65_35',
    'SPY_volatility_momentum_grid_08_65_40',
    'SPY_volatility_momentum_grid_08_65_45'
)
GROUP BY c.strat
ORDER BY 
    -- Rank by quality score (balance + stability)
    (100 - (ABS(COUNT(CASE WHEN c.val = 'high_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
            ABS(COUNT(CASE WHEN c.val = 'high_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
            ABS(COUNT(CASE WHEN c.val = 'low_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
            ABS(COUNT(CASE WHEN c.val = 'low_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
            ABS(COUNT(CASE WHEN c.val = 'neutral' THEN 1 END) * 100.0 / COUNT(*) - 20))) +
    (EXTRACT(EPOCH FROM (MAX(c.ts::timestamp) - MIN(c.ts::timestamp))) / (COUNT(*) - 1) / 60.0 * 2) DESC;