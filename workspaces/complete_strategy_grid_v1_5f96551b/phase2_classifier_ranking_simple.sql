-- Phase 2: Simple ranking of top 18 Volatility Momentum classifiers
-- Direct analysis without complex CTEs

PRAGMA memory_limit='3GB';
SET threads=4;

SELECT 
    '=== PHASE 2: TOP VOLATILITY MOMENTUM CLASSIFIERS ===' as header;

-- Direct analysis of our 18 excellent classifiers
SELECT 
    c.strat as classifier_id,
    
    -- Extract parameters
    CAST(REGEXP_EXTRACT(c.strat, '_(\\d+)_(\\d+)_(\\d+)$', 1) AS INTEGER) as vol_lookback,
    CAST(REGEXP_EXTRACT(c.strat, '_(\\d+)_(\\d+)_(\\d+)$', 2) AS INTEGER) as momentum_fast,
    CAST(REGEXP_EXTRACT(c.strat, '_(\\d+)_(\\d+)_(\\d+)$', 3) AS INTEGER) as momentum_slow,
    
    -- Basic metrics
    COUNT(*) as total_state_changes,
    COUNT(DISTINCT c.val) as num_unique_states,
    
    -- State distribution percentages
    ROUND(COUNT(CASE WHEN c.val = 'high_vol_bearish' THEN 1 END) * 100.0 / COUNT(*), 1) as hv_bear_pct,
    ROUND(COUNT(CASE WHEN c.val = 'high_vol_bullish' THEN 1 END) * 100.0 / COUNT(*), 1) as hv_bull_pct,
    ROUND(COUNT(CASE WHEN c.val = 'low_vol_bearish' THEN 1 END) * 100.0 / COUNT(*), 1) as lv_bear_pct,
    ROUND(COUNT(CASE WHEN c.val = 'low_vol_bullish' THEN 1 END) * 100.0 / COUNT(*), 1) as lv_bull_pct,
    ROUND(COUNT(CASE WHEN c.val = 'neutral' THEN 1 END) * 100.0 / COUNT(*), 1) as neutral_pct,
    
    -- Balance deviation (perfect balance = 20% each = 0 deviation)
    ROUND(
        ABS(COUNT(CASE WHEN c.val = 'high_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
        ABS(COUNT(CASE WHEN c.val = 'high_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
        ABS(COUNT(CASE WHEN c.val = 'low_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
        ABS(COUNT(CASE WHEN c.val = 'low_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) - 20) +
        ABS(COUNT(CASE WHEN c.val = 'neutral' THEN 1 END) * 100.0 / COUNT(*) - 20), 
        1
    ) as balance_deviation,
    
    -- Stability: average minutes between state changes
    ROUND(
        EXTRACT(EPOCH FROM (MAX(c.ts::timestamp) - MIN(c.ts::timestamp))) / (COUNT(*) - 1) / 60.0,
        1
    ) as avg_minutes_between_changes

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
    -- Simple ranking: best balance + stability
    (100 - ABS(COUNT(CASE WHEN c.val = 'high_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) - 20) -
           ABS(COUNT(CASE WHEN c.val = 'high_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) - 20) -
           ABS(COUNT(CASE WHEN c.val = 'low_vol_bearish' THEN 1 END) * 100.0 / COUNT(*) - 20) -
           ABS(COUNT(CASE WHEN c.val = 'low_vol_bullish' THEN 1 END) * 100.0 / COUNT(*) - 20) -
           ABS(COUNT(CASE WHEN c.val = 'neutral' THEN 1 END) * 100.0 / COUNT(*) - 20)) +
    EXTRACT(EPOCH FROM (MAX(c.ts::timestamp) - MIN(c.ts::timestamp))) / (COUNT(*) - 1) / 60.0 DESC;