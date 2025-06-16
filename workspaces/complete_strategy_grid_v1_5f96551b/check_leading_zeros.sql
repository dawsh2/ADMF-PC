-- Check if parquet files include leading zeros in parameter names
-- This could affect classifier ID matching

PRAGMA memory_limit='3GB';
SET threads=4;

-- Check actual classifier IDs in the volatility momentum files
SELECT 
    '=== ACTUAL CLASSIFIER IDs IN VOLATILITY MOMENTUM FILES ===' as header;

SELECT DISTINCT 
    strat as actual_classifier_id
FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet')
ORDER BY strat;

-- Check if our assumed IDs match what's actually in the files
WITH assumed_ids AS (
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
actual_ids AS (
    SELECT DISTINCT strat as actual_classifier_id
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet')
)
SELECT 
    '=== ID MATCHING CHECK ===' as header;

-- Show which IDs we assumed vs what exists
SELECT 
    a.classifier_id as assumed_id,
    CASE WHEN act.actual_classifier_id IS NOT NULL THEN 'EXISTS' ELSE 'MISSING' END as status,
    act.actual_classifier_id as actual_match
FROM assumed_ids a
LEFT JOIN actual_ids act ON a.classifier_id = act.actual_classifier_id
ORDER BY a.classifier_id;

-- Check for pattern differences (leading zeros, etc.)
SELECT 
    '=== SAMPLE OF ACTUAL IDs (to see format pattern) ===' as header;

SELECT 
    strat as actual_classifier_id,
    -- Extract numeric parameters to see zero-padding pattern
    REGEXP_EXTRACT(strat, '_(\d+)_(\d+)_(\d+)$', 1) as param1,
    REGEXP_EXTRACT(strat, '_(\d+)_(\d+)_(\d+)$', 2) as param2,
    REGEXP_EXTRACT(strat, '_(\d+)_(\d+)_(\d+)$', 3) as param3
FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet')
WHERE strat LIKE '%05%' OR strat LIKE '%08%'  -- Focus on our target parameters
GROUP BY strat
ORDER BY strat
LIMIT 20;