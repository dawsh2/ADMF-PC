
-- Get correct time period calculation
WITH time_calculation AS (
    SELECT 
        MIN(ts) as start_time,
        MAX(ts) as end_time,
        -- Correct time calculation in years
        (EXTRACT('days' FROM (MAX(ts) - MIN(ts))) / 365.25) as total_years
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
),

-- Regime distribution with correct time periods
regime_distribution AS (
    SELECT 
        val as regime,
        COUNT(*) as regime_minutes,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as regime_pct
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    GROUP BY val
),

-- Calculate regime-specific years correctly
regime_times AS (
    SELECT 
        r.regime,
        r.regime_minutes,
        r.regime_pct,
        t.total_years,
        (r.regime_pct / 100.0) * t.total_years as regime_years
    FROM regime_distribution r
    CROSS JOIN time_calculation t
)

-- Show corrected time calculations
SELECT 
    'TOTAL_DATASET' as analysis,
    'all_regimes' as regime,
    NULL as regime_pct,
    total_years as years
FROM time_calculation

UNION ALL

SELECT 
    'REGIME_SPECIFIC' as analysis,
    regime,
    regime_pct,
    regime_years as years
FROM regime_times

ORDER BY analysis, regime;
