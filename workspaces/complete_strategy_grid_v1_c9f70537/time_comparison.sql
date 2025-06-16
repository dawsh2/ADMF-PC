
WITH regime_distribution AS (
    SELECT 
        val as regime,
        COUNT(*) as regime_minutes,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as regime_percentage
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    GROUP BY val
),

total_time AS (
    SELECT COUNT(*) / (252.0 * 390.0) as total_years
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
),

regime_specific_time AS (
    SELECT 
        r.regime,
        r.regime_minutes,
        r.regime_percentage,
        t.total_years,
        -- CORRECT: Regime-specific time
        (r.regime_percentage / 100.0) * t.total_years as regime_specific_years,
        -- INCORRECT: Total time (what I was doing wrong)
        t.total_years as total_years_incorrect
    FROM regime_distribution r
    CROSS JOIN total_time t
)

SELECT 
    regime,
    ROUND(regime_percentage, 1) as regime_pct,
    ROUND(total_years_incorrect, 4) as total_time_wrong,
    ROUND(regime_specific_years, 4) as regime_time_correct,
    ROUND(regime_specific_years / total_years_incorrect, 2) as correction_factor
FROM regime_specific_time
ORDER BY regime;
