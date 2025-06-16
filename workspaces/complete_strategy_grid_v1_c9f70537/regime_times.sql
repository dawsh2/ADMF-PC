
WITH regime_times AS (
    SELECT 
        val as regime,
        (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () / 100.0) * 0.8055 as regime_years
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    GROUP BY val
)

SELECT 
    regime,
    ROUND(regime_years, 3) as years,
    ROUND((regime_years / 0.8055) * 100, 1) as pct_time
FROM regime_times
ORDER BY regime;
