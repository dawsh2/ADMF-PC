
WITH regime_times AS (
    SELECT 
        val as regime,
        COUNT(*) / (252.0 * 390.0) as regime_years
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    GROUP BY val
)

SELECT 
    regime,
    ROUND(regime_years, 3) as years,
    ROUND(regime_years * 252, 0) as trading_days
FROM regime_times
ORDER BY regime;
