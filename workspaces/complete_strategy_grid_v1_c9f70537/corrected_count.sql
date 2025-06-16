
-- CORRECTED trade counting: each signal = 1 trade
WITH regime_times AS (
    SELECT 
        val as regime,
        COUNT(*) * 100.0 / SUM(COUNT(*) OVER ()) / 100.0 * (297.0/365.25) as regime_years
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    GROUP BY val
),

-- Count MACD signals correctly: each signal = 1 trade
macd_corrected AS (
    SELECT 
        c.val as regime,
        COUNT(*) as total_trades  -- Each signal is a trade\!
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet') s
    ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
    ON s.ts >= c.ts
    GROUP BY c.val
)

SELECT 
    r.regime,
    ROUND(r.regime_years, 3) as regime_years,
    m.total_trades,
    ROUND(m.total_trades / (r.regime_years * 252), 1) as trades_per_day_corrected
FROM regime_times r
JOIN macd_corrected m ON r.regime = m.regime
ORDER BY r.regime;
