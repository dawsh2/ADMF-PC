
-- Calculate corrected trade frequencies
WITH regime_stats AS (
    SELECT 
        val as regime,
        COUNT(*) as regime_minutes,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as regime_pct
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    GROUP BY val
),

regime_times AS (
    SELECT 
        regime,
        regime_pct / 100.0 * (297.0/365.25) as regime_years
    FROM regime_stats
),

-- CORRECTED: Each signal = 1 trade (not pairs)
macd_trades AS (
    SELECT 
        c.val as regime,
        COUNT(*) as total_trades
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet') s
    ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
    ON s.ts >= c.ts
    GROUP BY c.val
),

williams_trades AS (
    SELECT 
        regime,
        non_zero_count as total_trades  -- Each non-zero signal = 1 trade
    FROM (
        SELECT 
            c.val as regime,
            COUNT(*) as non_zero_count
        FROM read_parquet('traces/SPY_1m/signals/williams_r_grid/SPY_williams_r_grid_7_-80_-20.parquet') s
        ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
        ON s.ts >= c.ts
        WHERE CAST(s.val AS INTEGER) \!= 0
        GROUP BY c.val
    )
    GROUP BY regime, non_zero_count
)

-- Show corrected frequencies
SELECT 'MACD_CORRECTED' as strategy, 
       m.regime,
       r.regime_years,
       m.total_trades,
       ROUND(m.total_trades / (r.regime_years * 252), 1) as trades_per_day
FROM macd_trades m
JOIN regime_times r ON m.regime = r.regime

UNION ALL

SELECT 'WILLIAMS_CORRECTED' as strategy,
       w.regime, 
       r.regime_years,
       w.total_trades,
       ROUND(w.total_trades / (r.regime_years * 252), 1) as trades_per_day
FROM williams_trades w
JOIN regime_times r ON w.regime = r.regime

ORDER BY strategy, regime;
