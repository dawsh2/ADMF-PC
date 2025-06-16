
-- Correct trade counting for different signal patterns
WITH macd_trades AS (
    -- MACD: Only +1/-1, perfect alternating
    SELECT 
        'macd_crossover_5_20_11' as strategy,
        c.val as regime,
        COUNT(*) as total_signals,
        COUNT(*) / 2.0 as complete_trades  -- Each pair is one complete trade
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet') s
    ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
    ON s.ts >= c.ts
    GROUP BY c.val
),

williams_trades AS (
    -- Williams %R: -1, 0, +1 pattern
    SELECT 
        'williams_r_7_-80_-20' as strategy,
        regime,
        non_zero_signals,
        non_zero_signals / 2.0 as complete_trades  -- Non-zero signals ÷ 2
    FROM (
        SELECT 
            c.val as regime,
            COUNT(*) as non_zero_signals
        FROM read_parquet('traces/SPY_1m/signals/williams_r_grid/SPY_williams_r_grid_7_-80_-20.parquet') s
        ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
        ON s.ts >= c.ts
        WHERE CAST(s.val AS INTEGER) \!= 0  -- Ignore zeros as you specified
        GROUP BY c.val
    )
    GROUP BY regime, non_zero_signals
),

regime_times AS (
    SELECT 
        val as regime,
        (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () / 100.0) * 
        (SELECT (EXTRACT('days' FROM (MAX(ts) - MIN(ts))) / 365.25) 
         FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')) as regime_years
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    GROUP BY val
)

-- Show trade counting comparison
SELECT 
    'MACD_PATTERN' as analysis,
    m.strategy,
    m.regime,
    m.complete_trades,
    r.regime_years,
    ROUND(m.complete_trades / (r.regime_years * 252), 1) as trades_per_day
FROM macd_trades m
JOIN regime_times r ON m.regime = r.regime

UNION ALL

SELECT 
    'WILLIAMS_PATTERN' as analysis,
    w.strategy,
    w.regime,
    w.complete_trades,
    r.regime_years,
    ROUND(w.complete_trades / (r.regime_years * 252), 1) as trades_per_day
FROM williams_trades w
JOIN regime_times r ON w.regime = r.regime

ORDER BY analysis, regime, trades_per_day DESC;
