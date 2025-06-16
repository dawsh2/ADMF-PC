
-- COMPREHENSIVE TRADE BOUNDARY AND SHARPE ANALYSIS
WITH source_prices AS (
    SELECT ts, close as price
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE ts >= '2024-03-26' AND ts <= '2025-01-17'
),

-- MACD signals with prices and regimes (each signal = 1 boundary)
macd_data AS (
    SELECT 
        s.ts,
        CAST(s.val AS INTEGER) as signal,
        p.price,
        c.val as regime,
        ROW_NUMBER() OVER (ORDER BY s.ts) as seq
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet') s
    LEFT JOIN source_prices p ON s.ts = p.ts
    ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
    ON s.ts >= c.ts
    WHERE p.price IS NOT NULL
),

-- Calculate position changes and returns
macd_returns AS (
    SELECT 
        seq,
        signal,
        price,
        regime,
        LAG(price) OVER (ORDER BY seq) as prev_price,
        LAG(signal) OVER (ORDER BY seq) as prev_signal,
        -- Return calculation based on previous position
        CASE 
            WHEN LAG(signal) OVER (ORDER BY seq) = 1 THEN 
                (price - LAG(price) OVER (ORDER BY seq)) / LAG(price) OVER (ORDER BY seq) * 100
            WHEN LAG(signal) OVER (ORDER BY seq) = -1 THEN 
                (LAG(price) OVER (ORDER BY seq) - price) / LAG(price) OVER (ORDER BY seq) * 100
            ELSE NULL
        END as return_pct
    FROM macd_data
),

-- Regime statistics
regime_stats AS (
    SELECT 
        regime,
        COUNT(*) as num_boundaries,
        AVG(return_pct) as avg_return,
        STDDEV(return_pct) as std_return
    FROM macd_returns 
    WHERE return_pct IS NOT NULL AND regime IS NOT NULL
    GROUP BY regime
),

-- Regime time periods  
regime_times AS (
    SELECT 
        val as regime,
        (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () / 100.0) * (297.0/365.25) as regime_years
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    GROUP BY val
)

-- Final results
SELECT 
    'macd_crossover_5_20_11' as strategy,
    s.regime,
    s.num_boundaries,
    ROUND(t.regime_years, 3) as regime_years,
    ROUND(s.num_boundaries / (t.regime_years * 252), 1) as boundaries_per_day,
    ROUND(s.avg_return, 4) as avg_return_pct,
    ROUND(s.std_return, 4) as std_return_pct,
    ROUND(
        CASE 
            WHEN s.std_return > 0 THEN 
                (s.avg_return / s.std_return) * SQRT(s.num_boundaries / t.regime_years)
            ELSE 0
        END, 3
    ) as annualized_sharpe
FROM regime_stats s
JOIN regime_times t ON s.regime = t.regime
ORDER BY annualized_sharpe DESC;
