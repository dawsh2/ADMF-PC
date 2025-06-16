
-- Calculate actual returns and Sharpe ratios
WITH source_prices AS (
    SELECT ts, close as price
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE ts >= '2024-03-26' AND ts <= '2025-01-17'
),

-- MACD signals with prices and regimes
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

-- Calculate returns between position changes
trade_returns AS (
    SELECT 
        seq,
        signal,
        price,
        regime,
        LAG(price) OVER (ORDER BY seq) as prev_price,
        LAG(signal) OVER (ORDER BY seq) as prev_signal,
        -- Return based on previous position direction
        CASE 
            WHEN LAG(signal) OVER (ORDER BY seq) = 1 THEN 
                (price - LAG(price) OVER (ORDER BY seq)) / LAG(price) OVER (ORDER BY seq) * 100
            WHEN LAG(signal) OVER (ORDER BY seq) = -1 THEN 
                (LAG(price) OVER (ORDER BY seq) - price) / LAG(price) OVER (ORDER BY seq) * 100
            ELSE NULL
        END as return_pct
    FROM macd_data
),

-- Statistics by regime  
regime_performance AS (
    SELECT 
        regime,
        COUNT(*) as num_boundaries,
        COUNT(*) - 1 as num_trades,
        AVG(return_pct) as avg_return_pct,
        STDDEV(return_pct) as std_return_pct
    FROM trade_returns 
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

-- Final calculation
SELECT 
    p.regime,
    p.num_trades,
    ROUND(t.regime_years, 3) as regime_years,
    ROUND(p.num_trades / (t.regime_years * 252), 1) as trades_per_day,
    ROUND(p.avg_return_pct, 6) as avg_return_pct,
    ROUND(p.std_return_pct, 4) as std_return_pct,
    ROUND(p.avg_return_pct / p.std_return_pct, 6) as raw_sharpe,
    ROUND(p.num_trades / t.regime_years, 0) as trades_per_year,
    ROUND(
        (p.avg_return_pct / p.std_return_pct) * SQRT(p.num_trades / t.regime_years), 4
    ) as annualized_sharpe
FROM regime_performance p
JOIN regime_times t ON p.regime = t.regime
ORDER BY annualized_sharpe DESC;
