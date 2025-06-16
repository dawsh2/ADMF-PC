
-- Calculate actual trade returns for MACD strategy
WITH source_prices AS (
    SELECT 
        ts,
        close as price
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE ts >= '2024-03-26' AND ts <= '2025-01-17'
),

-- Get MACD signals with actual prices
signals_with_prices AS (
    SELECT 
        s.ts,
        s.val as signal,
        p.price,
        c.val as regime,
        ROW_NUMBER() OVER (ORDER BY s.ts) as trade_num
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet') s
    LEFT JOIN source_prices p ON s.ts = p.ts
    ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
    ON s.ts >= c.ts
    ORDER BY s.ts
),

-- Calculate returns between consecutive trades
trade_returns AS (
    SELECT 
        trade_num,
        ts,
        signal,
        price,
        regime,
        LAG(price) OVER (ORDER BY trade_num) as prev_price,
        LAG(signal) OVER (ORDER BY trade_num) as prev_signal,
        -- Calculate return based on position direction
        CASE 
            WHEN LAG(signal) OVER (ORDER BY trade_num) = 1 THEN  -- Was long
                (price - LAG(price) OVER (ORDER BY trade_num)) / LAG(price) OVER (ORDER BY trade_num) * 100
            WHEN LAG(signal) OVER (ORDER BY trade_num) = -1 THEN -- Was short
                (LAG(price) OVER (ORDER BY trade_num) - price) / LAG(price) OVER (ORDER BY trade_num) * 100
            ELSE NULL
        END as return_pct
    FROM signals_with_prices
    WHERE price IS NOT NULL
)

-- Show sample returns and basic statistics
SELECT 'SAMPLE_RETURNS' as analysis,
       CAST(trade_num AS VARCHAR) as detail,
       CAST(ROUND(return_pct, 4) AS VARCHAR) as value,
       regime as extra
FROM trade_returns
WHERE return_pct IS NOT NULL
ORDER BY trade_num
LIMIT 10

UNION ALL

-- Basic statistics by regime
SELECT 'STATS' as analysis,
       regime as detail,
       CAST(ROUND(AVG(return_pct), 4) AS VARCHAR) as value,
       CAST(COUNT(*) AS VARCHAR) as extra
FROM trade_returns
WHERE return_pct IS NOT NULL
GROUP BY regime

ORDER BY analysis, detail;
