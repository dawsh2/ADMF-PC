
-- Get first 1000 actual trade returns to see real bps
WITH source_prices AS (
    SELECT ts, close as price
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE ts >= '2024-03-26' AND ts <= '2024-04-15'  -- Smaller date range
),

signals AS (
    SELECT 
        s.ts,
        CAST(s.val AS INTEGER) as signal,
        p.price,
        ROW_NUMBER() OVER (ORDER BY s.ts) as seq
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet') s
    LEFT JOIN source_prices p ON s.ts = p.ts
    WHERE p.price IS NOT NULL
    ORDER BY s.ts
    LIMIT 1000
),

returns AS (
    SELECT 
        signal,
        price,
        LAG(price) OVER (ORDER BY seq) as prev_price,
        LAG(signal) OVER (ORDER BY seq) as prev_signal,
        CASE 
            WHEN LAG(signal) OVER (ORDER BY seq) = 1 THEN 
                ((price - LAG(price) OVER (ORDER BY seq)) / LAG(price) OVER (ORDER BY seq)) * 10000
            WHEN LAG(signal) OVER (ORDER BY seq) = -1 THEN 
                ((LAG(price) OVER (ORDER BY seq) - price) / LAG(price) OVER (ORDER BY seq)) * 10000
            ELSE NULL
        END as return_bps
    FROM signals
)

SELECT 
    COUNT(*) as num_trades,
    ROUND(AVG(return_bps), 2) as avg_return_bps,
    ROUND(STDDEV(return_bps), 2) as std_return_bps,
    ROUND(MIN(return_bps), 2) as min_return_bps,
    ROUND(MAX(return_bps), 2) as max_return_bps
FROM returns 
WHERE return_bps IS NOT NULL;
