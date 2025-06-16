
WITH source_prices AS (
    SELECT ts, close as price
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE ts >= '2024-03-26' AND ts <= '2025-01-17'
),

macd_signals AS (
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

trade_returns AS (
    SELECT 
        signal,
        price,
        regime,
        LAG(price) OVER (ORDER BY seq) as prev_price,
        LAG(signal) OVER (ORDER BY seq) as prev_signal,
        CASE 
            WHEN LAG(signal) OVER (ORDER BY seq) = 1 THEN 
                (price - LAG(price) OVER (ORDER BY seq)) / LAG(price) OVER (ORDER BY seq)
            WHEN LAG(signal) OVER (ORDER BY seq) = -1 THEN 
                (LAG(price) OVER (ORDER BY seq) - price) / LAG(price) OVER (ORDER BY seq)
            ELSE NULL
        END as return_decimal
    FROM macd_signals
)

SELECT 
    regime,
    return_decimal,
    COUNT(*) OVER (PARTITION BY regime) as trade_count
FROM trade_returns 
WHERE return_decimal IS NOT NULL AND regime IS NOT NULL
ORDER BY regime, return_decimal;
