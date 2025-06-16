-- Calculate actual trade returns for any strategy
-- Usage: Replace strategy_file_path with actual strategy parquet file
-- Returns: trade-by-trade returns with regime information

WITH source_prices AS (
    SELECT ts, close as price
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE ts >= '2024-03-26' AND ts <= '2025-01-17'
),

signals_with_context AS (
    SELECT 
        s.ts,
        CAST(s.val AS INTEGER) as signal,
        p.price,
        c.val as regime,
        ROW_NUMBER() OVER (ORDER BY s.ts) as seq
    FROM read_parquet('${strategy_file_path}') s
    LEFT JOIN source_prices p ON s.ts = p.ts
    ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
    ON s.ts >= c.ts
    WHERE p.price IS NOT NULL
),

trade_returns AS (
    SELECT 
        seq,
        ts,
        signal,
        price,
        regime,
        LAG(price) OVER (ORDER BY seq) as prev_price,
        LAG(signal) OVER (ORDER BY seq) as prev_signal,
        -- Calculate return based on previous position
        CASE 
            WHEN LAG(signal) OVER (ORDER BY seq) = 1 THEN 
                (price - LAG(price) OVER (ORDER BY seq)) / LAG(price) OVER (ORDER BY seq)
            WHEN LAG(signal) OVER (ORDER BY seq) = -1 THEN 
                (LAG(price) OVER (ORDER BY seq) - price) / LAG(price) OVER (ORDER BY seq)
            ELSE NULL
        END as return_decimal
    FROM signals_with_context
)

SELECT 
    regime,
    return_decimal,
    return_decimal * 10000 as return_bps,
    ts,
    signal,
    prev_signal,
    price,
    prev_price
FROM trade_returns 
WHERE return_decimal IS NOT NULL 
  AND regime IS NOT NULL
ORDER BY ts;