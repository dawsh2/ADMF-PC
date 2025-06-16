
-- Correct trade boundary analysis
WITH source_prices AS (
    SELECT ts, close as price
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE ts >= '2024-03-26' AND ts <= '2025-01-17'
),

-- Get MACD signals with prices and regimes
signals_with_context AS (
    SELECT 
        s.ts,
        CAST(s.val AS INTEGER) as signal,
        p.price,
        c.val as regime,
        LAG(CAST(s.val AS INTEGER), 1, 0) OVER (ORDER BY s.ts) as prev_signal,
        ROW_NUMBER() OVER (ORDER BY s.ts) as seq
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet') s
    LEFT JOIN source_prices p ON s.ts = p.ts
    ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
    ON s.ts >= c.ts
    WHERE p.price IS NOT NULL
),

-- Identify trade boundaries correctly
trade_boundaries AS (
    SELECT 
        *,
        -- Count trade boundaries based on signal transitions
        CASE 
            -- Signal to 0 = 1 boundary (closure)
            WHEN signal = 0 AND prev_signal \!= 0 THEN 1
            -- 0 to signal = 1 boundary (opening)  
            WHEN signal \!= 0 AND prev_signal = 0 THEN 1
            -- Signal flip (+1 to -1 or -1 to +1) = 2 boundaries (close + open)
            WHEN signal \!= 0 AND prev_signal \!= 0 AND signal \!= prev_signal THEN 2
            ELSE 0
        END as boundaries,
        -- Track position state
        CASE 
            WHEN signal = 1 THEN 'LONG'
            WHEN signal = -1 THEN 'SHORT'
            WHEN signal = 0 THEN 'FLAT'
        END as position
    FROM signals_with_context
    WHERE seq > 1  -- Skip first signal (no previous to compare)
)

-- Show sample boundaries and count total
SELECT 'SAMPLE' as analysis, 
       CAST(seq AS VARCHAR) as detail,
       CAST(prev_signal AS VARCHAR) || ' -> ' || CAST(signal AS VARCHAR) as transition,
       CAST(boundaries AS VARCHAR) as boundary_count
FROM trade_boundaries
WHERE boundaries > 0
ORDER BY seq
LIMIT 15

UNION ALL

SELECT 'TOTALS' as analysis,
       regime as detail,
       'total_boundaries' as transition,
       CAST(SUM(boundaries) AS VARCHAR) as boundary_count
FROM trade_boundaries
WHERE regime IS NOT NULL
GROUP BY regime

ORDER BY analysis, detail;
