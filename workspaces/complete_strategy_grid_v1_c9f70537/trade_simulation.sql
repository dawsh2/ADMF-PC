
-- Merge signals with regimes and simulate trading
WITH signals_with_regimes AS (
    SELECT 
        s.ts,
        s.val as signal_val,
        s.px as price,
        c.val as regime,
        ROW_NUMBER() OVER (ORDER BY s.ts) as seq
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet') s
    ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
    ON s.ts >= c.ts
    ORDER BY s.ts
),

-- Simulate trades: track position changes
trade_simulation AS (
    SELECT 
        *,
        LAG(signal_val, 1, 0) OVER (ORDER BY seq) as prev_signal,
        -- Determine trade type based on signal transitions
        CASE 
            WHEN signal_val \!= 0 AND LAG(signal_val, 1, 0) OVER (ORDER BY seq) = 0 THEN 'ENTRY'
            WHEN signal_val = 0 AND LAG(signal_val, 1, 0) OVER (ORDER BY seq) \!= 0 THEN 'EXIT'
            WHEN signal_val \!= 0 AND LAG(signal_val, 1, 0) OVER (ORDER BY seq) \!= 0 
                 AND signal_val \!= LAG(signal_val, 1, 0) OVER (ORDER BY seq) THEN 'FLIP'
            ELSE 'HOLD'
        END as trade_type
    FROM signals_with_regimes
),

-- Count trade types by regime
trade_summary AS (
    SELECT 
        regime,
        trade_type,
        COUNT(*) as count
    FROM trade_simulation
    WHERE trade_type IN ('ENTRY', 'EXIT', 'FLIP')
    GROUP BY regime, trade_type
)

-- Show first 20 trades for understanding
SELECT 'SAMPLE_TRADES' as type, ts, signal_val, prev_signal, trade_type, regime, price
FROM trade_simulation
WHERE seq <= 20

UNION ALL

SELECT 'TRADE_SUMMARY' as type, NULL as ts, NULL as signal_val, NULL as prev_signal, 
       trade_type, regime, count as price
FROM trade_summary

ORDER BY type, ts;
