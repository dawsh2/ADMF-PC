-- Complete strategy analysis pipeline
-- Usage: Set strategy_file and strategy_name, then run
-- This combines trade returns calculation and performance stats

-- Configuration
SET VARIABLE strategy_file = 'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet';
SET VARIABLE strategy_name = 'macd_crossover_5_20_11';

-- Step 1: Calculate trade returns
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
    FROM read_parquet(getvariable('strategy_file')) s
    LEFT JOIN source_prices p ON s.ts = p.ts
    ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
    ON s.ts >= c.ts
    WHERE p.price IS NOT NULL
),

trade_returns AS (
    SELECT 
        regime,
        CASE 
            WHEN LAG(signal) OVER (ORDER BY seq) = 1 THEN 
                (price - LAG(price) OVER (ORDER BY seq)) / LAG(price) OVER (ORDER BY seq)
            WHEN LAG(signal) OVER (ORDER BY seq) = -1 THEN 
                (LAG(price) OVER (ORDER BY seq) - price) / LAG(price) OVER (ORDER BY seq)
            ELSE NULL
        END as return_decimal
    FROM signals_with_context
),

-- Step 2: Calculate regime statistics
regime_times AS (
    SELECT 
        val as regime,
        (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () / 100.0) * (297.0/365.25) as regime_years
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    GROUP BY val
),

performance_stats AS (
    SELECT 
        regime,
        COUNT(*) as num_trades,
        AVG(return_decimal) as mean_return,
        STDDEV(return_decimal) as std_return,
        AVG(return_decimal * 10000) as mean_return_bps
    FROM trade_returns
    WHERE return_decimal IS NOT NULL AND regime IS NOT NULL
    GROUP BY regime
)

-- Step 3: Final results
SELECT 
    getvariable('strategy_name') as strategy,
    s.regime,
    s.num_trades,
    ROUND(r.regime_years, 3) as regime_years,
    ROUND(s.num_trades / (r.regime_years * 252), 1) as trades_per_day,
    ROUND(s.mean_return_bps, 2) as mean_return_bps,
    ROUND(s.std_return * 100, 4) as std_return_pct,
    ROUND(s.mean_return / s.std_return, 6) as raw_sharpe,
    ROUND(
        (s.mean_return / s.std_return) * SQRT(s.num_trades / r.regime_years), 4
    ) as annualized_sharpe
FROM performance_stats s
JOIN regime_times r ON s.regime = r.regime
ORDER BY annualized_sharpe DESC;