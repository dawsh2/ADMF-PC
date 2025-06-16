
-- Final calculation with actual returns and proper Sharpe ratios
WITH source_prices AS (
    SELECT ts, close as price
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE ts >= '2024-03-26' AND ts <= '2025-01-17'
),

-- Get signals with actual prices and regimes
signals_with_data AS (
    SELECT 
        s.ts,
        s.val as signal,
        p.price,
        c.val as regime,
        ROW_NUMBER() OVER (ORDER BY s.ts) as seq
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet') s
    LEFT JOIN source_prices p ON s.ts = p.ts
    ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
    ON s.ts >= c.ts
    WHERE p.price IS NOT NULL
),

-- Calculate returns between trades
trade_returns AS (
    SELECT 
        seq,
        signal,
        price,
        regime,
        LAG(price) OVER (ORDER BY seq) as prev_price,
        LAG(signal) OVER (ORDER BY seq) as prev_signal,
        CASE 
            WHEN LAG(signal) OVER (ORDER BY seq) = 1 THEN 
                (price - LAG(price) OVER (ORDER BY seq)) / LAG(price) OVER (ORDER BY seq) * 100
            WHEN LAG(signal) OVER (ORDER BY seq) = -1 THEN 
                (LAG(price) OVER (ORDER BY seq) - price) / LAG(price) OVER (ORDER BY seq) * 100
            ELSE NULL
        END as return_pct
    FROM signals_with_data
),

-- Calculate statistics by regime
regime_stats AS (
    SELECT 
        regime,
        COUNT(*) as num_trades,
        AVG(return_pct) as avg_return,
        STDDEV(return_pct) as std_return
    FROM trade_returns
    WHERE return_pct IS NOT NULL AND regime IS NOT NULL
    GROUP BY regime
),

-- Get regime time periods
regime_times AS (
    SELECT 
        val as regime,
        (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () / 100.0) * (297.0/365.25) as regime_years
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    GROUP BY val
),

-- Final Sharpe calculation
sharpe_calculation AS (
    SELECT 
        s.regime,
        s.num_trades,
        t.regime_years,
        ROUND(s.num_trades / (t.regime_years * 252), 1) as trades_per_day,
        ROUND(s.avg_return, 4) as avg_return_pct,
        ROUND(s.std_return, 4) as std_return_pct,
        CASE 
            WHEN s.std_return > 0 THEN s.avg_return / s.std_return
            ELSE 0
        END as sharpe_ratio,
        -- Annualized Sharpe: multiply by sqrt of trades per year
        CASE 
            WHEN s.std_return > 0 THEN 
                (s.avg_return / s.std_return) * SQRT(s.num_trades / t.regime_years)
            ELSE 0
        END as annualized_sharpe
    FROM regime_stats s
    JOIN regime_times t ON s.regime = t.regime
)

SELECT 
    regime,
    num_trades,
    trades_per_day,
    avg_return_pct,
    std_return_pct,
    ROUND(annualized_sharpe, 3) as annualized_sharpe
FROM sharpe_calculation
ORDER BY annualized_sharpe DESC;
