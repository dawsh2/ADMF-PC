-- Find top 10 strategies by annualized Sharpe ratio per regime
-- Optimized for large datasets

SET memory_limit='4GB';
SET threads=2;
SET preserve_insertion_order=false;

-- Create strategy list to analyze
CREATE TEMP TABLE strategy_list AS
SELECT * FROM (VALUES
    ('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet', 'macd_5_20_11'),
    ('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_8_21_9.parquet', 'macd_8_21_9'),
    ('traces/SPY_1m/signals/williams_r_grid/SPY_williams_r_grid_14_80_20.parquet', 'williams_14_80_20'),
    ('traces/SPY_1m/signals/cci_grid/SPY_cci_grid_14_80_20.parquet', 'cci_14_80_20'),
    ('traces/SPY_1m/signals/rsi_grid/SPY_rsi_grid_14_30_70.parquet', 'rsi_14_30_70'),
    ('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet', 'macd_12_26_9'),
    ('traces/SPY_1m/signals/stoch_grid/SPY_stoch_grid_14_3_3_80_20.parquet', 'stoch_14_3_3_80_20')
) AS t(file_path, strategy_name);

-- Calculate regime periods once
CREATE TEMP TABLE regime_periods AS
SELECT 
    val as regime,
    COUNT(*) as regime_minutes,
    -- Use actual trading days calculation: 297 days over 8350 minutes
    ROUND((COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () / 100.0) * (297.0/365.25), 3) as regime_years
FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
GROUP BY val;

-- Function to analyze one strategy
CREATE OR REPLACE MACRO analyze_strategy(strategy_file, strategy_name) AS TABLE (
    WITH source_prices AS (
        SELECT ts, close as price
        FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
        WHERE ts >= '2024-03-26' AND ts <= '2025-01-17'
    ),
    
    signals_with_prices AS (
        SELECT 
            s.ts,
            CAST(s.val AS INTEGER) as signal,
            p.price,
            ROW_NUMBER() OVER (ORDER BY s.ts) as seq
        FROM read_parquet(strategy_file) s
        INNER JOIN source_prices p ON s.ts = p.ts
        ORDER BY s.ts
    ),
    
    signals_with_regime AS (
        SELECT 
            sp.*,
            c.val as regime
        FROM signals_with_prices sp
        ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
        ON sp.ts >= c.ts
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
        FROM signals_with_regime
    ),
    
    performance_stats AS (
        SELECT 
            regime,
            COUNT(*) as num_trades,
            AVG(return_decimal) as mean_return,
            STDDEV(return_decimal) as std_return
        FROM trade_returns
        WHERE return_decimal IS NOT NULL AND regime IS NOT NULL
        GROUP BY regime
        HAVING COUNT(*) >= 10  -- Minimum trades for reliable stats
    )
    
    SELECT 
        strategy_name as strategy,
        s.regime,
        s.num_trades,
        r.regime_years,
        ROUND(s.num_trades / (r.regime_years * 252), 1) as trades_per_day,
        ROUND(s.mean_return * 10000, 2) as mean_return_bps,
        ROUND(s.std_return * 10000, 2) as std_return_bps,
        ROUND(s.mean_return / s.std_return, 6) as raw_sharpe,
        ROUND((s.mean_return / s.std_return) * SQRT(s.num_trades / r.regime_years), 4) as annualized_sharpe
    FROM performance_stats s
    JOIN regime_periods r ON s.regime = r.regime
);

-- Test with one strategy first
SELECT * FROM analyze_strategy(
    'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet',
    'macd_5_20_11'
)
ORDER BY annualized_sharpe DESC;