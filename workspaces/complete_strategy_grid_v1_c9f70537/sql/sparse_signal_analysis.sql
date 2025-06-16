-- Sparse signal analysis with correct interpretation
-- Sparse storage only stores signal CHANGES, not every bar

SET memory_limit='2GB';
SET threads=1;

-- First, let's understand the sparse signals
WITH sparse_signals AS (
    SELECT 
        ts,
        CAST(val AS INTEGER) as signal_value,
        ROW_NUMBER() OVER (ORDER BY ts) as change_num
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet')
    ORDER BY ts
),

-- Get price at each signal change point
signal_changes_with_prices AS (
    SELECT 
        s.ts,
        s.signal_value,
        s.change_num,
        p.close as price,
        c.val as regime
    FROM sparse_signals s
    INNER JOIN read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet') p 
        ON s.ts = p.timestamp
    ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') c
        ON s.ts >= c.ts
    WHERE p.timestamp >= '2024-03-26' AND p.timestamp <= '2025-01-17'
),

-- Calculate returns between signal changes
-- Each signal change represents a trade boundary
trade_returns AS (
    SELECT 
        ts,
        signal_value,
        regime,
        price,
        LAG(price) OVER (ORDER BY change_num) as entry_price,
        LAG(signal_value) OVER (ORDER BY change_num) as prev_signal,
        -- Calculate return based on previous position
        CASE 
            WHEN LAG(signal_value) OVER (ORDER BY change_num) = 1 THEN 
                -- Was long, calculate long return
                (price - LAG(price) OVER (ORDER BY change_num)) / LAG(price) OVER (ORDER BY change_num)
            WHEN LAG(signal_value) OVER (ORDER BY change_num) = -1 THEN 
                -- Was short, calculate short return
                (LAG(price) OVER (ORDER BY change_num) - price) / LAG(price) OVER (ORDER BY change_num)
            ELSE NULL
        END as trade_return
    FROM signal_changes_with_prices
),

-- Group by regime
regime_performance AS (
    SELECT 
        regime,
        COUNT(*) as num_trades,
        AVG(trade_return) as mean_return,
        STDDEV(trade_return) as std_return,
        MIN(trade_return) as worst_trade,
        MAX(trade_return) as best_trade
    FROM trade_returns
    WHERE trade_return IS NOT NULL AND regime IS NOT NULL
    GROUP BY regime
),

-- Calculate regime durations
regime_periods AS (
    SELECT 
        val as regime,
        COUNT(*) as regime_minutes,
        -- Total dataset spans ~297 trading days
        ROUND((COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () / 100.0) * (297.0/365.25), 3) as regime_years
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    GROUP BY val
)

-- Final results
SELECT 
    'macd_5_20_11' as strategy,
    rp.regime,
    rp.num_trades,
    rt.regime_years,
    ROUND(rp.num_trades / (rt.regime_years * 252), 1) as trades_per_day,
    ROUND(rp.mean_return * 10000, 2) as mean_return_bps,
    ROUND(rp.std_return * 10000, 2) as std_return_bps,
    ROUND(rp.mean_return / NULLIF(rp.std_return, 0), 4) as raw_sharpe,
    ROUND((rp.mean_return / NULLIF(rp.std_return, 0)) * SQRT(rp.num_trades / rt.regime_years), 4) as annualized_sharpe,
    ROUND(rp.worst_trade * 10000, 2) as worst_trade_bps,
    ROUND(rp.best_trade * 10000, 2) as best_trade_bps
FROM regime_performance rp
JOIN regime_periods rt ON rp.regime = rt.regime
ORDER BY annualized_sharpe DESC;