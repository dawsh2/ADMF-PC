-- Simple strategy analysis with proper signal interpretation
-- Based on signal-storage-replay.md sparse signal format

SET memory_limit='2GB';
SET threads=1;

-- Analyze one strategy with simplified approach
WITH base_data AS (
    SELECT ts, close as price
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE ts >= '2024-03-26' AND ts <= '2025-01-17'
    ORDER BY ts
),

-- Get signals and add sequence numbers
signals AS (
    SELECT 
        ts,
        CAST(val AS INTEGER) as signal_value,
        ROW_NUMBER() OVER (ORDER BY ts) as seq
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet')
    ORDER BY ts
),

-- Join signals with prices and regimes
signals_with_context AS (
    SELECT 
        s.ts,
        s.signal_value,
        s.seq,
        p.price,
        r.val as regime
    FROM signals s
    INNER JOIN base_data p ON s.ts = p.ts
    ASOF LEFT JOIN read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet') r
    ON s.ts >= r.ts
),

-- Calculate trade boundaries and returns
-- Each signal change is a trade boundary
-- Trade return = return from previous signal to current signal
trade_calculations AS (
    SELECT 
        regime,
        signal_value,
        LAG(signal_value) OVER (ORDER BY seq) as prev_signal,
        price,
        LAG(price) OVER (ORDER BY seq) as prev_price,
        -- Calculate return based on previous position
        CASE 
            WHEN LAG(signal_value) OVER (ORDER BY seq) = 1 THEN 
                -- Long position: gain from price increase
                (price - LAG(price) OVER (ORDER BY seq)) / LAG(price) OVER (ORDER BY seq)
            WHEN LAG(signal_value) OVER (ORDER BY seq) = -1 THEN 
                -- Short position: gain from price decrease  
                (LAG(price) OVER (ORDER BY seq) - price) / LAG(price) OVER (ORDER BY seq)
            ELSE NULL
        END as trade_return
    FROM signals_with_context
),

-- Filter for valid trades
valid_trades AS (
    SELECT *
    FROM trade_calculations
    WHERE prev_signal IS NOT NULL 
      AND trade_return IS NOT NULL
),

-- Filter for valid trades and group by regime
regime_performance AS (
    SELECT 
        regime,
        COUNT(*) as num_trades,
        AVG(trade_return) as mean_return,
        STDDEV(trade_return) as std_return
    FROM valid_trades
    WHERE regime IS NOT NULL
    GROUP BY regime
    HAVING COUNT(*) >= 5  -- Minimum trades for stats
),

-- Calculate regime time periods
regime_periods AS (
    SELECT 
        val as regime,
        COUNT(*) as regime_minutes,
        -- Convert to years: total dataset is ~297 trading days
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
    ROUND(rp.mean_return / rp.std_return, 6) as raw_sharpe,
    ROUND((rp.mean_return / rp.std_return) * SQRT(rp.num_trades / rt.regime_years), 4) as annualized_sharpe
FROM regime_performance rp
JOIN regime_periods rt ON rp.regime = rt.regime
ORDER BY annualized_sharpe DESC;