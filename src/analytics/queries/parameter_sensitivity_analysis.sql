-- Analyze parameter sensitivity for stable strategy selection
-- Find parameter combinations in "neighborhoods" of positive returns

WITH rsi_parameters AS (
    -- Extract RSI parameters from strategy names
    SELECT 
        strat,
        CAST(REGEXP_EXTRACT(strat, 'rsi_grid_([0-9]+)_', 1) AS INTEGER) as period,
        CAST(REGEXP_EXTRACT(strat, 'rsi_grid_[0-9]+_([0-9]+)_', 1) AS INTEGER) as oversold,
        CAST(REGEXP_EXTRACT(strat, 'rsi_grid_[0-9]+_[0-9]+_([0-9]+)', 1) AS INTEGER) as overbought
    FROM read_parquet('workspaces/expansive_grid_search_db1cfd51/traces/SPY_1m/signals/rsi_grid/*.parquet')
    GROUP BY strat
),
performance_grid AS (
    SELECT 
        p.period,
        p.oversold,
        p.overbought,
        COUNT(*) as trades,
        AVG(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END) as avg_return
    FROM read_parquet('workspaces/expansive_grid_search_db1cfd51/traces/SPY_1m/signals/rsi_grid/*.parquet') s
    JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index
    JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index
    JOIN rsi_parameters p ON s.strat = p.strat
    WHERE s.val != 0
    GROUP BY p.period, p.oversold, p.overbought
),
neighborhood_analysis AS (
    SELECT 
        pg1.period,
        pg1.oversold,
        pg1.overbought,
        pg1.avg_return,
        pg1.trades,
        -- Count profitable neighbors
        COUNT(CASE WHEN pg2.avg_return > 0 THEN 1 END) as profitable_neighbors,
        -- Average return of neighborhood
        AVG(pg2.avg_return) as neighborhood_avg_return
    FROM performance_grid pg1
    JOIN performance_grid pg2 ON 
        ABS(pg1.period - pg2.period) <= 2 AND
        ABS(pg1.oversold - pg2.oversold) <= 5 AND
        ABS(pg1.overbought - pg2.overbought) <= 5
    GROUP BY pg1.period, pg1.oversold, pg1.overbought, pg1.avg_return, pg1.trades
)
SELECT 
    period,
    oversold,
    overbought,
    trades,
    ROUND(avg_return, 4) as avg_return,
    profitable_neighbors,
    ROUND(neighborhood_avg_return, 4) as neighborhood_return
FROM neighborhood_analysis
WHERE avg_return > 0 AND profitable_neighbors >= 3
ORDER BY neighborhood_return DESC, avg_return DESC;