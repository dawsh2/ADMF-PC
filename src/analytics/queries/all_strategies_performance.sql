-- Performance metrics for ALL strategy types
WITH trade_returns AS (
    SELECT 
        s.strat,
        -- Extract strategy type from filename
        REGEXP_EXTRACT(s.strat, '^[^_]+', 0) as strategy_type,
        CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END as return_pct
    FROM read_parquet('workspaces/expansive_grid_search_db1cfd51/traces/SPY_1m/signals/*/*.parquet') s
    JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index      
    JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index  
    WHERE s.val != 0
)
SELECT 
    strategy_type,
    COUNT(DISTINCT strat) as strategies_tested,
    COUNT(*) as total_trades,
    ROUND(AVG(return_pct), 4) as avg_return_pct,
    ROUND(MIN(return_pct), 4) as worst_trade,
    ROUND(MAX(return_pct), 4) as best_trade,
    ROUND(STDDEV(return_pct), 4) as volatility
FROM trade_returns
GROUP BY strategy_type
ORDER BY avg_return_pct DESC;