-- Calculate strategy performance metrics
WITH trade_returns AS (
    SELECT 
        s.strat,
        CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END as return_pct
    FROM read_parquet('workspaces/expansive_grid_search_db1cfd51/traces/SPY_1m/signals/rsi_grid/*.parquet') s
    JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index      
    JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index  
    WHERE s.val != 0
)
SELECT 
    strat,
    COUNT(*) as total_trades,
    ROUND(AVG(return_pct), 4) as avg_return_pct,
    ROUND(SUM(return_pct), 2) as total_return_pct,
    ROUND(STDDEV(return_pct), 4) as volatility,
    -- Simple Sharpe ratio (return/risk ratio) - not annualized
    -- For proper annualization, need to know actual trading frequency
    ROUND(AVG(return_pct) / NULLIF(STDDEV(return_pct), 0), 3) as sharpe_ratio
FROM trade_returns
GROUP BY strat
ORDER BY sharpe_ratio DESC;