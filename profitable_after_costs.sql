-- Find strategies profitable after realistic costs (10k bars)
WITH performance AS (
    SELECT 
        s.strat,
        COUNT(*) as trades,
        AVG(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END) as gross_return,
        -- Apply 2bp round trip (1bp each way for commission + slippage)
        AVG(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100 - 0.02
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100 - 0.02
        END) as net_return_conservative,
        -- Apply 1bp round trip (0.5bp each way - more optimistic)
        AVG(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100 - 0.01
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100 - 0.01
        END) as net_return_optimistic
    FROM read_parquet('workspaces/expansive_grid_search_8d35f12c/traces/*/signals/*/*.parquet') s
    JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index
    JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index
    WHERE s.val != 0
    GROUP BY s.strat
    HAVING COUNT(*) >= 50  -- Meaningful sample size
)
SELECT 
    strat,
    trades,
    ROUND(gross_return, 4) as gross_return_pct,
    ROUND(net_return_conservative, 4) as net_conservative_pct,
    ROUND(net_return_optimistic, 4) as net_optimistic_pct,
    ROUND(trades * net_return_conservative, 2) as total_return_conservative
FROM performance
WHERE net_return_conservative > 0  -- Profitable even with conservative costs
ORDER BY net_return_conservative DESC
LIMIT 20;