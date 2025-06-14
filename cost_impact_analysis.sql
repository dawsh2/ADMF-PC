-- Analyze cost impact on strategy profitability
WITH performance AS (
    SELECT 
        s.strat,
        COUNT(*) as trades,
        AVG(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END) as gross_return,
        STDDEV(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END) as volatility
    FROM read_parquet('workspaces/expansive_grid_search_8d35f12c/traces/*/signals/*/*.parquet') s
    JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index
    JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index
    WHERE s.val != 0
    GROUP BY s.strat
    HAVING COUNT(*) >= 250
)
SELECT 
    strat,
    trades,
    ROUND(gross_return, 4) as gross_return_pct,
    ROUND(gross_return - 0.005, 4) as net_0_5bp_pct,  -- 0.5bp round trip
    ROUND(gross_return - 0.01, 4) as net_1bp_pct,     -- 1bp round trip
    ROUND(gross_return - 0.02, 4) as net_2bp_pct,     -- 2bp round trip
    ROUND(volatility, 4) as volatility_pct,
    ROUND(gross_return / NULLIF(volatility, 0), 3) as gross_sharpe
FROM performance
ORDER BY gross_return DESC
LIMIT 20;