-- Show details of strategies at each filtering stage
-- First, let's see what strategies made it through each filter

-- 1. Strategies with enough trades (from analysis_results)
WITH filtered_strategies AS (
    SELECT DISTINCT strat
    FROM read_parquet('workspaces/expansive_grid_search_8d35f12c/traces/*/signals/*/*.parquet')
    WHERE val != 0
    GROUP BY strat
    HAVING COUNT(*) >= 250
),
-- 2. Calculate performance metrics for these strategies
performance AS (
    SELECT 
        s.strat,
        COUNT(*) as trades,
        AVG(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END) as gross_return,
        -- With 2bp round trip costs
        AVG(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100 - 0.02
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100 - 0.02
        END) as net_return,
        STDDEV(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END) as volatility
    FROM read_parquet('workspaces/expansive_grid_search_8d35f12c/traces/*/signals/*/*.parquet') s
    JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index
    JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index
    WHERE s.val != 0 AND s.strat IN (SELECT strat FROM filtered_strategies)
    GROUP BY s.strat
)
SELECT 
    strat,
    trades,
    ROUND(gross_return, 4) as gross_return_pct,
    ROUND(net_return, 4) as net_return_pct,
    ROUND(volatility, 4) as volatility_pct,
    ROUND(net_return / NULLIF(volatility, 0), 3) as sharpe_ratio,
    CASE WHEN net_return > 0 THEN 'Profitable' ELSE 'Unprofitable' END as status
FROM performance
WHERE net_return > 0  -- Only show profitable ones
ORDER BY sharpe_ratio DESC
LIMIT 10;