-- Compare different exit strategies
WITH exit_comparison AS (
    SELECT 
        s.strat,
        -- 1-bar exit
        CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END as return_1bar,
        -- 5-bar exit
        CASE 
            WHEN s.val = 1 THEN (m5.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m5.close) / m1.close * 100
        END as return_5bar,
        -- 10-bar exit
        CASE 
            WHEN s.val = 1 THEN (m10.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m10.close) / m1.close * 100
        END as return_10bar
    FROM read_parquet('workspaces/expansive_grid_search_db1cfd51/traces/SPY_1m/signals/rsi_grid/*.parquet') s
    JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index
    LEFT JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index
    LEFT JOIN read_parquet('data/SPY_1m.parquet') m5 ON s.idx + 5 = m5.bar_index
    LEFT JOIN read_parquet('data/SPY_1m.parquet') m10 ON s.idx + 10 = m10.bar_index
    WHERE s.val != 0
)
SELECT 
    strat,
    COUNT(*) as trades,
    ROUND(AVG(return_1bar), 4) as avg_return_1bar,
    ROUND(AVG(return_5bar), 4) as avg_return_5bar,
    ROUND(AVG(return_10bar), 4) as avg_return_10bar
FROM exit_comparison
GROUP BY strat
HAVING COUNT(*) >= 5
ORDER BY avg_return_5bar DESC;