-- Analyze trading frequency and its impact on returns (10k bars)
WITH strategy_activity AS (
    SELECT 
        s.strat,
        COUNT(*) as total_signals,
        MIN(s.idx) as first_signal,
        MAX(s.idx) as last_signal,
        (MAX(s.idx) - MIN(s.idx)) as trading_period,
        COUNT(*) * 1.0 / NULLIF(MAX(s.idx) - MIN(s.idx), 0) * 390 as signals_per_day,
        AVG(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END) as avg_return
    FROM read_parquet('workspaces/expansive_grid_search_8d35f12c/traces/*/signals/*/*.parquet') s
    JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index
    JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index
    WHERE s.val != 0
    GROUP BY s.strat
    HAVING COUNT(*) >= 20
)
SELECT 
    CASE 
        WHEN signals_per_day < 1 THEN '< 1 per day'
        WHEN signals_per_day < 5 THEN '1-5 per day'
        WHEN signals_per_day < 10 THEN '5-10 per day'
        WHEN signals_per_day < 20 THEN '10-20 per day'
        ELSE '20+ per day'
    END as frequency_bucket,
    COUNT(*) as strategies,
    ROUND(AVG(avg_return), 4) as avg_return_pct,
    ROUND(AVG(total_signals), 1) as avg_total_signals,
    ROUND(MIN(avg_return), 4) as worst_return,
    ROUND(MAX(avg_return), 4) as best_return
FROM strategy_activity
GROUP BY frequency_bucket
ORDER BY 
    CASE frequency_bucket
        WHEN '< 1 per day' THEN 1
        WHEN '1-5 per day' THEN 2
        WHEN '5-10 per day' THEN 3
        WHEN '10-20 per day' THEN 4
        ELSE 5
    END;