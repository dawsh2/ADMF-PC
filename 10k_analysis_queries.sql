-- Top strategies across all types (10k bars)
WITH strategy_performance AS (
    SELECT 
        s.strat,
        CASE 
            WHEN s.strat LIKE '%breakout%' THEN 'breakout'
            WHEN s.strat LIKE '%ma_crossover%' THEN 'ma_crossover'
            WHEN s.strat LIKE '%mean_reversion%' THEN 'mean_reversion'
            WHEN s.strat LIKE '%momentum%' THEN 'momentum'
            WHEN s.strat LIKE '%rsi%' THEN 'rsi'
        END as strategy_type,
        COUNT(*) as trades,
        AVG(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END) as avg_return,
        STDDEV(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END) as volatility
    FROM read_parquet('workspaces/expansive_grid_search_8d35f12c/traces/*/signals/*/*.parquet') s
    JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index
    JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index
    WHERE s.val != 0
    GROUP BY s.strat
    HAVING COUNT(*) >= 20  -- Higher threshold for 10k bars
)
SELECT 
    strategy_type,
    COUNT(*) as strategies_tested,
    SUM(trades) as total_trades,
    ROUND(AVG(avg_return), 4) as avg_return_pct,
    ROUND(AVG(volatility), 4) as avg_volatility,
    ROUND(AVG(avg_return / NULLIF(volatility, 0)), 3) as avg_sharpe
FROM strategy_performance
GROUP BY strategy_type
ORDER BY avg_sharpe DESC;