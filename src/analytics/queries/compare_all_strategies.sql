-- Compare performance across ALL strategy types
WITH all_signals AS (
    SELECT 
        strat,
        val,
        idx,
        -- Extract strategy type from directory structure
        CASE 
            WHEN strat LIKE 'breakout%' THEN 'breakout'
            WHEN strat LIKE 'ma_crossover%' THEN 'ma_crossover'
            WHEN strat LIKE 'mean_reversion%' THEN 'mean_reversion'
            WHEN strat LIKE 'momentum%' THEN 'momentum'
            WHEN strat LIKE 'rsi%' THEN 'rsi'
        END as strategy_type
    FROM read_parquet('workspaces/expansive_grid_search_db1cfd51/traces/SPY_1m/signals/*/*.parquet')
    WHERE val != 0
),
performance AS (
    SELECT 
        s.strategy_type,
        s.strat,
        COUNT(*) as trades,
        AVG(
            CASE 
                WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
            END
        ) as avg_return_pct
    FROM all_signals s
    JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index
    JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index
    GROUP BY s.strategy_type, s.strat
)
SELECT 
    strategy_type,
    COUNT(DISTINCT strat) as variations_tested,
    SUM(trades) as total_trades,
    ROUND(AVG(avg_return_pct), 4) as avg_return_pct,
    ROUND(MIN(avg_return_pct), 4) as worst_variation,
    ROUND(MAX(avg_return_pct), 4) as best_variation
FROM performance
GROUP BY strategy_type
ORDER BY avg_return_pct DESC;