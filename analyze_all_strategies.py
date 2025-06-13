import duckdb

# Connect to database
con = duckdb.connect('workspaces/expansive_grid_search_db1cfd51/analytics.duckdb')

# Query all strategy types
query = """
SELECT 
    CASE 
        WHEN strat LIKE '%breakout%' THEN 'breakout'
        WHEN strat LIKE '%ma_crossover%' THEN 'ma_crossover'
        WHEN strat LIKE '%mean_reversion%' THEN 'mean_reversion'
        WHEN strat LIKE '%momentum%' THEN 'momentum'
        WHEN strat LIKE '%rsi%' THEN 'rsi'
    END as strategy_type,
    COUNT(DISTINCT strat) as strategies,
    COUNT(*) as total_signals
FROM read_parquet('workspaces/expansive_grid_search_db1cfd51/traces/SPY_1m/signals/*/*.parquet')
WHERE val != 0
GROUP BY strategy_type
ORDER BY total_signals DESC
"""

result = con.execute(query).df()
print('\nSignal counts by strategy type:')
print(result.to_string(index=False))

# Performance by strategy type
perf_query = """
WITH signals_with_returns AS (
    SELECT 
        s.strat,
        CASE 
            WHEN s.strat LIKE '%breakout%' THEN 'breakout'
            WHEN s.strat LIKE '%ma_crossover%' THEN 'ma_crossover'
            WHEN s.strat LIKE '%mean_reversion%' THEN 'mean_reversion'
            WHEN s.strat LIKE '%momentum%' THEN 'momentum'
            WHEN s.strat LIKE '%rsi%' THEN 'rsi'
        END as strategy_type,
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
    COUNT(*) as trades,
    ROUND(AVG(return_pct), 4) as avg_return,
    ROUND(STDDEV(return_pct), 4) as volatility
FROM signals_with_returns
GROUP BY strategy_type
ORDER BY avg_return DESC
"""

print('\n\nPerformance by strategy type:')
result2 = con.execute(perf_query).df()
print(result2.to_string(index=False))

# Show top strategies across all types
best_query = """
WITH strategy_performance AS (
    SELECT 
        s.strat,
        COUNT(*) as trades,
        AVG(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END) as avg_return
    FROM read_parquet('workspaces/expansive_grid_search_db1cfd51/traces/SPY_1m/signals/*/*.parquet') s
    JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index
    JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index
    WHERE s.val != 0
    GROUP BY s.strat
    HAVING COUNT(*) >= 5  -- At least 5 trades
)
SELECT 
    strat,
    trades,
    ROUND(avg_return, 4) as avg_return_pct
FROM strategy_performance
ORDER BY avg_return DESC
LIMIT 10
"""

print('\n\nTop 10 strategies (with at least 5 trades):')
result3 = con.execute(best_query).df()
print(result3.to_string(index=False))

con.close()