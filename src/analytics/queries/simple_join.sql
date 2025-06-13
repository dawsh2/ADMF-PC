SELECT 
    s.strat,
    COUNT(*) as signal_changes,
    ROUND(AVG(m.close), 2) as avg_price_at_signal
FROM read_parquet('workspaces/expansive_grid_search_db1cfd51/traces/SPY_1m/signals/rsi_grid/*.parquet') s
JOIN read_parquet('data/SPY_1m.parquet') m ON s.idx = m.bar_index
WHERE s.val != 0
GROUP BY s.strat
ORDER BY signal_changes DESC
LIMIT 5;