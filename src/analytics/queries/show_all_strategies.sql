SELECT 
    strat,
    COUNT(*) as signal_changes
FROM read_parquet('workspaces/expansive_grid_search_db1cfd51/traces/SPY_1m/signals/*/*.parquet') 
WHERE val != 0 
GROUP BY strat 
ORDER BY signal_changes DESC 
LIMIT 20;