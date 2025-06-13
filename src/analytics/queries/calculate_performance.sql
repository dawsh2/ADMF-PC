-- Calculate real strategy performance with market data
WITH signal_prices AS (
    SELECT 
        s.strat,
        s.idx as signal_bar,
        s.val as signal_value,
        m.close as entry_price,
        LEAD(m.close, 1) OVER (ORDER BY s.idx) as next_price
    FROM read_parquet('workspaces/expansive_grid_search_db1cfd51/traces/SPY_1m/signals/rsi_grid/*.parquet') s
    JOIN read_parquet('data/SPY_1m.parquet') m ON s.idx = m.bar_index
    WHERE s.val != 0  -- Only actual signals, not flat
)
SELECT 
    strat,
    COUNT(*) as trades,
    ROUND(AVG(CASE 
        WHEN signal_value = 1 THEN (next_price - entry_price) / entry_price * 100
        WHEN signal_value = -1 THEN (entry_price - next_price) / entry_price * 100 
        ELSE 0
    END), 3) as avg_return_pct,
    ROUND(MIN(entry_price), 2) as min_entry_price,
    ROUND(MAX(entry_price), 2) as max_entry_price
FROM signal_prices
WHERE next_price IS NOT NULL
GROUP BY strat
HAVING COUNT(*) >= 3
ORDER BY avg_return_pct DESC;