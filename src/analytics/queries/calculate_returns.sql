-- Calculate actual returns for each signal
SELECT 
    s.strat,
    s.val as signal_direction,
    m1.close as entry_price,
    m2.close as exit_price,
    -- Calculate return based on signal direction
    CASE 
        WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100  -- Long: profit if price goes up
        WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100  -- Short: profit if price goes down
    END as return_pct
FROM read_parquet('workspaces/expansive_grid_search_db1cfd51/traces/SPY_1m/signals/rsi_grid/*.parquet') s
JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index      -- Entry price
JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index  -- Exit price (next bar)
WHERE s.val != 0  -- Only actual signals
LIMIT 10;