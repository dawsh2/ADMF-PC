-- Trade the best discovered signal combinations
-- Best pattern: Enter on RSI fast, exit on RSI slow

WITH entry_signals AS (
    -- RSI fast entries (7-period)
    SELECT 
        idx as entry_idx,
        ts,
        strat as entry_strategy
    FROM read_parquet('workspaces/expansive_grid_search_8d35f12c/traces/SPY_1m/signals/rsi_grid/*.parquet')
    WHERE val = 1 AND strat LIKE '%_7_%'  -- Fast RSI
),
exit_signals AS (
    -- RSI slow exits (14 or 21 period)
    SELECT 
        idx as exit_idx,
        ts,
        strat as exit_strategy
    FROM read_parquet('workspaces/expansive_grid_search_8d35f12c/traces/SPY_1m/signals/rsi_grid/*.parquet')
    WHERE val = -1 AND (strat LIKE '%_14_%' OR strat LIKE '%_21_%')  -- Slow RSI
),
composite_trades AS (
    SELECT 
        e.entry_idx,
        MIN(x.exit_idx) as exit_idx,
        e.entry_strategy,
        FIRST(x.exit_strategy) as exit_strategy,
        MIN(x.exit_idx) - e.entry_idx as holding_period
    FROM entry_signals e
    JOIN exit_signals x ON x.exit_idx > e.entry_idx AND x.exit_idx <= e.entry_idx + 10
    GROUP BY e.entry_idx, e.entry_strategy
)
SELECT 
    COUNT(*) as total_trades,
    AVG(holding_period) as avg_holding_period,
    AVG((m_exit.close - m_entry.close) / m_entry.close * 100) as avg_return_pct,
    STDDEV((m_exit.close - m_entry.close) / m_entry.close * 100) as volatility,
    MIN((m_exit.close - m_entry.close) / m_entry.close * 100) as worst_trade,
    MAX((m_exit.close - m_entry.close) / m_entry.close * 100) as best_trade,
    -- Alpaca costs: 0 commission + ~1bp slippage
    AVG((m_exit.close - m_entry.close) / m_entry.close * 100 - 0.01) as net_return_alpaca
FROM composite_trades ct
JOIN read_parquet('data/SPY_1m.parquet') m_entry ON ct.entry_idx = m_entry.bar_index
JOIN read_parquet('data/SPY_1m.parquet') m_exit ON ct.exit_idx = m_exit.bar_index;