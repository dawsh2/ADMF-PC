-- Analyze strategy performance by market regime
-- Requires classifier signals to be available

WITH classifier_signals AS (
    -- Load market regime classifications
    SELECT 
        idx as bar_index,
        val as regime,
        strat as classifier_name
    FROM read_parquet('workspaces/expansive_grid_search_db1cfd51/traces/SPY_1m/classifiers/*/*.parquet')
    WHERE val != 0
),
regime_windows AS (
    -- Create regime windows (start and end of each regime)
    SELECT 
        regime,
        bar_index as regime_start,
        LEAD(bar_index, 1, 999999) OVER (ORDER BY bar_index) as regime_end
    FROM classifier_signals
),
strategy_performance_by_regime AS (
    SELECT 
        s.strat,
        rw.regime,
        COUNT(*) as trades,
        AVG(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END) as avg_return,
        STDDEV(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END) as volatility
    FROM read_parquet('workspaces/expansive_grid_search_db1cfd51/traces/SPY_1m/signals/*/*.parquet') s
    JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index
    JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index
    JOIN regime_windows rw ON s.idx >= rw.regime_start AND s.idx < rw.regime_end
    WHERE s.val != 0
    GROUP BY s.strat, rw.regime
    HAVING COUNT(*) >= 3  -- At least 3 trades in regime
)
SELECT 
    regime,
    COUNT(DISTINCT strat) as strategies,
    SUM(trades) as total_trades,
    ROUND(AVG(avg_return), 4) as avg_return,
    -- Best strategies per regime
    MAX(CASE WHEN avg_return > 0 THEN strat END) as best_strategy
FROM strategy_performance_by_regime
GROUP BY regime
ORDER BY regime;