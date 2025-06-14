-- Analyze parameter stability for RSI strategies (10k bars)
WITH rsi_params AS (
    SELECT 
        strat,
        CAST(REGEXP_EXTRACT(strat, 'rsi_grid_([0-9]+)_', 1) AS INTEGER) as period,
        CAST(REGEXP_EXTRACT(strat, 'rsi_grid_[0-9]+_([0-9]+)_', 1) AS INTEGER) as oversold,
        CAST(REGEXP_EXTRACT(strat, 'rsi_grid_[0-9]+_[0-9]+_([0-9]+)', 1) AS INTEGER) as overbought
    FROM (SELECT DISTINCT strat FROM read_parquet('workspaces/expansive_grid_search_8d35f12c/traces/*/signals/rsi_grid/*.parquet'))
),
performance AS (
    SELECT 
        p.period,
        p.oversold,
        p.overbought,
        COUNT(*) as trades,
        AVG(CASE 
            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
        END) as avg_return
    FROM read_parquet('workspaces/expansive_grid_search_8d35f12c/traces/*/signals/rsi_grid/*.parquet') s
    JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index
    JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index
    JOIN rsi_params p ON s.strat = p.strat
    WHERE s.val != 0
    GROUP BY p.period, p.oversold, p.overbought
),
neighborhood_analysis AS (
    SELECT 
        p1.period,
        p1.oversold,
        p1.overbought,
        p1.trades,
        p1.avg_return,
        COUNT(p2.avg_return) as neighbors,
        AVG(p2.avg_return) as neighborhood_avg
    FROM performance p1
    JOIN performance p2 ON 
        ABS(p1.period - p2.period) <= 2 AND
        ABS(p1.oversold - p2.oversold) <= 5 AND
        ABS(p1.overbought - p2.overbought) <= 5
    WHERE p1.avg_return > 0
    GROUP BY p1.period, p1.oversold, p1.overbought, p1.trades, p1.avg_return
)
SELECT 
    period,
    oversold,
    overbought,
    trades,
    ROUND(avg_return, 4) as avg_return_pct,
    neighbors,
    ROUND(neighborhood_avg, 4) as neighborhood_avg_pct
FROM neighborhood_analysis
WHERE neighbors >= 3 AND neighborhood_avg > 0
ORDER BY neighborhood_avg DESC, avg_return DESC
LIMIT 10;