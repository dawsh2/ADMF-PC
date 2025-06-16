-- Strategy Performance Per Regime - FINAL VERSION
-- Properly handle sparse regime data with forward-fill

PRAGMA memory_limit='3GB';
SET threads=4;

SELECT 
    '=== STRATEGY PERFORMANCE PER REGIME - FINAL ===' as header;

-- Analyze one month with proper forward-fill
WITH 
-- Get April 2024 market data with timezone correction
market_minutes AS (
    SELECT 
        timestamp - INTERVAL 4 HOUR as timestamp_est,
        close
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE timestamp >= '2024-04-01 04:00:00'
      AND timestamp < '2024-05-01 04:00:00'
),
-- Get MACD signals with timezone correction
macd_signals AS (
    SELECT 
        s.ts::timestamp - INTERVAL 4 HOUR as signal_time_est,
        s.val as signal_value,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-04-01 04:00:00'
      AND s.ts::timestamp < '2024-05-01 04:00:00'
),
-- Get sparse regime data with timezone correction
regime_changes AS (
    SELECT 
        c.ts::timestamp - INTERVAL 4 HOUR as regime_time_est,
        c.val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
    WHERE c.ts::timestamp >= '2024-04-01 00:00:00'  -- Start earlier to catch initial state
      AND c.ts::timestamp < '2024-05-01 04:00:00'
),
-- Create forward-filled regime timeline
regime_timeline AS (
    SELECT 
        m.timestamp_est,
        LAST_VALUE(r.regime_state IGNORE NULLS) OVER (
            ORDER BY m.timestamp_est 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as current_regime
    FROM market_minutes m
    LEFT JOIN regime_changes r ON m.timestamp_est = r.regime_time_est
),
-- Identify trades from signal changes
trades AS (
    SELECT 
        signal_time_est as entry_time,
        signal_value as entry_signal,
        LEAD(signal_time_est) OVER (ORDER BY signal_time_est) as exit_time,
        LEAD(signal_value) OVER (ORDER BY signal_time_est) as exit_signal,
        CASE 
            WHEN signal_value = 1 THEN 'LONG'
            WHEN signal_value = -1 THEN 'SHORT'
        END as trade_side
    FROM macd_signals
    WHERE prev_signal IS NOT NULL
      AND prev_signal != signal_value  -- Only actual signal changes
),
-- Match trades with regimes and calculate returns
trades_with_performance AS (
    SELECT 
        t.*,
        rt.current_regime as entry_regime,
        m1.close as entry_price,
        m2.close as exit_price,
        EXTRACT(EPOCH FROM (t.exit_time - t.entry_time)) / 60.0 as duration_minutes,
        CASE 
            WHEN t.entry_signal = 1 THEN (m2.close - m1.close) / m1.close
            WHEN t.entry_signal = -1 THEN (m1.close - m2.close) / m1.close
        END as trade_return
    FROM trades t
    LEFT JOIN regime_timeline rt ON t.entry_time = rt.timestamp_est
    LEFT JOIN market_minutes m1 ON t.entry_time = m1.timestamp_est
    LEFT JOIN market_minutes m2 ON t.exit_time = m2.timestamp_est
    WHERE t.exit_time IS NOT NULL
      AND rt.current_regime IS NOT NULL
      AND m1.close IS NOT NULL
      AND m2.close IS NOT NULL
)
-- Final performance metrics by regime
SELECT 
    entry_regime,
    trade_side,
    COUNT(*) as trade_count,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(STDDEV(trade_return) * 100, 3) as return_volatility_pct,
    ROUND(SUM(trade_return) * 100, 3) as total_return_pct,
    ROUND(AVG(trade_return) / NULLIF(STDDEV(trade_return), 0), 3) as trade_sharpe,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    ROUND(AVG(duration_minutes), 1) as avg_duration_min,
    ROUND(MAX(trade_return) * 100, 2) as best_trade_pct,
    ROUND(MIN(trade_return) * 100, 2) as worst_trade_pct,
    -- After transaction costs (0.05% round-trip)
    ROUND(SUM(trade_return - 0.0005) * 100, 3) as net_return_pct
FROM trades_with_performance
GROUP BY entry_regime, trade_side
ORDER BY entry_regime, net_return_pct DESC;

-- Summary by regime only
SELECT 
    '=== PERFORMANCE SUMMARY BY REGIME ===' as section_header;

WITH 
market_minutes AS (
    SELECT 
        timestamp - INTERVAL 4 HOUR as timestamp_est,
        close
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE timestamp >= '2024-04-01 04:00:00'
      AND timestamp < '2024-05-01 04:00:00'
),
macd_signals AS (
    SELECT 
        s.ts::timestamp - INTERVAL 4 HOUR as signal_time_est,
        s.val as signal_value,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-04-01 04:00:00'
      AND s.ts::timestamp < '2024-05-01 04:00:00'
),
regime_changes AS (
    SELECT 
        c.ts::timestamp - INTERVAL 4 HOUR as regime_time_est,
        c.val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
    WHERE c.ts::timestamp >= '2024-04-01 00:00:00'
      AND c.ts::timestamp < '2024-05-01 04:00:00'
),
regime_timeline AS (
    SELECT 
        m.timestamp_est,
        LAST_VALUE(r.regime_state IGNORE NULLS) OVER (
            ORDER BY m.timestamp_est 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as current_regime
    FROM market_minutes m
    LEFT JOIN regime_changes r ON m.timestamp_est = r.regime_time_est
),
trades AS (
    SELECT 
        signal_time_est as entry_time,
        signal_value as entry_signal,
        LEAD(signal_time_est) OVER (ORDER BY signal_time_est) as exit_time
    FROM macd_signals
    WHERE prev_signal IS NOT NULL
      AND prev_signal != signal_value
),
trades_with_performance AS (
    SELECT 
        t.*,
        rt.current_regime as entry_regime,
        m1.close as entry_price,
        m2.close as exit_price,
        CASE 
            WHEN t.entry_signal = 1 THEN (m2.close - m1.close) / m1.close
            WHEN t.entry_signal = -1 THEN (m1.close - m2.close) / m1.close
        END as trade_return
    FROM trades t
    LEFT JOIN regime_timeline rt ON t.entry_time = rt.timestamp_est
    LEFT JOIN market_minutes m1 ON t.entry_time = m1.timestamp_est
    LEFT JOIN market_minutes m2 ON t.exit_time = m2.timestamp_est
    WHERE t.exit_time IS NOT NULL
      AND rt.current_regime IS NOT NULL
      AND m1.close IS NOT NULL
      AND m2.close IS NOT NULL
)
SELECT 
    entry_regime,
    COUNT(*) as total_trades,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(SUM(trade_return) * 100, 3) as gross_return_pct,
    ROUND(SUM(trade_return - 0.0005) * 100, 3) as net_return_pct,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    ROUND(AVG(trade_return) / NULLIF(STDDEV(trade_return), 0), 3) as sharpe_ratio
FROM trades_with_performance
GROUP BY entry_regime
ORDER BY net_return_pct DESC;