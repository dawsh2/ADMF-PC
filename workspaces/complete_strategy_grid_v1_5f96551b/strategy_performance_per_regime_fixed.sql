-- Strategy Performance Per Regime - WITH TIMEZONE FIX
-- Apply -4 hour correction to align UTC timestamps with EST data

PRAGMA memory_limit='3GB';
SET threads=4;

SELECT 
    '=== STRATEGY PERFORMANCE PER REGIME - TIMEZONE CORRECTED ===' as header;

-- Analyze with proper timezone alignment
SELECT 
    '=== PERFORMANCE BY REGIME - APRIL 2024 ===' as section_header;

WITH april_market_data AS (
    SELECT 
        timestamp - INTERVAL 4 HOUR as timestamp_est,  -- Apply timezone correction
        close
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE timestamp >= '2024-04-01 04:00:00'  -- Adjust for timezone
      AND timestamp < '2024-05-01 04:00:00'
),
april_signals AS (
    SELECT 
        s.ts::timestamp - INTERVAL 4 HOUR as signal_time_est,  -- Apply timezone correction
        s.val as signal_value,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-04-01 04:00:00'  -- Adjust for timezone
      AND s.ts::timestamp < '2024-05-01 04:00:00'
),
april_regimes AS (
    SELECT 
        c.ts::timestamp - INTERVAL 4 HOUR as regime_time_est,  -- Apply timezone correction
        c.val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
    WHERE c.ts::timestamp >= '2024-04-01 04:00:00'  -- Adjust for timezone
      AND c.ts::timestamp < '2024-05-01 04:00:00'
),
-- Identify trades (signal changes)
trades AS (
    SELECT 
        signal_time_est as entry_time,
        signal_value as entry_signal,
        prev_signal,
        LEAD(signal_time_est) OVER (ORDER BY signal_time_est) as exit_time,
        LEAD(signal_value) OVER (ORDER BY signal_time_est) as exit_signal
    FROM april_signals
    WHERE prev_signal IS NOT NULL
      AND prev_signal != signal_value  -- Only actual signal changes
),
-- Join with regime data
trades_with_regime AS (
    SELECT 
        t.*,
        ar.regime_state as entry_regime
    FROM trades t
    LEFT JOIN april_regimes ar ON t.entry_time = ar.regime_time_est
    WHERE ar.regime_state IS NOT NULL
),
-- Calculate returns
trades_with_returns AS (
    SELECT 
        tr.*,
        m_entry.close as entry_price,
        m_exit.close as exit_price,
        EXTRACT(EPOCH FROM (tr.exit_time - tr.entry_time)) / 60.0 as duration_minutes,
        CASE 
            WHEN tr.entry_signal = 1 THEN (m_exit.close - m_entry.close) / m_entry.close
            WHEN tr.entry_signal = -1 THEN (m_entry.close - m_exit.close) / m_entry.close
            ELSE 0
        END as trade_return,
        CASE 
            WHEN tr.entry_signal = 1 THEN 'LONG'
            WHEN tr.entry_signal = -1 THEN 'SHORT'
            ELSE 'NEUTRAL'
        END as trade_side
    FROM trades_with_regime tr
    LEFT JOIN april_market_data m_entry ON tr.entry_time = m_entry.timestamp_est
    LEFT JOIN april_market_data m_exit ON tr.exit_time = m_exit.timestamp_est
    WHERE tr.exit_time IS NOT NULL
      AND m_entry.close IS NOT NULL
      AND m_exit.close IS NOT NULL
)
SELECT 
    entry_regime,
    trade_side,
    COUNT(*) as trade_count,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(STDDEV(trade_return) * 100, 3) as return_volatility_pct,
    ROUND(SUM(trade_return) * 100, 3) as cumulative_return_pct,
    ROUND(AVG(trade_return) / NULLIF(STDDEV(trade_return), 0), 3) as trade_sharpe,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    ROUND(AVG(duration_minutes), 1) as avg_duration_min,
    ROUND(MAX(trade_return) * 100, 2) as best_trade_pct,
    ROUND(MIN(trade_return) * 100, 2) as worst_trade_pct
FROM trades_with_returns
GROUP BY entry_regime, trade_side
ORDER BY entry_regime, cumulative_return_pct DESC;

-- Overall regime performance summary
SELECT 
    '=== OVERALL REGIME PERFORMANCE SUMMARY ===' as section_header;

WITH april_market_data AS (
    SELECT 
        timestamp - INTERVAL 4 HOUR as timestamp_est,
        close
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE timestamp >= '2024-04-01 04:00:00'
      AND timestamp < '2024-05-01 04:00:00'
),
april_signals AS (
    SELECT 
        s.ts::timestamp - INTERVAL 4 HOUR as signal_time_est,
        s.val as signal_value,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-04-01 04:00:00'
      AND s.ts::timestamp < '2024-05-01 04:00:00'
),
april_regimes AS (
    SELECT 
        c.ts::timestamp - INTERVAL 4 HOUR as regime_time_est,
        c.val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
    WHERE c.ts::timestamp >= '2024-04-01 04:00:00'
      AND c.ts::timestamp < '2024-05-01 04:00:00'
),
trades AS (
    SELECT 
        signal_time_est as entry_time,
        signal_value as entry_signal,
        prev_signal,
        LEAD(signal_time_est) OVER (ORDER BY signal_time_est) as exit_time
    FROM april_signals
    WHERE prev_signal IS NOT NULL
      AND prev_signal != signal_value
),
trades_with_regime AS (
    SELECT 
        t.*,
        ar.regime_state as entry_regime
    FROM trades t
    LEFT JOIN april_regimes ar ON t.entry_time = ar.regime_time_est
    WHERE ar.regime_state IS NOT NULL
),
trades_with_returns AS (
    SELECT 
        tr.*,
        m_entry.close as entry_price,
        m_exit.close as exit_price,
        CASE 
            WHEN tr.entry_signal = 1 THEN (m_exit.close - m_entry.close) / m_entry.close
            WHEN tr.entry_signal = -1 THEN (m_entry.close - m_exit.close) / m_entry.close
            ELSE 0
        END as trade_return
    FROM trades_with_regime tr
    LEFT JOIN april_market_data m_entry ON tr.entry_time = m_entry.timestamp_est
    LEFT JOIN april_market_data m_exit ON tr.exit_time = m_exit.timestamp_est
    WHERE tr.exit_time IS NOT NULL
      AND m_entry.close IS NOT NULL
      AND m_exit.close IS NOT NULL
)
SELECT 
    entry_regime,
    COUNT(*) as total_trades,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(SUM(trade_return) * 100, 3) as cumulative_return_pct,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    ROUND(MAX(trade_return) * 100, 2) as best_trade_pct,
    ROUND(MIN(trade_return) * 100, 2) as worst_trade_pct,
    -- Apply 0.05% round-trip transaction cost
    ROUND(SUM(trade_return - 0.0005) * 100, 3) as net_return_after_costs_pct
FROM trades_with_returns
GROUP BY entry_regime
ORDER BY net_return_after_costs_pct DESC;

-- Check trading hours distribution 
SELECT 
    '=== TRADING HOURS DISTRIBUTION (EST) ===' as section_header;

WITH april_signals AS (
    SELECT 
        s.ts::timestamp - INTERVAL 4 HOUR as signal_time_est,
        s.val as signal_value,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-04-01 04:00:00'
      AND s.ts::timestamp < '2024-05-01 04:00:00'
),
trade_times AS (
    SELECT 
        signal_time_est,
        EXTRACT(HOUR FROM signal_time_est) as hour_est
    FROM april_signals
    WHERE prev_signal IS NOT NULL
      AND prev_signal != signal_value
)
SELECT 
    hour_est,
    COUNT(*) as trade_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct_of_trades
FROM trade_times
WHERE hour_est BETWEEN 9 AND 16  -- Market hours
GROUP BY hour_est
ORDER BY hour_est;