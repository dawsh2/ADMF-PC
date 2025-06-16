-- Strategy Performance Per Regime Analysis - Batched Approach
-- Process data in monthly batches to handle 10 months of data

PRAGMA memory_limit='3GB';
SET threads=4;

SELECT 
    '=== STRATEGY PERFORMANCE PER REGIME - BATCHED ANALYSIS ===' as header;

-- First, let's analyze one month (April 2024) for detailed regime performance
SELECT 
    '=== APRIL 2024 BATCH: REGIME PERFORMANCE ANALYSIS ===' as section_header;

WITH april_2024_timeframe AS (
    SELECT 
        timestamp,
        open,
        high,
        low,
        close,
        volume
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE timestamp >= '2024-04-01 00:00:00'
      AND timestamp < '2024-05-01 00:00:00'
),
-- Get regime states for April 2024
april_regimes AS (
    SELECT 
        c.ts::timestamp as timestamp,
        c.val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
    WHERE c.ts::timestamp >= '2024-04-01 00:00:00'
      AND c.ts::timestamp < '2024-05-01 00:00:00'
),
-- Sample one high-frequency strategy for detailed analysis
sample_strategy_signals AS (
    SELECT 
        s.ts::timestamp as signal_time,
        s.val as signal_value,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-04-01 00:00:00'
      AND s.ts::timestamp < '2024-05-01 00:00:00'
),
-- Identify trade signals (position changes)
trade_signals AS (
    SELECT 
        signal_time,
        signal_value,
        prev_signal,
        CASE 
            WHEN prev_signal IS NOT NULL AND prev_signal != signal_value THEN 1 
            ELSE 0 
        END as is_trade_signal
    FROM sample_strategy_signals
),
-- Get regime at each trade signal
signals_with_regime AS (
    SELECT 
        ts.signal_time,
        ts.signal_value,
        ts.prev_signal,
        ar.regime_state
    FROM trade_signals ts
    LEFT JOIN april_regimes ar ON ts.signal_time = ar.timestamp
    WHERE ts.is_trade_signal = 1
      AND ar.regime_state IS NOT NULL
),
-- Construct trades from consecutive signals
trades_constructed AS (
    SELECT 
        signal_time as entry_time,
        signal_value as entry_signal,
        regime_state as entry_regime,
        LEAD(signal_time) OVER (ORDER BY signal_time) as exit_time,
        LEAD(signal_value) OVER (ORDER BY signal_time) as exit_signal,
        LEAD(regime_state) OVER (ORDER BY signal_time) as exit_regime
    FROM signals_with_regime
    ORDER BY signal_time
),
-- Calculate trade returns
trades_with_returns AS (
    SELECT 
        tc.*,
        m_entry.close as entry_price,
        m_exit.close as exit_price,
        EXTRACT(EPOCH FROM (tc.exit_time - tc.entry_time)) / 60.0 as duration_minutes,
        CASE 
            WHEN tc.entry_signal = 1 THEN (m_exit.close - m_entry.close) / m_entry.close
            WHEN tc.entry_signal = -1 THEN (m_entry.close - m_exit.close) / m_entry.close
            ELSE 0
        END as trade_return,
        CASE 
            WHEN tc.entry_signal = 1 THEN 'LONG'
            WHEN tc.entry_signal = -1 THEN 'SHORT'
            ELSE 'NEUTRAL'
        END as trade_side
    FROM trades_constructed tc
    LEFT JOIN april_2024_timeframe m_entry ON tc.entry_time = m_entry.timestamp
    LEFT JOIN april_2024_timeframe m_exit ON tc.exit_time = m_exit.timestamp
    WHERE tc.exit_time IS NOT NULL
      AND tc.entry_signal IN (-1, 1)
      AND m_entry.close IS NOT NULL
      AND m_exit.close IS NOT NULL
)
SELECT 
    'MACD_12_26_9' as strategy_name,
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
    ROUND(MIN(trade_return) * 100, 2) as worst_trade_pct,
    MIN(entry_time) as first_trade,
    MAX(exit_time) as last_trade
FROM trades_with_returns
GROUP BY entry_regime, trade_side
ORDER BY entry_regime, trade_side;

-- Summary across all regimes for April 2024
SELECT 
    '=== APRIL 2024: OVERALL REGIME PERFORMANCE SUMMARY ===' as section_header;

WITH april_2024_timeframe AS (
    SELECT timestamp, close
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE timestamp >= '2024-04-01 00:00:00'
      AND timestamp < '2024-05-01 00:00:00'
),
april_regimes AS (
    SELECT 
        c.ts::timestamp as timestamp,
        c.val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
    WHERE c.ts::timestamp >= '2024-04-01 00:00:00'
      AND c.ts::timestamp < '2024-05-01 00:00:00'
),
sample_strategy_signals AS (
    SELECT 
        s.ts::timestamp as signal_time,
        s.val as signal_value,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-04-01 00:00:00'
      AND s.ts::timestamp < '2024-05-01 00:00:00'
),
trade_signals AS (
    SELECT 
        signal_time,
        signal_value,
        prev_signal,
        CASE 
            WHEN prev_signal IS NOT NULL AND prev_signal != signal_value THEN 1 
            ELSE 0 
        END as is_trade_signal
    FROM sample_strategy_signals
),
signals_with_regime AS (
    SELECT 
        ts.signal_time,
        ts.signal_value,
        ar.regime_state
    FROM trade_signals ts
    LEFT JOIN april_regimes ar ON ts.signal_time = ar.timestamp
    WHERE ts.is_trade_signal = 1
      AND ar.regime_state IS NOT NULL
),
trades_constructed AS (
    SELECT 
        signal_time as entry_time,
        signal_value as entry_signal,
        regime_state as entry_regime,
        LEAD(signal_time) OVER (ORDER BY signal_time) as exit_time
    FROM signals_with_regime
    ORDER BY signal_time
),
trades_with_returns AS (
    SELECT 
        tc.*,
        m_entry.close as entry_price,
        m_exit.close as exit_price,
        CASE 
            WHEN tc.entry_signal = 1 THEN (m_exit.close - m_entry.close) / m_entry.close
            WHEN tc.entry_signal = -1 THEN (m_entry.close - m_exit.close) / m_entry.close
            ELSE 0
        END as trade_return
    FROM trades_constructed tc
    LEFT JOIN april_2024_timeframe m_entry ON tc.entry_time = m_entry.timestamp
    LEFT JOIN april_2024_timeframe m_exit ON tc.exit_time = m_exit.timestamp
    WHERE tc.exit_time IS NOT NULL
      AND tc.entry_signal IN (-1, 1)
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
    ROUND(MIN(trade_return) * 100, 2) as worst_trade_pct
FROM trades_with_returns
GROUP BY entry_regime
ORDER BY cumulative_return_pct DESC;

-- Regime distribution for April 2024
SELECT 
    '=== APRIL 2024: REGIME TIME DISTRIBUTION ===' as section_header;

SELECT 
    c.val as regime_state,
    COUNT(*) as minutes_in_regime,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage_of_month,
    ROUND(COUNT(*) / 60.0, 1) as hours_in_regime
FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
WHERE c.ts::timestamp >= '2024-04-01 00:00:00'
  AND c.ts::timestamp < '2024-05-01 00:00:00'
GROUP BY c.val
ORDER BY COUNT(*) DESC;