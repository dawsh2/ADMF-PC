-- Simple Strategy Performance Per Regime - Direct Signal Analysis
-- Focus on signal distribution and basic performance metrics

PRAGMA memory_limit='3GB';
SET threads=4;

SELECT 
    '=== SIMPLE REGIME-BASED SIGNAL ANALYSIS ===' as header;

-- Check signal distribution by regime for April 2024
SELECT 
    '=== MACD SIGNALS BY REGIME - APRIL 2024 ===' as section_header;

WITH april_signals AS (
    SELECT 
        s.ts::timestamp as signal_time,
        s.val as signal_value
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-04-01 00:00:00'
      AND s.ts::timestamp < '2024-05-01 00:00:00'
),
april_regimes AS (
    SELECT 
        c.ts::timestamp as regime_time,
        c.val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
    WHERE c.ts::timestamp >= '2024-04-01 00:00:00'
      AND c.ts::timestamp < '2024-05-01 00:00:00'
),
-- Find regime state for each signal (using nearest timestamp)
signals_with_regime AS (
    SELECT 
        s.signal_time,
        s.signal_value,
        -- Find closest regime timestamp (within 5 minutes)
        (SELECT r.regime_state 
         FROM april_regimes r 
         WHERE ABS(EXTRACT(EPOCH FROM (r.regime_time - s.signal_time))) <= 300
         ORDER BY ABS(EXTRACT(EPOCH FROM (r.regime_time - s.signal_time)))
         LIMIT 1) as regime_state
    FROM april_signals s
)
SELECT 
    regime_state,
    signal_value,
    COUNT(*) as signal_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY regime_state), 1) as pct_of_regime,
    MIN(signal_time) as first_signal,
    MAX(signal_time) as last_signal
FROM signals_with_regime
WHERE regime_state IS NOT NULL
GROUP BY regime_state, signal_value
ORDER BY regime_state, signal_value;

-- Count signal changes by regime (actual trading signals)
SELECT 
    '=== SIGNAL CHANGES BY REGIME - APRIL 2024 ===' as section_header;

WITH april_signals AS (
    SELECT 
        s.ts::timestamp as signal_time,
        s.val as signal_value,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-04-01 00:00:00'
      AND s.ts::timestamp < '2024-05-01 00:00:00'
),
april_regimes AS (
    SELECT 
        c.ts::timestamp as regime_time,
        c.val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
    WHERE c.ts::timestamp >= '2024-04-01 00:00:00'
      AND c.ts::timestamp < '2024-05-01 00:00:00'
),
signal_changes AS (
    SELECT 
        signal_time,
        prev_signal,
        signal_value,
        CASE 
            WHEN prev_signal = -1 AND signal_value = 1 THEN 'SHORT_TO_LONG'
            WHEN prev_signal = 1 AND signal_value = -1 THEN 'LONG_TO_SHORT'
            ELSE 'NO_CHANGE'
        END as signal_change_type
    FROM april_signals
    WHERE prev_signal IS NOT NULL
      AND prev_signal != signal_value
),
changes_with_regime AS (
    SELECT 
        sc.*,
        (SELECT r.regime_state 
         FROM april_regimes r 
         WHERE ABS(EXTRACT(EPOCH FROM (r.regime_time - sc.signal_time))) <= 300
         ORDER BY ABS(EXTRACT(EPOCH FROM (r.regime_time - sc.signal_time)))
         LIMIT 1) as regime_state
    FROM signal_changes sc
)
SELECT 
    regime_state,
    signal_change_type,
    COUNT(*) as change_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct_of_all_changes,
    MIN(signal_time) as first_change,
    MAX(signal_time) as last_change
FROM changes_with_regime
WHERE regime_state IS NOT NULL
  AND signal_change_type != 'NO_CHANGE'
GROUP BY regime_state, signal_change_type
ORDER BY regime_state, change_count DESC;

-- Simple return analysis for signal changes
SELECT 
    '=== SIMPLE RETURN ANALYSIS BY REGIME - APRIL 2024 ===' as section_header;

WITH april_market_data AS (
    SELECT 
        timestamp,
        close
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE timestamp >= '2024-04-01 00:00:00'
      AND timestamp < '2024-05-01 00:00:00'
),
april_signals AS (
    SELECT 
        s.ts::timestamp as signal_time,
        s.val as signal_value,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-04-01 00:00:00'
      AND s.ts::timestamp < '2024-05-01 00:00:00'
),
april_regimes AS (
    SELECT 
        c.ts::timestamp as regime_time,
        c.val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
    WHERE c.ts::timestamp >= '2024-04-01 00:00:00'
      AND c.ts::timestamp < '2024-05-01 00:00:00'
),
signal_changes AS (
    SELECT 
        signal_time,
        signal_value,
        prev_signal,
        LEAD(signal_time) OVER (ORDER BY signal_time) as next_signal_time,
        CASE 
            WHEN prev_signal = -1 AND signal_value = 1 THEN 'ENTER_LONG'
            WHEN prev_signal = 1 AND signal_value = -1 THEN 'ENTER_SHORT'
        END as trade_type
    FROM april_signals
    WHERE prev_signal IS NOT NULL
      AND prev_signal != signal_value
),
trades_with_regime AS (
    SELECT 
        sc.*,
        m1.close as entry_price,
        m2.close as exit_price,
        (SELECT r.regime_state 
         FROM april_regimes r 
         WHERE ABS(EXTRACT(EPOCH FROM (r.regime_time - sc.signal_time))) <= 300
         ORDER BY ABS(EXTRACT(EPOCH FROM (r.regime_time - sc.signal_time)))
         LIMIT 1) as entry_regime,
        CASE 
            WHEN sc.trade_type = 'ENTER_LONG' THEN (m2.close - m1.close) / m1.close
            WHEN sc.trade_type = 'ENTER_SHORT' THEN (m1.close - m2.close) / m1.close
            ELSE NULL
        END as trade_return
    FROM signal_changes sc
    LEFT JOIN april_market_data m1 ON sc.signal_time = m1.timestamp
    LEFT JOIN april_market_data m2 ON sc.next_signal_time = m2.timestamp
    WHERE sc.trade_type IS NOT NULL
      AND sc.next_signal_time IS NOT NULL
)
SELECT 
    entry_regime,
    trade_type,
    COUNT(*) as trade_count,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(SUM(trade_return) * 100, 3) as total_return_pct,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    ROUND(MAX(trade_return) * 100, 2) as best_trade_pct,
    ROUND(MIN(trade_return) * 100, 2) as worst_trade_pct
FROM trades_with_regime
WHERE entry_regime IS NOT NULL
  AND entry_price IS NOT NULL
  AND exit_price IS NOT NULL
GROUP BY entry_regime, trade_type
ORDER BY entry_regime, total_return_pct DESC;