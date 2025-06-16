-- Debug regime matching issue
-- Check if we have regime data at signal times

PRAGMA memory_limit='3GB';
SET threads=4;

SELECT 
    '=== DEBUG: REGIME MATCHING ANALYSIS ===' as header;

-- Check sample of signal times vs regime times
SELECT 
    '=== SAMPLE SIGNAL TIMES VS REGIME DATA ===' as section_header;

WITH sample_signals AS (
    SELECT 
        s.ts::timestamp - INTERVAL 4 HOUR as signal_time_est,
        s.val as signal_value,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-04-01 04:00:00'
      AND s.ts::timestamp < '2024-04-02 04:00:00'  -- Just one day
    LIMIT 10
),
regime_coverage AS (
    SELECT 
        ss.signal_time_est,
        ss.signal_value,
        -- Check if we have exact regime match
        (SELECT c.val 
         FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
         WHERE c.ts::timestamp - INTERVAL 4 HOUR = ss.signal_time_est
         LIMIT 1) as exact_regime_match,
        -- Check nearest regime within 1 minute
        (SELECT c.val 
         FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
         WHERE ABS(EXTRACT(EPOCH FROM ((c.ts::timestamp - INTERVAL 4 HOUR) - ss.signal_time_est))) <= 60
         ORDER BY ABS(EXTRACT(EPOCH FROM ((c.ts::timestamp - INTERVAL 4 HOUR) - ss.signal_time_est)))
         LIMIT 1) as nearest_regime_1min
    FROM sample_signals ss
)
SELECT * FROM regime_coverage;

-- Check regime data frequency
SELECT 
    '=== REGIME DATA FREQUENCY CHECK ===' as section_header;

WITH april_regimes AS (
    SELECT 
        c.ts::timestamp - INTERVAL 4 HOUR as regime_time_est,
        c.val as regime_state,
        LAG(c.ts::timestamp - INTERVAL 4 HOUR) OVER (ORDER BY c.ts::timestamp) as prev_regime_time
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
    WHERE c.ts::timestamp >= '2024-04-01 04:00:00'
      AND c.ts::timestamp < '2024-04-02 04:00:00'  -- Just one day
)
SELECT 
    regime_state,
    COUNT(*) as change_count,
    ROUND(AVG(EXTRACT(EPOCH FROM (regime_time_est - prev_regime_time)) / 60.0), 1) as avg_minutes_between_changes,
    MIN(regime_time_est) as first_change,
    MAX(regime_time_est) as last_change
FROM april_regimes
WHERE prev_regime_time IS NOT NULL
GROUP BY regime_state
ORDER BY change_count DESC;

-- Use forward-fill approach for regime matching
SELECT 
    '=== FORWARD-FILL REGIME MATCHING ===' as section_header;

WITH signal_trades AS (
    SELECT 
        s.ts::timestamp - INTERVAL 4 HOUR as entry_time_est,
        s.val as entry_signal,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal,
        LEAD(s.ts::timestamp - INTERVAL 4 HOUR) OVER (ORDER BY s.ts::timestamp) as exit_time_est
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-04-01 04:00:00'
      AND s.ts::timestamp < '2024-04-15 04:00:00'  -- Two weeks
),
regime_sparse AS (
    SELECT 
        c.ts::timestamp - INTERVAL 4 HOUR as regime_time_est,
        c.val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
    WHERE c.ts::timestamp >= '2024-04-01 04:00:00'
      AND c.ts::timestamp < '2024-04-15 04:00:00'
),
market_data AS (
    SELECT 
        timestamp - INTERVAL 4 HOUR as timestamp_est,
        close
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE timestamp >= '2024-04-01 04:00:00'
      AND timestamp < '2024-04-15 04:00:00'
),
-- Forward-fill regimes to all timestamps
regime_filled AS (
    SELECT 
        m.timestamp_est,
        LAST_VALUE(r.regime_state IGNORE NULLS) OVER (
            ORDER BY m.timestamp_est 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as regime_state
    FROM market_data m
    LEFT JOIN regime_sparse r ON m.timestamp_est = r.regime_time_est
),
-- Now join trades with filled regimes
trades_with_regime AS (
    SELECT 
        st.entry_time_est,
        st.entry_signal,
        st.exit_time_est,
        rf.regime_state as entry_regime,
        m1.close as entry_price,
        m2.close as exit_price,
        CASE 
            WHEN st.entry_signal = 1 THEN (m2.close - m1.close) / m1.close
            WHEN st.entry_signal = -1 THEN (m1.close - m2.close) / m1.close
            ELSE 0
        END as trade_return
    FROM signal_trades st
    LEFT JOIN regime_filled rf ON st.entry_time_est = rf.timestamp_est
    LEFT JOIN market_data m1 ON st.entry_time_est = m1.timestamp_est
    LEFT JOIN market_data m2 ON st.exit_time_est = m2.timestamp_est
    WHERE st.prev_signal IS NOT NULL
      AND st.prev_signal != st.entry_signal  -- Only actual trades
      AND st.exit_time_est IS NOT NULL
      AND rf.regime_state IS NOT NULL
      AND m1.close IS NOT NULL
      AND m2.close IS NOT NULL
)
SELECT 
    entry_regime,
    COUNT(*) as trade_count,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(SUM(trade_return) * 100, 3) as total_return_pct,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
FROM trades_with_regime
GROUP BY entry_regime
ORDER BY total_return_pct DESC;