-- Strategy Performance Per Regime Analysis
-- Build performance metrics from raw signals and correlate with regime states

PRAGMA memory_limit='3GB';
SET threads=4;

SELECT 
    '=== STRATEGY PERFORMANCE PER REGIME ANALYSIS ===' as header;

-- First, let's get market data for returns calculation
SELECT 
    '=== MARKET DATA AVAILABILITY ===' as section_header;

SELECT 
    COUNT(*) as total_bars,
    MIN(timestamp) as earliest_bar,
    MAX(timestamp) as latest_bar,
    COUNT(DISTINCT DATE(timestamp)) as trading_days
FROM analytics.market_data
WHERE timestamp >= '2024-03-26 13:30:00'
  AND timestamp <= '2024-04-02 20:00:00';

-- Sample one high-frequency strategy for detailed regime analysis
SELECT 
    '=== SAMPLE STRATEGY: MACD CROSSOVER ===' as section_header;

WITH sample_strategy AS (
    SELECT 
        signal_file_path, 
        strategy_id, 
        strategy_name,
        strategy_type
    FROM strategies 
    WHERE strategy_type = 'macd_crossover'
      AND strategy_name = 'macd_crossover_grid_12_26_9'
    LIMIT 1
),
-- Get strategy signals
strategy_signals AS (
    SELECT 
        s.ts::timestamp as signal_time,
        s.val as signal_value,
        ss.strategy_id,
        ss.strategy_name,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal
    FROM sample_strategy ss,
         read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-03-26 13:30:00'
      AND s.ts::timestamp <= '2024-04-02 20:00:00'
),
-- Identify trade entries and exits
trade_signals AS (
    SELECT 
        *,
        CASE 
            WHEN prev_signal = -1 AND signal_value = 1 THEN 'LONG_ENTRY'
            WHEN prev_signal = 1 AND signal_value = -1 THEN 'SHORT_ENTRY'
            WHEN prev_signal = 1 AND signal_value = -1 THEN 'LONG_EXIT'
            WHEN prev_signal = -1 AND signal_value = 1 THEN 'SHORT_EXIT'
            ELSE 'HOLD'
        END as trade_action,
        CASE 
            WHEN prev_signal != signal_value THEN 1 ELSE 0
        END as is_signal_change
    FROM strategy_signals
    WHERE prev_signal IS NOT NULL
),
-- Get regime state at each signal
signals_with_regime AS (
    SELECT 
        ts.*,
        c.val as regime_state
    FROM trade_signals ts
    LEFT JOIN read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
        ON ts.signal_time = c.ts::timestamp
),
-- Construct trades from signal changes
trades_constructed AS (
    SELECT 
        strategy_id,
        strategy_name,
        signal_time as entry_time,
        signal_value as entry_signal,
        regime_state as entry_regime,
        LEAD(signal_time) OVER (ORDER BY signal_time) as exit_time,
        LEAD(signal_value) OVER (ORDER BY signal_time) as exit_signal,
        LEAD(regime_state) OVER (ORDER BY signal_time) as exit_regime,
        EXTRACT(EPOCH FROM (LEAD(signal_time) OVER (ORDER BY signal_time) - signal_time)) / 60.0 as duration_minutes
    FROM signals_with_regime
    WHERE is_signal_change = 1
      AND regime_state IS NOT NULL
),
-- Get market prices for PnL calculation
trades_with_prices AS (
    SELECT 
        tc.*,
        m_entry.close as entry_price,
        m_exit.close as exit_price,
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
    LEFT JOIN analytics.market_data m_entry ON tc.entry_time = m_entry.timestamp
    LEFT JOIN analytics.market_data m_exit ON tc.exit_time = m_exit.timestamp
    WHERE tc.exit_time IS NOT NULL
      AND tc.entry_signal IN (-1, 1)
)
SELECT 
    strategy_id,
    strategy_name,
    entry_regime,
    trade_side,
    COUNT(*) as trade_count,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(STDDEV(trade_return) * 100, 3) as return_std_pct,
    ROUND(AVG(trade_return) / NULLIF(STDDEV(trade_return), 0), 3) as trade_sharpe,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    ROUND(AVG(duration_minutes), 1) as avg_duration_min,
    ROUND(SUM(trade_return) * 100, 3) as total_return_pct,
    MIN(entry_time) as first_trade,
    MAX(exit_time) as last_trade
FROM trades_with_prices
WHERE entry_price IS NOT NULL 
  AND exit_price IS NOT NULL
GROUP BY strategy_id, strategy_name, entry_regime, trade_side
ORDER BY entry_regime, trade_side, avg_return_pct DESC;

-- Summary across all regimes for this strategy
SELECT 
    '=== REGIME PERFORMANCE SUMMARY: MACD STRATEGY ===' as section_header;

WITH sample_strategy AS (
    SELECT 
        signal_file_path, 
        strategy_id, 
        strategy_name,
        strategy_type
    FROM strategies 
    WHERE strategy_type = 'macd_crossover'
      AND strategy_name = 'macd_crossover_grid_12_26_9'
    LIMIT 1
),
strategy_signals AS (
    SELECT 
        s.ts::timestamp as signal_time,
        s.val as signal_value,
        ss.strategy_id,
        ss.strategy_name,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal
    FROM sample_strategy ss,
         read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-03-26 13:30:00'
      AND s.ts::timestamp <= '2024-04-02 20:00:00'
),
signals_with_regime AS (
    SELECT 
        ss.*,
        c.val as regime_state,
        CASE WHEN prev_signal != signal_value THEN 1 ELSE 0 END as is_signal_change
    FROM strategy_signals ss
    LEFT JOIN read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
        ON ss.signal_time = c.ts::timestamp
    WHERE prev_signal IS NOT NULL
),
trades_constructed AS (
    SELECT 
        strategy_id,
        strategy_name,
        signal_time as entry_time,
        signal_value as entry_signal,
        regime_state as entry_regime,
        LEAD(signal_time) OVER (ORDER BY signal_time) as exit_time,
        LEAD(signal_value) OVER (ORDER BY signal_time) as exit_signal,
        EXTRACT(EPOCH FROM (LEAD(signal_time) OVER (ORDER BY signal_time) - signal_time)) / 60.0 as duration_minutes
    FROM signals_with_regime
    WHERE is_signal_change = 1
      AND regime_state IS NOT NULL
),
trades_with_prices AS (
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
    LEFT JOIN analytics.market_data m_entry ON tc.entry_time = m_entry.timestamp
    LEFT JOIN analytics.market_data m_exit ON tc.exit_time = m_exit.timestamp
    WHERE tc.exit_time IS NOT NULL
      AND tc.entry_signal IN (-1, 1)
)
SELECT 
    entry_regime,
    COUNT(*) as total_trades,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(STDDEV(trade_return) * 100, 3) as return_volatility_pct,
    ROUND(SUM(trade_return) * 100, 3) as cumulative_return_pct,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    ROUND(AVG(duration_minutes), 1) as avg_duration_min,
    MAX(trade_return) * 100 as best_trade_pct,
    MIN(trade_return) * 100 as worst_trade_pct
FROM trades_with_prices
WHERE entry_price IS NOT NULL 
  AND exit_price IS NOT NULL
  AND entry_regime IS NOT NULL
GROUP BY entry_regime
ORDER BY cumulative_return_pct DESC;