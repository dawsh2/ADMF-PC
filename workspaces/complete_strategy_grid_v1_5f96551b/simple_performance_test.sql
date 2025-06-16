-- Simple test to get strategy performance per regime
-- Direct approach without complex joins

PRAGMA memory_limit='3GB';
SET threads=4;

SELECT 
    '=== SIMPLE PERFORMANCE TEST ===' as header;

-- First, let's just count trades in April 2024
WITH macd_trades AS (
    SELECT 
        s.ts::timestamp as signal_time,
        s.val as signal_value,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal,
        LEAD(s.ts::timestamp) OVER (ORDER BY s.ts::timestamp) as next_signal_time
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-04-01 00:00:00'
      AND s.ts::timestamp < '2024-05-01 00:00:00'
),
trades_only AS (
    SELECT *
    FROM macd_trades
    WHERE prev_signal IS NOT NULL
      AND prev_signal != signal_value
      AND next_signal_time IS NOT NULL
)
SELECT 
    COUNT(*) as total_trades,
    MIN(signal_time) as first_trade,
    MAX(signal_time) as last_trade
FROM trades_only;

-- Now let's try to match with SPY data using simpler approach
SELECT 
    '=== TRADE RETURNS CALCULATION ===' as section_header;

WITH macd_trades AS (
    SELECT 
        s.ts::timestamp as entry_time,
        s.val as entry_signal,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal,
        LEAD(s.ts::timestamp) OVER (ORDER BY s.ts::timestamp) as exit_time
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-04-01 00:00:00'
      AND s.ts::timestamp < '2024-04-15 00:00:00'  -- Two weeks
),
valid_trades AS (
    SELECT *
    FROM macd_trades
    WHERE prev_signal IS NOT NULL
      AND prev_signal != entry_signal
      AND exit_time IS NOT NULL
),
-- Get prices - convert SPY timestamps to match signal timestamps
spy_prices AS (
    SELECT 
        -- Convert to timestamp without timezone to match signals
        timestamp::timestamp as price_time,
        close
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE timestamp >= '2024-04-01 00:00:00-07'
      AND timestamp < '2024-04-15 00:00:00-07'
)
SELECT 
    COUNT(*) as trades_with_prices,
    COUNT(DISTINCT vt.entry_time) as unique_entry_times,
    COUNT(DISTINCT sp1.price_time) as unique_price_times,
    MIN(vt.entry_time) as min_trade_time,
    MAX(vt.entry_time) as max_trade_time,
    MIN(sp1.price_time) as min_price_time,
    MAX(sp1.price_time) as max_price_time
FROM valid_trades vt
LEFT JOIN spy_prices sp1 ON vt.entry_time = sp1.price_time
LEFT JOIN spy_prices sp2 ON vt.exit_time = sp2.price_time;

-- Finally, let's compute returns with regime data
SELECT 
    '=== REGIME-BASED RETURNS ===' as section_header;

WITH 
-- Get regime timeline
regime_data AS (
    SELECT 
        c.ts::timestamp as regime_time,
        c.val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
    WHERE c.ts::timestamp >= '2024-04-01 00:00:00'
      AND c.ts::timestamp < '2024-04-15 00:00:00'
),
-- Get trades
macd_trades AS (
    SELECT 
        s.ts::timestamp as entry_time,
        s.val as entry_signal,
        LAG(s.val) OVER (ORDER BY s.ts::timestamp) as prev_signal,
        LEAD(s.ts::timestamp) OVER (ORDER BY s.ts::timestamp) as exit_time
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet') s
    WHERE s.ts::timestamp >= '2024-04-01 00:00:00'
      AND s.ts::timestamp < '2024-04-15 00:00:00'
),
valid_trades AS (
    SELECT 
        entry_time,
        entry_signal,
        exit_time,
        -- Find nearest regime
        (SELECT r.regime_state 
         FROM regime_data r 
         WHERE r.regime_time <= entry_time
         ORDER BY r.regime_time DESC
         LIMIT 1) as entry_regime
    FROM macd_trades
    WHERE prev_signal IS NOT NULL
      AND prev_signal != entry_signal
      AND exit_time IS NOT NULL
),
-- Get prices with timezone handling
trades_with_prices AS (
    SELECT 
        vt.*,
        sp1.close as entry_price,
        sp2.close as exit_price,
        CASE 
            WHEN vt.entry_signal = 1 THEN (sp2.close - sp1.close) / sp1.close
            WHEN vt.entry_signal = -1 THEN (sp1.close - sp2.close) / sp1.close
        END as trade_return
    FROM valid_trades vt
    LEFT JOIN (
        SELECT timestamp::timestamp + INTERVAL 4 HOUR as price_time, close
        FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
        WHERE timestamp >= '2024-04-01 00:00:00-07'
          AND timestamp < '2024-04-15 00:00:00-07'
    ) sp1 ON vt.entry_time = sp1.price_time
    LEFT JOIN (
        SELECT timestamp::timestamp + INTERVAL 4 HOUR as price_time, close
        FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
        WHERE timestamp >= '2024-04-01 00:00:00-07'
          AND timestamp < '2024-04-15 00:00:00-07'
    ) sp2 ON vt.exit_time = sp2.price_time
)
SELECT 
    entry_regime,
    COUNT(*) as trade_count,
    SUM(CASE WHEN entry_price IS NOT NULL AND exit_price IS NOT NULL THEN 1 ELSE 0 END) as trades_with_prices,
    ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
    ROUND(SUM(trade_return) * 100, 3) as total_return_pct,
    SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / 
        NULLIF(SUM(CASE WHEN trade_return IS NOT NULL THEN 1 ELSE 0 END), 0) as win_rate
FROM trades_with_prices
WHERE entry_regime IS NOT NULL
GROUP BY entry_regime
ORDER BY total_return_pct DESC;