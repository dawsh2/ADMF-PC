-- Phase 3: Simple strategy performance by regime analysis
-- Step-by-step approach to avoid CTE complexity

PRAGMA memory_limit='3GB';
SET threads=4;

SELECT 
    '=== PHASE 3: STRATEGY PERFORMANCE BY REGIME ===' as header;

-- First, let's check what regime states we actually have
SELECT 
    '=== AVAILABLE REGIME STATES ===' as section_header;

SELECT 
    c.strat as classifier_id,
    c.val as regime_state,
    COUNT(*) as state_changes,
    MIN(c.ts::timestamp) as first_occurrence,
    MAX(c.ts::timestamp) as last_occurrence
FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet') c
WHERE c.strat IN (
    'SPY_volatility_momentum_grid_05_65_40',
    'SPY_volatility_momentum_grid_05_55_45', 
    'SPY_volatility_momentum_grid_05_65_35'
)
GROUP BY c.strat, c.val
ORDER BY c.strat, COUNT(*) DESC;

-- Check if we have strategy performance data and what timeframe it covers
SELECT 
    '=== STRATEGY PERFORMANCE DATA OVERVIEW ===' as section_header;

SELECT 
    COUNT(DISTINCT strategy_id) as total_strategies,
    COUNT(*) as total_trades,
    MIN(entry_time) as earliest_trade,
    MAX(exit_time) as latest_trade,
    ROUND(AVG(sharpe_ratio), 3) as avg_sharpe,
    COUNT(CASE WHEN sharpe_ratio > 1.0 THEN 1 END) as strategies_sharpe_above_1
FROM analytics.strategy_performance
WHERE total_trades >= 10;

-- Get a sample of high-performing strategies for regime analysis
SELECT 
    '=== TOP 20 STRATEGIES FOR REGIME ANALYSIS ===' as section_header;

SELECT 
    strategy_id,
    total_trades,
    ROUND(sharpe_ratio, 3) as sharpe_ratio,
    ROUND(total_return, 4) as total_return,
    ROUND(win_rate, 1) as win_rate,
    ROUND(avg_trade_duration, 1) as avg_duration_min
FROM analytics.strategy_performance
WHERE total_trades >= 50  -- Minimum frequency
  AND sharpe_ratio > 1.0   -- Decent performance
ORDER BY sharpe_ratio DESC
LIMIT 20;

-- Sample regime-aware trade analysis for one top strategy
SELECT 
    '=== SAMPLE REGIME ANALYSIS: TOP STRATEGY ===' as section_header;

WITH top_strategy AS (
    SELECT strategy_id 
    FROM analytics.strategy_performance
    WHERE total_trades >= 50 AND sharpe_ratio > 1.0
    ORDER BY sharpe_ratio DESC
    LIMIT 1
),
strategy_trades_sample AS (
    SELECT 
        st.strategy_id,
        st.entry_time,
        st.exit_time,
        st.pnl,
        st.side,
        st.duration_minutes,
        -- Apply timezone correction
        (st.entry_time - INTERVAL 4 HOUR) as entry_time_est
    FROM analytics.strategy_trades st
    INNER JOIN top_strategy ts ON st.strategy_id = ts.strategy_id
    WHERE st.entry_time >= '2024-03-26 13:30:00'
      AND st.exit_time <= '2024-04-02 20:00:00'
    LIMIT 100  -- Sample for testing
),
classifier_at_trade_time AS (
    SELECT 
        sts.*,
        c.strat as classifier_id,
        c.val as regime_state
    FROM strategy_trades_sample sts
    LEFT JOIN read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet') c
        ON ABS(EXTRACT(EPOCH FROM (sts.entry_time - c.ts::timestamp))) <= 300  -- Within 5 minutes
    WHERE c.val IS NOT NULL
)
SELECT 
    strategy_id,
    classifier_id,
    regime_state,
    COUNT(*) as trades_in_regime,
    ROUND(SUM(pnl), 4) as total_pnl,
    ROUND(AVG(pnl), 4) as avg_pnl,
    ROUND(AVG(duration_minutes), 1) as avg_duration,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
FROM classifier_at_trade_time
WHERE regime_state IS NOT NULL
GROUP BY strategy_id, classifier_id, regime_state
ORDER BY total_pnl DESC;

-- Quick regime distribution check for our timeframe
SELECT 
    '=== REGIME DISTRIBUTION IN TRADING PERIOD ===' as section_header;

SELECT 
    c.strat as classifier_id,
    c.val as regime_state,
    COUNT(*) as minutes_in_regime,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY c.strat), 1) as regime_percentage
FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet') c
WHERE c.strat IN (
    'SPY_volatility_momentum_grid_05_65_40',
    'SPY_volatility_momentum_grid_05_55_45', 
    'SPY_volatility_momentum_grid_05_65_35'
)
  AND c.ts::timestamp >= '2024-03-26 13:30:00'
  AND c.ts::timestamp <= '2024-04-02 20:00:00'
GROUP BY c.strat, c.val
ORDER BY c.strat, COUNT(*) DESC;