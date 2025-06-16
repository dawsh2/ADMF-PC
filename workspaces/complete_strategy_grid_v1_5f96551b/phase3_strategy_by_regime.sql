-- Phase 3: Analyze strategy performance by regime for top 3 classifiers
-- Focus on the best performing classifiers with 4 unique states

PRAGMA memory_limit='3GB';
SET threads=4;

-- Top 3 classifiers from Phase 2
WITH top_classifiers AS (
    SELECT classifier_id FROM (VALUES 
        ('SPY_volatility_momentum_grid_05_65_40'),
        ('SPY_volatility_momentum_grid_05_55_45'),
        ('SPY_volatility_momentum_grid_05_65_35')
    ) AS t(classifier_id)
),

-- Get classifier data with forward-fill to complete timeline
classifier_sparse AS (
    SELECT 
        c.ts::timestamp as timestamp,
        c.strat as classifier_id,
        c.val as regime_state
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet') c
    INNER JOIN top_classifiers tc ON c.strat = tc.classifier_id
),

-- Forward-fill classifier states over market data timeline
classifier_full_series AS (
    SELECT 
        m.timestamp,
        tc.classifier_id,
        LAST_VALUE(cs.regime_state IGNORE NULLS) OVER (
            PARTITION BY tc.classifier_id 
            ORDER BY m.timestamp 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as current_regime
    FROM analytics.market_data m
    CROSS JOIN top_classifiers tc
    LEFT JOIN classifier_sparse cs ON m.timestamp = cs.timestamp AND tc.classifier_id = cs.classifier_id
    WHERE m.timestamp >= '2024-03-26 13:30:00'  -- Start of data availability
      AND m.timestamp <= '2024-04-02 20:00:00'  -- End of data window
),

-- Sample a subset of high-performing strategies for analysis
strategy_sample AS (
    SELECT DISTINCT strategy_id FROM analytics.strategy_performance 
    WHERE total_trades >= 50  -- Minimum trade frequency
      AND sharpe_ratio > 1.0   -- Decent performance
    ORDER BY sharpe_ratio DESC
    LIMIT 100  -- Top 100 strategies for manageable analysis
),

-- Get strategy trades with regime information
strategy_trades_with_regime AS (
    SELECT 
        st.strategy_id,
        st.entry_time,
        st.exit_time,
        st.side,
        st.entry_price,
        st.exit_price,
        st.pnl,
        st.duration_minutes,
        cfs.classifier_id,
        cfs.current_regime,
        -- Apply -4 hour timezone correction
        (st.entry_time - INTERVAL 4 HOUR) as entry_time_est,
        (st.exit_time - INTERVAL 4 HOUR) as exit_time_est
    FROM analytics.strategy_trades st
    INNER JOIN strategy_sample ss ON st.strategy_id = ss.strategy_id
    INNER JOIN classifier_full_series cfs ON st.entry_time = cfs.timestamp
    WHERE cfs.current_regime IS NOT NULL
      AND st.entry_time >= '2024-03-26 13:30:00'
      AND st.exit_time <= '2024-04-02 20:00:00'
),

-- Calculate performance by regime for each classifier-strategy combination
regime_performance AS (
    SELECT 
        classifier_id,
        current_regime,
        strategy_id,
        COUNT(*) as trade_count,
        SUM(pnl) as total_pnl,
        AVG(pnl) as avg_pnl,
        STDDEV(pnl) as pnl_std,
        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
        AVG(duration_minutes) as avg_duration,
        
        -- Risk-adjusted metrics
        CASE WHEN STDDEV(pnl) > 0 THEN AVG(pnl) / STDDEV(pnl) ELSE 0 END as trade_sharpe,
        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
        
        -- Regime-specific insights
        MIN(entry_time_est) as first_trade_time,
        MAX(exit_time_est) as last_trade_time
        
    FROM strategy_trades_with_regime
    GROUP BY classifier_id, current_regime, strategy_id
    HAVING COUNT(*) >= 5  -- Minimum 5 trades per regime
)

SELECT 
    '=== PHASE 3: STRATEGY PERFORMANCE BY REGIME ===' as header;

-- Overall regime performance summary
SELECT 
    '=== REGIME PERFORMANCE SUMMARY ===' as section_header;

SELECT 
    classifier_id,
    current_regime,
    COUNT(DISTINCT strategy_id) as strategies_tested,
    SUM(trade_count) as total_trades,
    ROUND(AVG(total_pnl), 4) as avg_strategy_pnl,
    ROUND(AVG(trade_sharpe), 3) as avg_trade_sharpe,
    ROUND(AVG(win_rate), 1) as avg_win_rate,
    ROUND(AVG(avg_duration), 1) as avg_trade_duration,
    
    -- Count of profitable strategies per regime
    SUM(CASE WHEN total_pnl > 0 THEN 1 ELSE 0 END) as profitable_strategies,
    SUM(CASE WHEN total_pnl <= 0 THEN 1 ELSE 0 END) as unprofitable_strategies
    
FROM regime_performance
GROUP BY classifier_id, current_regime
ORDER BY classifier_id, avg_trade_sharpe DESC;

-- Top performing strategies by regime
SELECT 
    '=== TOP 5 STRATEGIES PER REGIME ===' as section_header;

WITH ranked_strategies AS (
    SELECT 
        classifier_id,
        current_regime,
        strategy_id,
        trade_count,
        total_pnl,
        trade_sharpe,
        win_rate,
        avg_duration,
        ROW_NUMBER() OVER (PARTITION BY classifier_id, current_regime ORDER BY trade_sharpe DESC) as rank
    FROM regime_performance
    WHERE trade_count >= 10  -- Higher minimum for top strategies
)
SELECT 
    classifier_id,
    current_regime,
    rank,
    strategy_id,
    trade_count,
    ROUND(total_pnl, 4) as total_pnl,
    ROUND(trade_sharpe, 3) as trade_sharpe,
    ROUND(win_rate, 1) as win_rate,
    ROUND(avg_duration, 1) as avg_duration
FROM ranked_strategies
WHERE rank <= 5
ORDER BY classifier_id, current_regime, rank;

-- Regime comparison for individual strategies
SELECT 
    '=== REGIME COMPARISON FOR VERSATILE STRATEGIES ===' as section_header;

WITH strategy_regime_counts AS (
    SELECT 
        strategy_id,
        COUNT(DISTINCT current_regime) as regimes_traded,
        COUNT(DISTINCT classifier_id) as classifiers_tested
    FROM regime_performance
    GROUP BY strategy_id
    HAVING COUNT(DISTINCT current_regime) >= 3  -- Strategies that work in multiple regimes
)
SELECT 
    rp.strategy_id,
    rp.classifier_id,
    rp.current_regime,
    rp.trade_count,
    ROUND(rp.total_pnl, 4) as total_pnl,
    ROUND(rp.trade_sharpe, 3) as trade_sharpe,
    ROUND(rp.win_rate, 1) as win_rate,
    src.regimes_traded
FROM regime_performance rp
INNER JOIN strategy_regime_counts src ON rp.strategy_id = src.strategy_id
WHERE src.regimes_traded >= 3
ORDER BY rp.strategy_id, rp.classifier_id, rp.trade_sharpe DESC;

-- Time-based regime analysis
SELECT 
    '=== REGIME TIMING ANALYSIS ===' as section_header;

SELECT 
    classifier_id,
    current_regime,
    COUNT(*) as total_regime_trades,
    ROUND(AVG(EXTRACT(HOUR FROM first_trade_time)), 1) as avg_start_hour,
    ROUND(AVG(EXTRACT(HOUR FROM last_trade_time)), 1) as avg_end_hour,
    
    -- Market session distribution
    SUM(CASE WHEN EXTRACT(HOUR FROM first_trade_time) BETWEEN 9 AND 11 THEN 1 ELSE 0 END) as morning_trades,
    SUM(CASE WHEN EXTRACT(HOUR FROM first_trade_time) BETWEEN 12 AND 14 THEN 1 ELSE 0 END) as midday_trades,
    SUM(CASE WHEN EXTRACT(HOUR FROM first_trade_time) BETWEEN 15 AND 16 THEN 1 ELSE 0 END) as afternoon_trades
    
FROM regime_performance
GROUP BY classifier_id, current_regime
ORDER BY classifier_id, total_regime_trades DESC;