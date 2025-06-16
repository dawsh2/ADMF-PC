-- Phase 1.3: Market Differentiation Test
-- Test if ~10 minute regime changes predict different market conditions
-- This is the KEY test - do regimes actually matter for trading?

PRAGMA memory_limit='3GB';
SET threads=4;

-- Test multiple classifier types to see which best differentiate market conditions
WITH classifier_full_series AS (
    -- Create full time series for multiple classifiers
    -- Hidden Markov
    SELECT 
        'hidden_markov' as classifier_type,
        'SPY_hidden_markov_grid_11_0001_05' as classifier_id,
        m.timestamp,
        COALESCE(c.val, LAG(c.val) IGNORE NULLS OVER (ORDER BY m.timestamp)) as regime_state,
        m.close as price,
        m.volume
    FROM analytics.market_data m
    LEFT JOIN (
        SELECT ts::timestamp as timestamp, val
        FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet')
        WHERE strat = 'SPY_hidden_markov_grid_11_0001_05'
    ) c ON m.timestamp = c.timestamp
    WHERE m.timestamp >= '2024-03-26' AND m.timestamp <= '2025-01-17'
      AND EXTRACT(hour FROM m.timestamp) BETWEEN 9 AND 15  -- Market hours only
    
    UNION ALL
    
    -- Market Regime
    SELECT 
        'market_regime' as classifier_type,
        'SPY_market_regime_grid_00015_03' as classifier_id,
        m.timestamp,
        COALESCE(c.val, LAG(c.val) IGNORE NULLS OVER (ORDER BY m.timestamp)) as regime_state,
        m.close as price,
        m.volume
    FROM analytics.market_data m
    LEFT JOIN (
        SELECT ts::timestamp as timestamp, val
        FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/market_regime_grid/SPY_market_regime_grid_00015_03.parquet')
        WHERE strat = 'SPY_market_regime_grid_00015_03'
    ) c ON m.timestamp = c.timestamp
    WHERE m.timestamp >= '2024-03-26' AND m.timestamp <= '2025-01-17'
      AND EXTRACT(hour FROM m.timestamp) BETWEEN 9 AND 15
    
    UNION ALL
    
    -- Volatility Momentum
    SELECT 
        'volatility_momentum' as classifier_type,
        'SPY_volatility_momentum_grid_05_55_35' as classifier_id,
        m.timestamp,
        COALESCE(c.val, LAG(c.val) IGNORE NULLS OVER (ORDER BY m.timestamp)) as regime_state,
        m.close as price,
        m.volume
    FROM analytics.market_data m
    LEFT JOIN (
        SELECT ts::timestamp as timestamp, val
        FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_55_35.parquet')
        WHERE strat = 'SPY_volatility_momentum_grid_05_55_35'
    ) c ON m.timestamp = c.timestamp
    WHERE m.timestamp >= '2024-03-26' AND m.timestamp <= '2025-01-17'
      AND EXTRACT(hour FROM m.timestamp) BETWEEN 9 AND 15
),
market_metrics AS (
    SELECT 
        classifier_type,
        classifier_id,
        regime_state,
        timestamp,
        price,
        volume,
        -- Forward-looking returns for predictive testing
        LEAD(price, 5) OVER (PARTITION BY classifier_type ORDER BY timestamp) / price - 1 as next_5min_return,
        LEAD(price, 10) OVER (PARTITION BY classifier_type ORDER BY timestamp) / price - 1 as next_10min_return,
        LEAD(price, 15) OVER (PARTITION BY classifier_type ORDER BY timestamp) / price - 1 as next_15min_return,
        -- Volatility measures
        (LAG(price, 1) OVER (PARTITION BY classifier_type ORDER BY timestamp) - LAG(price, 6) OVER (PARTITION BY classifier_type ORDER BY timestamp)) / LAG(price, 6) OVER (PARTITION BY classifier_type ORDER BY timestamp) as momentum_5min,
        -- Volume measures
        volume / NULLIF(AVG(volume) OVER (PARTITION BY classifier_type ORDER BY timestamp ROWS 20 PRECEDING), 0) as volume_ratio
    FROM classifier_full_series
    WHERE regime_state IS NOT NULL
)
SELECT 
    '=== MARKET DIFFERENTIATION BY REGIME ===' as header;

-- Test if regimes predict different forward returns
SELECT 
    classifier_type,
    regime_state,
    COUNT(*) as sample_size,
    -- Forward return characteristics by regime (the key test!)
    ROUND(AVG(next_5min_return) * 10000, 2) as avg_5min_return_bps,
    ROUND(STDDEV(next_5min_return) * 10000, 2) as vol_5min_return_bps,
    ROUND(AVG(next_10min_return) * 10000, 2) as avg_10min_return_bps,
    ROUND(STDDEV(next_10min_return) * 10000, 2) as vol_10min_return_bps,
    ROUND(AVG(next_15min_return) * 10000, 2) as avg_15min_return_bps,
    ROUND(STDDEV(next_15min_return) * 10000, 2) as vol_15min_return_bps,
    -- Directional bias detection
    ROUND(COUNT(CASE WHEN next_5min_return > 0 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_positive_5min,
    ROUND(COUNT(CASE WHEN next_10min_return > 0 THEN 1 END) * 100.0 / COUNT(*), 1) as pct_positive_10min,
    -- Volume characteristics
    ROUND(AVG(volume_ratio), 2) as avg_volume_ratio,
    ROUND(AVG(momentum_5min) * 10000, 2) as avg_momentum_bps
FROM market_metrics
WHERE next_5min_return IS NOT NULL
  AND next_10min_return IS NOT NULL
  AND next_15min_return IS NOT NULL
GROUP BY classifier_type, regime_state
HAVING COUNT(*) > 100  -- Sufficient sample size
ORDER BY classifier_type, avg_5min_return_bps DESC;

-- Calculate regime differentiation scores
WITH regime_stats AS (
    SELECT 
        classifier_type,
        regime_state,
        AVG(next_5min_return) * 10000 as avg_5min_return_bps,
        STDDEV(next_5min_return) * 10000 as vol_5min_return_bps,
        AVG(volume_ratio) as avg_volume_ratio,
        COUNT(*) as sample_size
    FROM market_metrics
    WHERE next_5min_return IS NOT NULL
    GROUP BY classifier_type, regime_state
    HAVING COUNT(*) > 100
),
differentiation_scores AS (
    SELECT 
        classifier_type,
        -- Measure how different regimes are from each other
        ROUND(STDDEV(avg_5min_return_bps), 2) as return_differentiation,
        ROUND(STDDEV(vol_5min_return_bps), 2) as vol_differentiation,
        ROUND(STDDEV(avg_volume_ratio), 2) as volume_differentiation,
        COUNT(DISTINCT regime_state) as num_regimes,
        ROUND(MAX(avg_5min_return_bps) - MIN(avg_5min_return_bps), 2) as return_spread_bps
    FROM regime_stats
    GROUP BY classifier_type
)
SELECT 
    '=== CLASSIFIER DIFFERENTIATION POWER ===' as header;

SELECT 
    classifier_type,
    num_regimes,
    return_differentiation,
    vol_differentiation,
    volume_differentiation,
    return_spread_bps,
    -- Combined differentiation score
    ROUND((return_differentiation + vol_differentiation + volume_differentiation) / 3, 2) as overall_differentiation,
    CASE 
        WHEN return_spread_bps > 10 THEN 'HIGH_SIGNAL (>10 bps spread)'
        WHEN return_spread_bps > 5 THEN 'MODERATE_SIGNAL (5-10 bps spread)'
        WHEN return_spread_bps > 2 THEN 'WEAK_SIGNAL (2-5 bps spread)'
        ELSE 'NO_SIGNAL (<2 bps spread)'
    END as signal_quality
FROM differentiation_scores
ORDER BY return_spread_bps DESC;

-- Test session-specific behavior (crucial for intraday regimes)
WITH session_analysis AS (
    SELECT 
        classifier_type,
        regime_state,
        CASE 
            WHEN EXTRACT(hour FROM timestamp) = 9 OR (EXTRACT(hour FROM timestamp) = 10 AND EXTRACT(minute FROM timestamp) < 30) THEN 'OPEN'
            WHEN EXTRACT(hour FROM timestamp) BETWEEN 10 AND 12 THEN 'MORNING'
            WHEN EXTRACT(hour FROM timestamp) BETWEEN 12 AND 14 THEN 'MIDDAY'
            WHEN EXTRACT(hour FROM timestamp) >= 14 THEN 'CLOSE'
        END as session_period,
        next_5min_return,
        volume_ratio
    FROM market_metrics
    WHERE next_5min_return IS NOT NULL
)
SELECT 
    '=== REGIME BEHAVIOR BY SESSION ===' as header;

SELECT 
    classifier_type,
    session_period,
    regime_state,
    COUNT(*) as observations,
    ROUND(AVG(next_5min_return) * 10000, 2) as avg_return_bps,
    ROUND(STDDEV(next_5min_return) * 10000, 2) as vol_bps,
    ROUND(AVG(volume_ratio), 2) as avg_vol_ratio,
    ROUND(COUNT(CASE WHEN next_5min_return > 0 THEN 1 END) * 100.0 / COUNT(*), 1) as win_rate_pct
FROM session_analysis
GROUP BY classifier_type, session_period, regime_state
HAVING COUNT(*) > 50
ORDER BY classifier_type, session_period, avg_return_bps DESC;

-- Critical test: Do regime transitions predict immediate market moves?
WITH regime_transitions AS (
    SELECT 
        classifier_type,
        timestamp,
        regime_state,
        LAG(regime_state) OVER (PARTITION BY classifier_type ORDER BY timestamp) as prev_regime,
        price,
        LEAD(price, 3) OVER (PARTITION BY classifier_type ORDER BY timestamp) / price - 1 as next_3min_return,
        LEAD(price, 5) OVER (PARTITION BY classifier_type ORDER BY timestamp) / price - 1 as next_5min_return
    FROM classifier_full_series
    WHERE regime_state IS NOT NULL
),
transition_analysis AS (
    SELECT 
        classifier_type,
        prev_regime || ' → ' || regime_state as transition_type,
        COUNT(*) as num_transitions,
        ROUND(AVG(next_3min_return) * 10000, 2) as avg_3min_return_bps,
        ROUND(AVG(next_5min_return) * 10000, 2) as avg_5min_return_bps,
        ROUND(STDDEV(next_3min_return) * 10000, 2) as vol_3min_bps,
        ROUND(COUNT(CASE WHEN next_3min_return > 0 THEN 1 END) * 100.0 / COUNT(*), 1) as win_rate_3min
    FROM regime_transitions
    WHERE prev_regime IS NOT NULL 
      AND prev_regime != regime_state  -- Only actual transitions
      AND next_3min_return IS NOT NULL
    GROUP BY classifier_type, transition_type
    HAVING COUNT(*) > 20  -- Sufficient transitions
)
SELECT 
    '=== REGIME TRANSITION PREDICTIVE POWER ===' as header;

SELECT 
    classifier_type,
    transition_type,
    num_transitions,
    avg_3min_return_bps,
    avg_5min_return_bps,
    vol_3min_bps,
    win_rate_3min,
    CASE 
        WHEN ABS(avg_3min_return_bps) > 5 THEN 'STRONG_SIGNAL'
        WHEN ABS(avg_3min_return_bps) > 2 THEN 'MODERATE_SIGNAL'
        ELSE 'WEAK_SIGNAL'
    END as transition_signal_strength
FROM transition_analysis
ORDER BY classifier_type, ABS(avg_3min_return_bps) DESC;