-- Quick State Distribution Filter
-- Identify useless classifiers before expensive analysis
-- Filter criteria: balanced states, sufficient data, meaningful differentiation

PRAGMA memory_limit='3GB';
SET threads=4;

-- Analyze all classifiers for state distribution quality
WITH all_classifier_states AS (
    -- Hidden Markov classifiers
    SELECT 
        'hidden_markov' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as state_occurrences
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/hidden_markov_grid/*.parquet')
    GROUP BY strat, val
    
    UNION ALL
    
    -- Market Regime classifiers
    SELECT 
        'market_regime' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as state_occurrences
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/market_regime_grid/*.parquet')
    GROUP BY strat, val
    
    UNION ALL
    
    -- Microstructure classifiers
    SELECT 
        'microstructure' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as state_occurrences
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/microstructure_grid/*.parquet')
    GROUP BY strat, val
    
    UNION ALL
    
    -- Multi-timeframe trend classifiers
    SELECT 
        'multi_timeframe_trend' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as state_occurrences
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/multi_timeframe_trend_grid/*.parquet')
    GROUP BY strat, val
    
    UNION ALL
    
    -- Volatility momentum classifiers
    SELECT 
        'volatility_momentum' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as state_occurrences
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet')
    GROUP BY strat, val
),
classifier_distribution_stats AS (
    SELECT 
        classifier_type,
        classifier_id,
        COUNT(DISTINCT regime_state) as num_states,
        SUM(state_occurrences) as total_signals,
        -- Calculate state balance metrics
        MAX(state_occurrences * 100.0 / SUM(state_occurrences) OVER (PARTITION BY classifier_type, classifier_id)) as max_state_pct,
        MIN(state_occurrences * 100.0 / SUM(state_occurrences) OVER (PARTITION BY classifier_type, classifier_id)) as min_state_pct,
        STDDEV(state_occurrences * 100.0 / SUM(state_occurrences) OVER (PARTITION BY classifier_type, classifier_id)) as state_balance_std
    FROM all_classifier_states
    GROUP BY classifier_type, classifier_id
),
classifier_quality_scores AS (
    SELECT 
        classifier_type,
        classifier_id,
        num_states,
        total_signals,
        ROUND(max_state_pct, 1) as max_state_pct,
        ROUND(min_state_pct, 1) as min_state_pct,
        ROUND(state_balance_std, 1) as balance_std,
        -- Quality scoring
        CASE 
            WHEN max_state_pct > 80 THEN 'DOMINATED (Useless)'
            WHEN max_state_pct > 60 THEN 'UNBALANCED (Poor)'
            WHEN min_state_pct < 5 THEN 'RARE_STATES (Poor)'
            WHEN state_balance_std < 5 THEN 'EXCELLENT'
            WHEN state_balance_std < 10 THEN 'GOOD'
            WHEN state_balance_std < 20 THEN 'FAIR'
            ELSE 'POOR'
        END as balance_quality,
        CASE 
            WHEN total_signals < 1000 THEN 'INSUFFICIENT_DATA'
            WHEN total_signals < 5000 THEN 'LIMITED_DATA'
            ELSE 'SUFFICIENT_DATA'
        END as data_adequacy
    FROM classifier_distribution_stats
)
SELECT 
    '=== CLASSIFIER STATE DISTRIBUTION QUALITY ===' as header;

-- Summary by classifier type
SELECT 
    classifier_type,
    COUNT(*) as total_classifiers,
    COUNT(CASE WHEN balance_quality = 'EXCELLENT' THEN 1 END) as excellent_count,
    COUNT(CASE WHEN balance_quality = 'GOOD' THEN 1 END) as good_count,
    COUNT(CASE WHEN balance_quality = 'FAIR' THEN 1 END) as fair_count,
    COUNT(CASE WHEN balance_quality IN ('DOMINATED (Useless)', 'UNBALANCED (Poor)', 'RARE_STATES (Poor)', 'POOR') THEN 1 END) as poor_count,
    ROUND(AVG(balance_std), 1) as avg_balance_std,
    ROUND(AVG(max_state_pct), 1) as avg_max_state_pct
FROM classifier_quality_scores
GROUP BY classifier_type
ORDER BY excellent_count + good_count DESC;

-- Show top candidates for further analysis
SELECT 
    '=== TOP CLASSIFIER CANDIDATES (Balanced States) ===' as header;

SELECT 
    classifier_type,
    classifier_id,
    num_states,
    total_signals,
    max_state_pct,
    min_state_pct,
    balance_std,
    balance_quality,
    data_adequacy
FROM classifier_quality_scores
WHERE balance_quality IN ('EXCELLENT', 'GOOD', 'FAIR')
  AND data_adequacy = 'SUFFICIENT_DATA'
ORDER BY 
    CASE balance_quality 
        WHEN 'EXCELLENT' THEN 1 
        WHEN 'GOOD' THEN 2 
        ELSE 3 
    END,
    balance_std ASC;

-- Show worst classifiers to avoid
SELECT 
    '=== USELESS CLASSIFIERS TO EXCLUDE ===' as header;

SELECT 
    classifier_type,
    classifier_id,
    num_states,
    total_signals,
    max_state_pct,
    min_state_pct,
    balance_quality,
    data_adequacy,
    CASE 
        WHEN max_state_pct > 80 THEN 'Single state dominates >80%'
        WHEN min_state_pct < 5 THEN 'Has rare states <5%'
        WHEN total_signals < 1000 THEN 'Insufficient data'
        ELSE 'Poor balance overall'
    END as exclusion_reason
FROM classifier_quality_scores
WHERE balance_quality IN ('DOMINATED (Useless)', 'UNBALANCED (Poor)', 'RARE_STATES (Poor)', 'POOR')
   OR data_adequacy = 'INSUFFICIENT_DATA'
ORDER BY max_state_pct DESC, total_signals ASC;

-- Detailed state breakdown for top candidates
WITH top_candidates AS (
    SELECT classifier_type, classifier_id
    FROM classifier_quality_scores
    WHERE balance_quality IN ('EXCELLENT', 'GOOD')
      AND data_adequacy = 'SUFFICIENT_DATA'
    ORDER BY balance_std ASC
    LIMIT 10
),
detailed_states AS (
    SELECT 
        a.classifier_type,
        a.classifier_id,
        a.regime_state,
        a.state_occurrences,
        ROUND(a.state_occurrences * 100.0 / SUM(a.state_occurrences) OVER (PARTITION BY a.classifier_type, a.classifier_id), 1) as state_pct
    FROM all_classifier_states a
    JOIN top_candidates t ON a.classifier_type = t.classifier_type AND a.classifier_id = t.classifier_id
)
SELECT 
    '=== DETAILED STATE BREAKDOWN (Top 10 Candidates) ===' as header;

SELECT 
    classifier_type,
    classifier_id,
    regime_state,
    state_occurrences,
    state_pct
FROM detailed_states
ORDER BY classifier_type, classifier_id, state_pct DESC;