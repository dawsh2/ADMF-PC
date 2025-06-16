-- State Distribution Analysis (Simple Version)
-- Identify useless classifiers before expensive analysis

PRAGMA memory_limit='3GB';
SET threads=4;

-- First, get the raw state data
WITH all_classifier_states AS (
    SELECT 
        'hidden_markov' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as state_occurrences
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/hidden_markov_grid/*.parquet')
    GROUP BY strat, val
    
    UNION ALL
    
    SELECT 
        'market_regime' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as state_occurrences
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/market_regime_grid/*.parquet')
    GROUP BY strat, val
    
    UNION ALL
    
    SELECT 
        'microstructure' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as state_occurrences
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/microstructure_grid/*.parquet')
    GROUP BY strat, val
    
    UNION ALL
    
    SELECT 
        'multi_timeframe_trend' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as state_occurrences
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/multi_timeframe_trend_grid/*.parquet')
    GROUP BY strat, val
    
    UNION ALL
    
    SELECT 
        'volatility_momentum' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as state_occurrences
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet')
    GROUP BY strat, val
),
-- Calculate state percentages and stats
state_percentages AS (
    SELECT 
        classifier_type,
        classifier_id,
        regime_state,
        state_occurrences,
        state_occurrences * 100.0 / SUM(state_occurrences) OVER (PARTITION BY classifier_type, classifier_id) as state_pct
    FROM all_classifier_states
),
-- Aggregate classifier stats
classifier_stats AS (
    SELECT 
        classifier_type,
        classifier_id,
        COUNT(DISTINCT regime_state) as num_states,
        SUM(state_occurrences) as total_signals,
        MAX(state_pct) as max_state_pct,
        MIN(state_pct) as min_state_pct,
        ROUND(STDDEV(state_pct), 1) as balance_std
    FROM state_percentages
    GROUP BY classifier_type, classifier_id
),
-- Quality assessment
classifier_quality AS (
    SELECT 
        classifier_type,
        classifier_id,
        num_states,
        total_signals,
        ROUND(max_state_pct, 1) as max_state_pct,
        ROUND(min_state_pct, 1) as min_state_pct,
        balance_std,
        CASE 
            WHEN max_state_pct > 80 THEN 'DOMINATED (Useless)'
            WHEN max_state_pct > 60 THEN 'UNBALANCED (Poor)'
            WHEN min_state_pct < 5 THEN 'RARE_STATES (Poor)'
            WHEN balance_std < 5 THEN 'EXCELLENT'
            WHEN balance_std < 10 THEN 'GOOD'
            WHEN balance_std < 20 THEN 'FAIR'
            ELSE 'POOR'
        END as balance_quality,
        CASE 
            WHEN total_signals < 1000 THEN 'INSUFFICIENT_DATA'
            WHEN total_signals < 5000 THEN 'LIMITED_DATA'
            ELSE 'SUFFICIENT_DATA'
        END as data_adequacy
    FROM classifier_stats
)

-- Report 1: Summary by classifier type
SELECT 
    '*** CLASSIFIER STATE DISTRIBUTION QUALITY ***' as report_section,
    classifier_type,
    COUNT(*) as total_classifiers,
    COUNT(CASE WHEN balance_quality = 'EXCELLENT' THEN 1 END) as excellent_count,
    COUNT(CASE WHEN balance_quality = 'GOOD' THEN 1 END) as good_count,
    COUNT(CASE WHEN balance_quality = 'FAIR' THEN 1 END) as fair_count,
    COUNT(CASE WHEN balance_quality IN ('DOMINATED (Useless)', 'UNBALANCED (Poor)', 'RARE_STATES (Poor)', 'POOR') THEN 1 END) as poor_count,
    ROUND(AVG(balance_std), 1) as avg_balance_std,
    ROUND(AVG(max_state_pct), 1) as avg_max_state_pct
FROM classifier_quality
GROUP BY classifier_type
ORDER BY excellent_count + good_count DESC;