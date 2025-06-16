-- Get state details for top candidates
WITH all_classifier_states AS (
    SELECT 
        'volatility_momentum' as classifier_type,
        strat as classifier_id,
        val as regime_state,
        COUNT(*) as state_occurrences
    FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/*.parquet')
    WHERE strat = 'SPY_volatility_momentum_grid_08_55_35'  -- Pick one of the best ones
    GROUP BY strat, val
),
state_percentages AS (
    SELECT 
        classifier_type,
        classifier_id,
        regime_state,
        state_occurrences,
        ROUND(state_occurrences * 100.0 / SUM(state_occurrences) OVER (PARTITION BY classifier_type, classifier_id), 1) as state_pct
    FROM all_classifier_states
)

SELECT 
    'STATE BREAKDOWN FOR BEST CLASSIFIER' as analysis_type,
    classifier_type,
    classifier_id,
    regime_state,
    state_occurrences,
    state_pct
FROM state_percentages
ORDER BY state_pct DESC;