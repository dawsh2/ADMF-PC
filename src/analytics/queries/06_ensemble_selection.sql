-- Ensemble Selection Query
-- Selects optimal strategy ensembles per regime based on multiple criteria

-- Load strategy results
CREATE OR REPLACE TABLE ensemble_analysis AS 
SELECT 
    strategy_id,
    strategy_name,
    strategy_type,
    current_regime,
    trading_days,
    avg_daily_return_pct / 100.0 as avg_daily_return,
    daily_volatility_pct / 100.0 as daily_volatility,
    TRY_CAST(annualized_sharpe_ratio AS DOUBLE) as sharpe_ratio,
    total_return_pct / 100.0 as total_return,
    win_days_pct / 100.0 as win_rate
FROM read_csv_auto('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/results/all_strategies_analysis_20250616_175832.csv')
WHERE TRY_CAST(annualized_sharpe_ratio AS DOUBLE) IS NOT NULL
  AND trading_days >= 20;

-- Calculate selection scores
CREATE OR REPLACE TABLE strategy_scores AS
SELECT 
    strategy_name,
    strategy_type,
    current_regime,
    sharpe_ratio,
    total_return,
    daily_volatility,
    win_rate,
    trading_days,
    
    -- Consistency score (Sharpe × sqrt(win_rate))
    sharpe_ratio * SQRT(win_rate) as consistency_score,
    
    -- Risk-adjusted score
    sharpe_ratio * (1 - daily_volatility) as risk_adj_score,
    
    -- Composite selection score
    (0.30 * LEAST(sharpe_ratio / 3.0, 1.0) +       -- Sharpe (capped at 1.0)
     0.20 * LEAST(sharpe_ratio * SQRT(win_rate) / 2.0, 1.0) +  -- Consistency
     0.20 * win_rate +                              -- Win rate
     0.15 * LEAST(total_return / 0.20, 1.0) +      -- Returns (capped)
     0.15 * (1 - LEAST(daily_volatility / 0.05, 1.0))  -- Low volatility bonus
    ) as selection_score
FROM ensemble_analysis
WHERE sharpe_ratio >= 0.5;  -- Minimum Sharpe filter

-- Apply diversification rules and select top candidates
CREATE OR REPLACE TABLE ensemble_candidates AS
WITH ranked_strategies AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY current_regime, strategy_type 
            ORDER BY selection_score DESC
        ) as type_rank,
        ROW_NUMBER() OVER (
            PARTITION BY current_regime 
            ORDER BY selection_score DESC
        ) as overall_rank
    FROM strategy_scores
),
type_limited AS (
    -- Maximum 2 strategies per type for diversification
    SELECT *
    FROM ranked_strategies
    WHERE type_rank <= 2
)
SELECT 
    current_regime,
    strategy_name,
    strategy_type,
    ROUND(sharpe_ratio, 3) as sharpe_ratio,
    ROUND(total_return * 100, 2) as total_return_pct,
    ROUND(daily_volatility * 100, 2) as volatility_pct,
    ROUND(win_rate * 100, 1) as win_rate_pct,
    ROUND(consistency_score, 3) as consistency_score,
    ROUND(selection_score, 3) as selection_score,
    overall_rank
FROM type_limited
WHERE overall_rank <= 20  -- Keep top 20 for final selection
ORDER BY current_regime, selection_score DESC;

-- Show top 10 ensemble for each regime
SELECT 
    '=== ENSEMBLE RECOMMENDATIONS BY REGIME ===' as header;

WITH top_ensembles AS (
    SELECT *
    FROM ensemble_candidates
    WHERE overall_rank <= 10
)
SELECT 
    current_regime,
    strategy_name,
    strategy_type,
    sharpe_ratio,
    total_return_pct,
    win_rate_pct,
    selection_score
FROM top_ensembles
ORDER BY current_regime, selection_score DESC;

-- Summary statistics per regime
SELECT 
    '=== ENSEMBLE CHARACTERISTICS BY REGIME ===' as header;

WITH selected_strategies AS (
    SELECT * 
    FROM ensemble_candidates 
    WHERE overall_rank <= 10
)
SELECT 
    current_regime,
    COUNT(*) as ensemble_size,
    COUNT(DISTINCT strategy_type) as unique_strategy_types,
    ROUND(AVG(sharpe_ratio), 3) as avg_sharpe,
    ROUND(MIN(sharpe_ratio), 3) as min_sharpe,
    ROUND(MAX(sharpe_ratio), 3) as max_sharpe,
    ROUND(AVG(total_return_pct), 2) as avg_return_pct,
    ROUND(AVG(volatility_pct), 2) as avg_volatility_pct,
    ROUND(AVG(win_rate_pct), 1) as avg_win_rate_pct
FROM selected_strategies
GROUP BY current_regime
ORDER BY current_regime;

-- Export detailed ensemble
SELECT * FROM ensemble_candidates
WHERE overall_rank <= 10
ORDER BY current_regime, selection_score DESC;