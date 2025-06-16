-- 05_regime_performance_summary.sql
-- Summarizes strategy performance across regimes
-- Can be run after batch analysis to show top performers

PRAGMA memory_limit='2GB';

-- Assumes results are stored in a table or CSV
-- Parameters: results_table or results_csv_path

-- If using CSV input
SET VARIABLE results_csv = COALESCE(getvariable('results_csv'), 'strategy_regime_results.csv');

-- Load results (from CSV or existing table)
CREATE OR REPLACE TEMP TABLE analysis_results AS
SELECT * FROM read_csv_auto(getvariable('results_csv'));

-- Top strategies by Sharpe ratio per regime
SELECT 
    '=== TOP STRATEGIES BY SHARPE RATIO PER REGIME ===' as section;

WITH ranked_strategies AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY entry_regime ORDER BY annualized_sharpe DESC) as sharpe_rank,
        ROW_NUMBER() OVER (PARTITION BY entry_regime ORDER BY net_return_pct DESC) as return_rank
    FROM analysis_results
    WHERE trade_count >= 10  -- Minimum trades for reliability
)
SELECT 
    entry_regime,
    strategy_name,
    trade_count,
    ROUND(annualized_sharpe, 2) as annualized_sharpe,
    ROUND(daily_volatility_pct, 2) as daily_vol_pct,
    ROUND(net_return_pct, 2) as net_return_pct,
    ROUND(win_rate_pct, 1) as win_rate_pct,
    ROUND(avg_duration_min, 1) as avg_duration_min
FROM ranked_strategies
WHERE sharpe_rank <= 5
ORDER BY entry_regime, sharpe_rank;

-- Regime performance comparison
SELECT 
    '=== REGIME PERFORMANCE COMPARISON ===' as section;

SELECT 
    entry_regime,
    COUNT(DISTINCT strategy_name) as strategies_analyzed,
    SUM(trade_count) as total_trades,
    ROUND(AVG(annualized_sharpe), 3) as avg_sharpe,
    ROUND(STDDEV(annualized_sharpe), 3) as sharpe_std,
    ROUND(MAX(annualized_sharpe), 3) as best_sharpe,
    ROUND(MIN(annualized_sharpe), 3) as worst_sharpe,
    ROUND(AVG(net_return_pct), 2) as avg_net_return_pct,
    ROUND(AVG(win_rate_pct), 1) as avg_win_rate_pct
FROM analysis_results
WHERE trade_count >= 10
GROUP BY entry_regime
ORDER BY avg_sharpe DESC;

-- Strategy type performance
SELECT 
    '=== STRATEGY TYPE PERFORMANCE ===' as section;

SELECT 
    strategy_type,
    COUNT(DISTINCT strategy_name) as strategy_count,
    ROUND(AVG(annualized_sharpe), 3) as avg_sharpe,
    ROUND(MAX(annualized_sharpe), 3) as best_sharpe,
    ROUND(AVG(net_return_pct), 2) as avg_net_return_pct,
    ROUND(AVG(trade_count), 0) as avg_trades
FROM analysis_results
GROUP BY strategy_type
HAVING COUNT(DISTINCT strategy_name) >= 3
ORDER BY avg_sharpe DESC
LIMIT 10;

-- Identify regime specialists
SELECT 
    '=== REGIME SPECIALISTS ===' as section;

WITH strategy_regime_spread AS (
    SELECT 
        strategy_name,
        COUNT(DISTINCT entry_regime) as regimes_traded,
        MAX(annualized_sharpe) - MIN(annualized_sharpe) as sharpe_spread,
        STRING_AGG(
            entry_regime || ':' || ROUND(annualized_sharpe, 2), 
            ', ' 
            ORDER BY annualized_sharpe DESC
        ) as regime_sharpes
    FROM analysis_results
    WHERE trade_count >= 10
    GROUP BY strategy_name
)
SELECT 
    strategy_name,
    regimes_traded,
    ROUND(sharpe_spread, 2) as sharpe_spread,
    regime_sharpes
FROM strategy_regime_spread
WHERE sharpe_spread > 1.0  -- Large performance differences across regimes
ORDER BY sharpe_spread DESC
LIMIT 20;