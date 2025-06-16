-- 03_analyze_strategy_batch.sql
-- Analyzes multiple strategies of the same type by regime
-- Requires: 01_setup_regime_analysis.sql to be run first
-- Parameters: strategy_type, limit (optional)

PRAGMA memory_limit='4GB';

-- Parameters
SET VARIABLE strategy_type = getvariable('strategy_type');
SET VARIABLE limit = COALESCE(getvariable('limit'), 1000);

-- Create results table for batch
CREATE OR REPLACE TEMP TABLE batch_results (
    strategy_id VARCHAR,
    strategy_name VARCHAR,
    strategy_type VARCHAR,
    entry_regime VARCHAR,
    trade_count BIGINT,
    avg_return_pct DOUBLE,
    gross_return_pct DOUBLE,
    net_return_pct DOUBLE,
    sharpe_ratio DOUBLE,
    annualized_sharpe DOUBLE,
    win_rate_pct DOUBLE,
    long_pct DOUBLE,
    avg_duration_min DOUBLE,
    trades_per_day DOUBLE,
    daily_return_pct DOUBLE,
    daily_volatility_pct DOUBLE
);

-- Get strategies to analyze
CREATE TEMP TABLE strategies_to_analyze AS
SELECT 
    strategy_id,
    strategy_name,
    strategy_type,
    signal_file_path
FROM analytics.strategies
WHERE strategy_type = getvariable('strategy_type')
  AND signal_file_path IS NOT NULL
ORDER BY strategy_id
LIMIT getvariable('limit')::integer;

-- Show what we're analyzing
SELECT 
    COUNT(*) as strategies_to_analyze,
    strategy_type
FROM strategies_to_analyze
GROUP BY strategy_type;

-- Note: In DuckDB, we need to process strategies individually due to dynamic file path limitations
-- This would typically be done in a loop via Python or another scripting language
-- For demonstration, we'll show the structure for analyzing one strategy

-- Example for first strategy (would be looped in practice)
WITH first_strategy AS (
    SELECT * FROM strategies_to_analyze LIMIT 1
)
SELECT 
    'To analyze all strategies, use the Python script: analyze_strategies_batch.py' as note,
    'This SQL shows the structure for a single strategy analysis' as info;