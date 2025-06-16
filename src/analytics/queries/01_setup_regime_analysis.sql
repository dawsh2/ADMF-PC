-- 01_setup_regime_analysis.sql
-- Creates temporary tables for regime-based strategy analysis
-- These tables are used by subsequent queries

PRAGMA memory_limit='4GB';
SET threads=8;

-- Parameters (can be overridden)
SET VARIABLE start_date = COALESCE(getvariable('start_date'), '2024-03-26 00:00:00');
SET VARIABLE end_date = COALESCE(getvariable('end_date'), '2025-01-17 20:00:00');
SET VARIABLE classifier_path = COALESCE(getvariable('classifier_path'), 
    '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet');
SET VARIABLE market_data_path = COALESCE(getvariable('market_data_path'), 
    '/Users/daws/ADMF-PC/data/SPY_1m.parquet');

-- Create regime timeline with forward-fill
CREATE OR REPLACE TEMP TABLE regime_timeline AS
WITH 
regime_sparse AS (
    SELECT 
        ts::timestamp as regime_time,
        val as regime_state
    FROM read_parquet(getvariable('classifier_path'))
    WHERE ts::timestamp >= getvariable('start_date')::timestamp
      AND ts::timestamp <= getvariable('end_date')::timestamp
),
market_times AS (
    SELECT DISTINCT
        timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est
    FROM read_parquet(getvariable('market_data_path'))
    WHERE timestamp >= getvariable('start_date')::timestamp WITH TIME ZONE - INTERVAL 4 HOUR
      AND timestamp <= getvariable('end_date')::timestamp WITH TIME ZONE + INTERVAL 4 HOUR
)
SELECT 
    mt.timestamp_est,
    LAST_VALUE(rs.regime_state IGNORE NULLS) OVER (
        ORDER BY mt.timestamp_est 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as current_regime
FROM market_times mt
LEFT JOIN regime_sparse rs ON mt.timestamp_est = rs.regime_time;

CREATE INDEX idx_regime_time ON regime_timeline(timestamp_est);

-- Create market prices table with timezone adjustment
CREATE OR REPLACE TEMP TABLE market_prices AS
SELECT 
    timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est,
    close
FROM read_parquet(getvariable('market_data_path'))
WHERE timestamp >= getvariable('start_date')::timestamp WITH TIME ZONE - INTERVAL 4 HOUR
  AND timestamp <= getvariable('end_date')::timestamp WITH TIME ZONE + INTERVAL 4 HOUR;

CREATE INDEX idx_price_time ON market_prices(timestamp_est);

-- Verify data loaded
SELECT 
    'Regime Timeline' as data_type,
    COUNT(*) as rows,
    COUNT(DISTINCT current_regime) as unique_regimes,
    MIN(timestamp_est) as min_time,
    MAX(timestamp_est) as max_time
FROM regime_timeline
WHERE current_regime IS NOT NULL
UNION ALL
SELECT 
    'Market Prices' as data_type,
    COUNT(*) as rows,
    NULL as unique_regimes,
    MIN(timestamp_est) as min_time,
    MAX(timestamp_est) as max_time
FROM market_prices;

-- Show regime distribution
SELECT 
    current_regime,
    COUNT(*) as minutes_in_regime,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct_of_time
FROM regime_timeline
WHERE current_regime IS NOT NULL
GROUP BY current_regime
ORDER BY pct_of_time DESC;