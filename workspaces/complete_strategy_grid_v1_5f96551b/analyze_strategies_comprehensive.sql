-- Comprehensive strategy analysis with proper daily Sharpe ratios
-- Analyzes multiple strategies across all regimes

PRAGMA memory_limit='6GB';
SET threads=8;

-- Set parameters
SET VARIABLE start_date = '2024-03-26 00:00:00';
SET VARIABLE end_date = '2025-01-17 20:00:00';
SET VARIABLE classifier_path = '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet';
SET VARIABLE market_data_path = '/Users/daws/ADMF-PC/data/SPY_1m.parquet';
SET VARIABLE risk_free_rate = 0.0;

-- Setup regime analysis tables (run once)
.read 01_setup_regime_analysis.sql

-- Create results table
CREATE OR REPLACE TABLE strategy_regime_comprehensive_results (
    strategy_id VARCHAR,
    strategy_name VARCHAR,
    strategy_type VARCHAR,
    current_regime VARCHAR,
    trading_days BIGINT,
    avg_daily_return_pct DOUBLE,
    daily_volatility_pct DOUBLE,
    daily_sharpe_ratio DOUBLE,
    annualized_sharpe_ratio DOUBLE,
    total_return_pct DOUBLE,
    win_days_pct DOUBLE,
    avg_trades_per_day DOUBLE
);

-- Analyze MACD Crossover
SET VARIABLE strategy_path = '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet';
SET VARIABLE strategy_id = 'macd_12_26_9';
SET VARIABLE strategy_name = 'MACD Crossover 12_26_9';

INSERT INTO strategy_regime_comprehensive_results
SELECT 
    getvariable('strategy_id') as strategy_id,
    getvariable('strategy_name') as strategy_name,
    'macd_crossover' as strategy_type,
    current_regime,
    trading_days,
    avg_daily_return_pct,
    daily_volatility_pct,
    daily_sharpe_ratio,
    annualized_sharpe_ratio,
    total_return_pct,
    win_days_pct,
    avg_trades_per_day
FROM (.read 04_calculate_daily_sharpe.sql);

-- Show results so far
SELECT * FROM strategy_regime_comprehensive_results;