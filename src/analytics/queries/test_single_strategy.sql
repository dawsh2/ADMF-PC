-- Test single strategy analysis with proper daily Sharpe calculation
-- This combines setup and analysis in one file for testing

PRAGMA memory_limit='4GB';
SET threads=8;

-- Set parameters
SET VARIABLE start_date = '2024-03-26 00:00:00';
SET VARIABLE end_date = '2025-01-17 20:00:00';
SET VARIABLE classifier_path = '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet';
SET VARIABLE market_data_path = '/Users/daws/ADMF-PC/data/SPY_1m.parquet';

-- Test with MACD strategy
SET VARIABLE strategy_path = '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet';
SET VARIABLE strategy_id = 'macd_12_26_9';
SET VARIABLE strategy_name = 'MACD Crossover 12_26_9';
SET VARIABLE risk_free_rate = 0.0;

-- First run the setup
.read 01_setup_regime_analysis.sql

-- Then calculate daily Sharpe
.read 04_calculate_daily_sharpe.sql