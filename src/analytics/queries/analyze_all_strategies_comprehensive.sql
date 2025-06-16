-- analyze_all_strategies_comprehensive.sql
-- Analyzes all strategies with proper daily Sharpe ratios
-- This creates a comprehensive analysis for a sample of each strategy type

PRAGMA memory_limit='6GB';
SET threads=8;

-- Set parameters
SET VARIABLE start_date = '2024-03-26 00:00:00';
SET VARIABLE end_date = '2025-01-17 20:00:00';
SET VARIABLE classifier_path = '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet';
SET VARIABLE market_data_path = '/Users/daws/ADMF-PC/data/SPY_1m.parquet';
SET VARIABLE risk_free_rate = 0.0;

-- Setup regime analysis tables
.read 01_setup_regime_analysis.sql

-- Create comprehensive results table
CREATE OR REPLACE TABLE strategy_regime_comprehensive_results AS
WITH empty_results AS (
    SELECT 
        NULL::VARCHAR as strategy_id,
        NULL::VARCHAR as strategy_name,
        NULL::VARCHAR as strategy_type,
        NULL::VARCHAR as current_regime,
        NULL::BIGINT as trading_days,
        NULL::DOUBLE as avg_daily_return_pct,
        NULL::DOUBLE as daily_volatility_pct,
        NULL::DOUBLE as daily_sharpe_ratio,
        NULL::DOUBLE as annualized_sharpe_ratio,
        NULL::DOUBLE as total_return_pct,
        NULL::DOUBLE as win_days_pct,
        NULL::DOUBLE as avg_trades_per_day
    WHERE 1=0
)
SELECT * FROM empty_results;

-- Analyze sample strategies from each type
-- MACD Crossover
INSERT INTO strategy_regime_comprehensive_results
WITH strategy_analysis AS (
    .read 04_calculate_daily_sharpe.sql
)
SELECT 
    'macd_12_26_9' as strategy_id,
    'MACD Crossover 12_26_9' as strategy_name,
    'macd_crossover' as strategy_type,
    * EXCLUDE (strategy_id, strategy_name)
FROM strategy_analysis
WHERE getvariable('strategy_path') = '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet';

-- EMA Crossover
SET VARIABLE strategy_path = '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/ema_crossover_grid/SPY_ema_crossover_grid_7_35.parquet';
SET VARIABLE strategy_id = 'ema_7_35';
SET VARIABLE strategy_name = 'EMA Crossover 7_35';

INSERT INTO strategy_regime_comprehensive_results
.read 04_calculate_daily_sharpe.sql;

-- RSI Threshold
SET VARIABLE strategy_path = '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/rsi_threshold_grid/SPY_rsi_threshold_grid_11_40.parquet';
SET VARIABLE strategy_id = 'rsi_11_40';
SET VARIABLE strategy_name = 'RSI Threshold 11_40';

INSERT INTO strategy_regime_comprehensive_results
.read 04_calculate_daily_sharpe.sql;

-- Bollinger Breakout
SET VARIABLE strategy_path = '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/bollinger_breakout_grid/SPY_bollinger_breakout_grid_11_1.5.parquet';
SET VARIABLE strategy_id = 'bollinger_11_1.5';
SET VARIABLE strategy_name = 'Bollinger Breakout 11_1.5';

INSERT INTO strategy_regime_comprehensive_results
.read 04_calculate_daily_sharpe.sql;

-- Stochastic RSI
SET VARIABLE strategy_path = '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/stochastic_rsi_grid/SPY_stochastic_rsi_grid_12_2_2_80_20.parquet';
SET VARIABLE strategy_id = 'stoch_rsi_12_2_2_80_20';
SET VARIABLE strategy_name = 'Stochastic RSI 12_2_2_80_20';

INSERT INTO strategy_regime_comprehensive_results
.read 04_calculate_daily_sharpe.sql;

-- Show results sorted by Sharpe per regime
SELECT 
    '=== STRATEGIES SORTED BY ANNUALIZED SHARPE RATIO PER REGIME ===' as header;

-- High Volatility Bearish
SELECT 
    '=== HIGH VOLATILITY BEARISH REGIME ===' as regime_section;

SELECT 
    strategy_name,
    trading_days,
    ROUND(avg_daily_return_pct, 3) as avg_daily_ret_pct,
    ROUND(daily_volatility_pct, 3) as daily_vol_pct,
    ROUND(annualized_sharpe_ratio, 2) as ann_sharpe,
    ROUND(total_return_pct, 2) as total_ret_pct,
    ROUND(win_days_pct, 1) as win_days_pct,
    ROUND(avg_trades_per_day, 1) as trades_per_day
FROM strategy_regime_comprehensive_results
WHERE current_regime = 'high_vol_bearish'
ORDER BY annualized_sharpe_ratio DESC;

-- Neutral
SELECT 
    '=== NEUTRAL REGIME ===' as regime_section;

SELECT 
    strategy_name,
    trading_days,
    ROUND(avg_daily_return_pct, 3) as avg_daily_ret_pct,
    ROUND(daily_volatility_pct, 3) as daily_vol_pct,
    ROUND(annualized_sharpe_ratio, 2) as ann_sharpe,
    ROUND(total_return_pct, 2) as total_ret_pct,
    ROUND(win_days_pct, 1) as win_days_pct,
    ROUND(avg_trades_per_day, 1) as trades_per_day
FROM strategy_regime_comprehensive_results
WHERE current_regime = 'neutral'
ORDER BY annualized_sharpe_ratio DESC;

-- Low Vol Bullish
SELECT 
    '=== LOW VOLATILITY BULLISH REGIME ===' as regime_section;

SELECT 
    strategy_name,
    trading_days,
    ROUND(avg_daily_return_pct, 3) as avg_daily_ret_pct,
    ROUND(daily_volatility_pct, 3) as daily_vol_pct,
    ROUND(annualized_sharpe_ratio, 2) as ann_sharpe,
    ROUND(total_return_pct, 2) as total_ret_pct,
    ROUND(win_days_pct, 1) as win_days_pct,
    ROUND(avg_trades_per_day, 1) as trades_per_day
FROM strategy_regime_comprehensive_results
WHERE current_regime = 'low_vol_bullish'
ORDER BY annualized_sharpe_ratio DESC;

-- Low Vol Bearish
SELECT 
    '=== LOW VOLATILITY BEARISH REGIME ===' as regime_section;

SELECT 
    strategy_name,
    trading_days,
    ROUND(avg_daily_return_pct, 3) as avg_daily_ret_pct,
    ROUND(daily_volatility_pct, 3) as daily_vol_pct,
    ROUND(annualized_sharpe_ratio, 2) as ann_sharpe,
    ROUND(total_return_pct, 2) as total_ret_pct,
    ROUND(win_days_pct, 1) as win_days_pct,
    ROUND(avg_trades_per_day, 1) as trades_per_day
FROM strategy_regime_comprehensive_results
WHERE current_regime = 'low_vol_bearish'
ORDER BY annualized_sharpe_ratio DESC;

-- Overall summary
SELECT 
    '=== OVERALL BEST STRATEGIES BY SHARPE ===' as section;

SELECT 
    strategy_name,
    current_regime,
    ROUND(annualized_sharpe_ratio, 2) as ann_sharpe,
    ROUND(total_return_pct, 2) as total_ret_pct,
    ROUND(win_days_pct, 1) as win_days_pct,
    trading_days
FROM strategy_regime_comprehensive_results
WHERE annualized_sharpe_ratio IS NOT NULL
ORDER BY annualized_sharpe_ratio DESC
LIMIT 10;

-- Export results
COPY strategy_regime_comprehensive_results 
TO 'strategy_regime_comprehensive_results.csv' 
WITH (HEADER, DELIMITER ',');