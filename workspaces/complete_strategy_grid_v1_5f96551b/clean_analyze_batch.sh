#!/bin/bash
# Clean batch analysis of strategies with proper daily Sharpe ratios

# Create results directory
mkdir -p results

# Initialize results file
echo "strategy_id,strategy_name,strategy_type,current_regime,trading_days,avg_daily_return_pct,daily_volatility_pct,daily_sharpe_ratio,annualized_sharpe_ratio,total_return_pct,win_days_pct,avg_trades_per_day" > results/clean_strategies_sharpe.csv

# First, setup regime analysis tables once
echo "Setting up regime analysis tables..."
duckdb analytics.duckdb << EOF > /dev/null 2>&1
PRAGMA memory_limit='4GB';
SET threads=8;

SET VARIABLE start_date = '2024-03-26 00:00:00';
SET VARIABLE end_date = '2025-01-17 20:00:00';
SET VARIABLE classifier_path = '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet';
SET VARIABLE market_data_path = '/Users/daws/ADMF-PC/data/SPY_1m.parquet';

.read 01_setup_regime_analysis.sql
EOF

# Function to analyze a single strategy
analyze_strategy() {
    local strategy_path=$1
    local strategy_id=$2
    local strategy_name=$3
    local strategy_type=$4
    
    echo "Analyzing: $strategy_name"
    
    # Run analysis and extract only the data rows
    duckdb analytics.duckdb -csv -noheader << EOF 2>/dev/null | grep -E "^$strategy_id," | sed "s/$/,$strategy_type/" >> results/clean_strategies_sharpe.csv
PRAGMA memory_limit='4GB';
SET threads=8;

SET VARIABLE start_date = '2024-03-26 00:00:00';
SET VARIABLE end_date = '2025-01-17 20:00:00';
SET VARIABLE classifier_path = '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet';
SET VARIABLE market_data_path = '/Users/daws/ADMF-PC/data/SPY_1m.parquet';
SET VARIABLE risk_free_rate = 0.0;
SET VARIABLE strategy_path = '$strategy_path';
SET VARIABLE strategy_id = '$strategy_id';
SET VARIABLE strategy_name = '$strategy_name';

.read 04_calculate_daily_sharpe.sql
EOF
}

# Analyze sample strategies
echo "Starting batch analysis..."

# MACD strategies
analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet" \
    "macd_12_26_9" "MACD Crossover 12_26_9" "macd_crossover"

analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_20_7.parquet" \
    "macd_12_20_7" "MACD Crossover 12_20_7" "macd_crossover"

# EMA strategies
analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/ema_crossover_grid/SPY_ema_crossover_grid_7_35.parquet" \
    "ema_7_35" "EMA Crossover 7_35" "ema_crossover"

analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/ema_crossover_grid/SPY_ema_crossover_grid_10_50.parquet" \
    "ema_10_50" "EMA Crossover 10_50" "ema_crossover"

# RSI strategies
analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/rsi_threshold_grid/SPY_rsi_threshold_grid_11_40.parquet" \
    "rsi_11_40" "RSI Threshold 11_40" "rsi_threshold"

analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/rsi_threshold_grid/SPY_rsi_threshold_grid_7_45.parquet" \
    "rsi_7_45" "RSI Threshold 7_45" "rsi_threshold"

# Bollinger strategies
analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/bollinger_breakout_grid/SPY_bollinger_breakout_grid_11_1.5.parquet" \
    "bollinger_11_1.5" "Bollinger Breakout 11_1.5" "bollinger_breakout"

analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/bollinger_breakout_grid/SPY_bollinger_breakout_grid_19_2.0.parquet" \
    "bollinger_19_2.0" "Bollinger Breakout 19_2.0" "bollinger_breakout"

# Ultimate Oscillator
analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/ultimate_oscillator_grid/SPY_ultimate_oscillator_grid_1_2_4_70_30.parquet" \
    "uo_1_2_4_70_30" "Ultimate Oscillator 1_2_4_70_30" "ultimate_oscillator"

echo "Analysis complete! Results saved to results/clean_strategies_sharpe.csv"

# Create summary
echo -e "\n=== ANALYSIS SUMMARY ==="
duckdb analytics.duckdb << EOF
-- Load results
CREATE TABLE clean_results AS 
SELECT * FROM read_csv_auto('results/clean_strategies_sharpe.csv');

-- Show top strategies by Sharpe
SELECT 
    '=== TOP STRATEGIES BY ANNUALIZED SHARPE RATIO ===' as header;

SELECT 
    strategy_name,
    current_regime,
    ROUND(annualized_sharpe_ratio, 2) as ann_sharpe,
    ROUND(total_return_pct, 2) as total_ret_pct,
    ROUND(win_days_pct, 1) as win_days_pct,
    trading_days
FROM clean_results
ORDER BY annualized_sharpe_ratio DESC
LIMIT 15;

-- Summary by regime
SELECT 
    '=== AVERAGE SHARPE BY REGIME ===' as header;

SELECT 
    current_regime,
    COUNT(*) as strategies,
    ROUND(AVG(annualized_sharpe_ratio), 2) as avg_sharpe,
    ROUND(MIN(annualized_sharpe_ratio), 2) as min_sharpe,
    ROUND(MAX(annualized_sharpe_ratio), 2) as max_sharpe
FROM clean_results
GROUP BY current_regime
ORDER BY avg_sharpe DESC;

-- Best strategy per regime
SELECT 
    '=== BEST STRATEGY PER REGIME ===' as header;

WITH ranked AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY current_regime ORDER BY annualized_sharpe_ratio DESC) as rank
    FROM clean_results
)
SELECT 
    current_regime,
    strategy_name,
    ROUND(annualized_sharpe_ratio, 2) as ann_sharpe,
    ROUND(total_return_pct, 2) as total_ret_pct
FROM ranked
WHERE rank = 1
ORDER BY ann_sharpe DESC;
EOF