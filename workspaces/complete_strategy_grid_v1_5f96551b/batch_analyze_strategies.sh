#!/bin/bash
# Batch analyze strategies with proper daily Sharpe ratios

# Create results directory
mkdir -p results

# Initialize results file
echo "strategy_id,strategy_name,strategy_type,current_regime,trading_days,avg_daily_return_pct,daily_volatility_pct,daily_sharpe_ratio,annualized_sharpe_ratio,total_return_pct,win_days_pct,avg_trades_per_day" > results/all_strategies_sharpe.csv

# Function to analyze a single strategy
analyze_strategy() {
    local strategy_path=$1
    local strategy_id=$2
    local strategy_name=$3
    local strategy_type=$4
    
    echo "Analyzing: $strategy_name"
    
    # Create temporary SQL file
    cat > temp_analyze.sql << EOF
PRAGMA memory_limit='4GB';
SET threads=8;

-- Set parameters
SET VARIABLE start_date = '2024-03-26 00:00:00';
SET VARIABLE end_date = '2025-01-17 20:00:00';
SET VARIABLE classifier_path = '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet';
SET VARIABLE market_data_path = '/Users/daws/ADMF-PC/data/SPY_1m.parquet';
SET VARIABLE risk_free_rate = 0.0;
SET VARIABLE strategy_path = '$strategy_path';
SET VARIABLE strategy_id = '$strategy_id';
SET VARIABLE strategy_name = '$strategy_name';

-- Run setup if not already done
.read 01_setup_regime_analysis.sql

-- Calculate daily Sharpe
.read 04_calculate_daily_sharpe.sql
EOF
    
    # Run analysis and append to CSV
    duckdb analytics.duckdb -csv -noheader < temp_analyze.sql | \
    awk -v type="$strategy_type" '{print $0 "," type}' >> results/all_strategies_sharpe.csv
    
    rm temp_analyze.sql
}

# Analyze sample strategies
echo "Starting batch analysis..."

# MACD strategies
analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet" \
    "macd_12_26_9" "MACD Crossover 12_26_9" "macd_crossover"

analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_13_3.parquet" \
    "macd_5_13_3" "MACD Crossover 5_13_3" "macd_crossover"

# EMA strategies
analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/ema_crossover_grid/SPY_ema_crossover_grid_7_35.parquet" \
    "ema_7_35" "EMA Crossover 7_35" "ema_crossover"

analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/ema_crossover_grid/SPY_ema_crossover_grid_3_18.parquet" \
    "ema_3_18" "EMA Crossover 3_18" "ema_crossover"

# RSI strategies
analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/rsi_threshold_grid/SPY_rsi_threshold_grid_11_40.parquet" \
    "rsi_11_40" "RSI Threshold 11_40" "rsi_threshold"

analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/rsi_threshold_grid/SPY_rsi_threshold_grid_7_45.parquet" \
    "rsi_7_45" "RSI Threshold 7_45" "rsi_threshold"

# Bollinger strategies
analyze_strategy "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals/bollinger_breakout_grid/SPY_bollinger_breakout_grid_11_1.5.parquet" \
    "bollinger_11_1.5" "Bollinger Breakout 11_1.5" "bollinger_breakout"

echo "Analysis complete! Results saved to results/all_strategies_sharpe.csv"

# Create summary
duckdb analytics.duckdb << EOF
-- Load results
CREATE TABLE results AS 
SELECT * FROM read_csv_auto('results/all_strategies_sharpe.csv');

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
FROM results
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
FROM results
GROUP BY current_regime
ORDER BY avg_sharpe DESC;
EOF