#!/usr/bin/env python3
"""
Comprehensive batch analysis of ALL strategies by regime
Processes all 1,235 strategies with proper daily Sharpe ratios
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Configuration
ANALYTICS_DB = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/analytics.duckdb"
WORKSPACE_PATH = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b"
RESULTS_DIR = Path(WORKSPACE_PATH) / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Analysis parameters
START_DATE = "2024-03-26 00:00:00"
END_DATE = "2025-01-17 20:00:00"
CLASSIFIER_PATH = f"{WORKSPACE_PATH}/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet"
MARKET_DATA_PATH = "/Users/daws/ADMF-PC/data/SPY_1m.parquet"

# Number of parallel workers
MAX_WORKERS = 4

def setup_regime_tables():
    """One-time setup of regime and market data tables"""
    print("Setting up regime analysis tables...")
    
    setup_query = f"""
    PRAGMA memory_limit='4GB';
    SET threads=4;
    
    -- Create regime timeline
    CREATE OR REPLACE TABLE regime_timeline AS
    WITH 
    regime_sparse AS (
        SELECT 
            ts::timestamp as regime_time,
            val as regime_state
        FROM read_parquet('{CLASSIFIER_PATH}')
        WHERE ts::timestamp >= '{START_DATE}'
          AND ts::timestamp <= '{END_DATE}'
    ),
    market_times AS (
        SELECT DISTINCT
            timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est
        FROM read_parquet('{MARKET_DATA_PATH}')
        WHERE timestamp >= TIMESTAMP '{START_DATE}' - INTERVAL 4 HOUR
          AND timestamp <= TIMESTAMP '{END_DATE}' + INTERVAL 4 HOUR
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
    
    -- Create market prices table
    CREATE OR REPLACE TABLE market_prices AS
    SELECT 
        timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est,
        close
    FROM read_parquet('{MARKET_DATA_PATH}')
    WHERE timestamp >= TIMESTAMP '{START_DATE}' - INTERVAL 4 HOUR
      AND timestamp <= TIMESTAMP '{END_DATE}' + INTERVAL 4 HOUR;
    
    CREATE INDEX idx_price_time ON market_prices(timestamp_est);
    
    -- Verify setup
    SELECT 
        'Setup complete' as status,
        (SELECT COUNT(*) FROM regime_timeline) as regime_rows,
        (SELECT COUNT(*) FROM market_prices) as price_rows;
    """
    
    # Execute setup
    result = subprocess.run(
        ['duckdb', ANALYTICS_DB],
        input=setup_query,
        text=True,
        capture_output=True
    )
    
    if result.returncode != 0:
        print(f"Setup error: {result.stderr}")
        return False
    
    print("Setup complete!")
    return True

def analyze_single_strategy(strategy_info):
    """Analyze a single strategy and return results"""
    strategy_id = strategy_info['strategy_id']
    signal_path = strategy_info['signal_file_path']
    
    # Skip if no signal file
    if not signal_path or pd.isna(signal_path):
        return None
    
    # Build full path
    full_signal_path = f"{WORKSPACE_PATH}/{signal_path}"
    
    # Check if file exists
    if not Path(full_signal_path).exists():
        print(f"Signal file not found: {full_signal_path}")
        return None
    
    analysis_query = f"""
    PRAGMA memory_limit='2GB';
    
    WITH 
    strategy_signals AS (
        SELECT 
            ts::timestamp as signal_time,
            val as signal_value,
            LAG(val) OVER (ORDER BY ts::timestamp) as prev_signal,
            LEAD(ts::timestamp) OVER (ORDER BY ts::timestamp) as next_signal_time
        FROM read_parquet('{full_signal_path}')
        WHERE ts::timestamp >= '{START_DATE}'
          AND ts::timestamp <= '{END_DATE}'
    ),
    position_changes AS (
        SELECT 
            signal_time,
            signal_value as position,
            ROW_NUMBER() OVER (ORDER BY signal_time) as change_num
        FROM strategy_signals
        WHERE (prev_signal IS NULL OR prev_signal != signal_value)
    ),
    position_timeline AS (
        SELECT 
            mp.timestamp_est,
            mp.close,
            rt.current_regime,
            LAST_VALUE(pc.position IGNORE NULLS) OVER (
                ORDER BY mp.timestamp_est 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as position
        FROM market_prices mp
        LEFT JOIN regime_timeline rt ON mp.timestamp_est = rt.timestamp_est
        LEFT JOIN position_changes pc ON mp.timestamp_est = pc.signal_time
    ),
    minute_returns AS (
        SELECT 
            timestamp_est,
            current_regime,
            position,
            close,
            LAG(close) OVER (ORDER BY timestamp_est) as prev_close,
            CASE 
                WHEN position = 1 THEN (close - LAG(close) OVER (ORDER BY timestamp_est)) / LAG(close) OVER (ORDER BY timestamp_est)
                WHEN position = -1 THEN (LAG(close) OVER (ORDER BY timestamp_est) - close) / LAG(close) OVER (ORDER BY timestamp_est)
                ELSE 0
            END as minute_return
        FROM position_timeline
        WHERE position IS NOT NULL
    ),
    daily_returns AS (
        SELECT 
            DATE_TRUNC('day', timestamp_est) as trading_day,
            current_regime,
            SUM(minute_return) as daily_return,
            COUNT(DISTINCT position) as position_changes,
            SUM(CASE WHEN position != 0 THEN 1 ELSE 0 END) as minutes_in_position
        FROM minute_returns
        WHERE prev_close IS NOT NULL
        GROUP BY DATE_TRUNC('day', timestamp_est), current_regime
    ),
    regime_performance AS (
        SELECT 
            current_regime,
            COUNT(*) as trading_days,
            AVG(daily_return) as avg_daily_return,
            STDDEV(daily_return) as daily_volatility,
            (AVG(daily_return) - 0.0) / NULLIF(STDDEV(daily_return), 0) as daily_sharpe,
            (AVG(daily_return) - 0.0) / NULLIF(STDDEV(daily_return), 0) * SQRT(252) as annualized_sharpe,
            SUM(daily_return) as total_return,
            SUM(CASE WHEN daily_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_days_pct,
            AVG(position_changes) as avg_trades_per_day
        FROM daily_returns
        GROUP BY current_regime
        HAVING COUNT(*) >= 5
    )
    SELECT 
        '{strategy_id}' as strategy_id,
        '{strategy_info['strategy_name']}' as strategy_name,
        '{strategy_info['strategy_type']}' as strategy_type,
        current_regime,
        trading_days,
        ROUND(avg_daily_return * 100, 4) as avg_daily_return_pct,
        ROUND(daily_volatility * 100, 4) as daily_volatility_pct,
        ROUND(daily_sharpe, 3) as daily_sharpe_ratio,
        ROUND(annualized_sharpe, 3) as annualized_sharpe_ratio,
        ROUND(total_return * 100, 2) as total_return_pct,
        ROUND(win_days_pct, 1) as win_days_pct,
        ROUND(avg_trades_per_day, 1) as avg_trades_per_day
    FROM regime_performance;
    """
    
    try:
        # Execute analysis
        result = subprocess.run(
            ['duckdb', ANALYTICS_DB, '-csv', '-noheader'],
            input=analysis_query,
            text=True,
            capture_output=True
        )
        
        if result.returncode != 0:
            print(f"Error analyzing {strategy_id}: {result.stderr}")
            return None
        
        # Parse CSV output
        if result.stdout.strip():
            rows = []
            for line in result.stdout.strip().split('\n'):
                rows.append(line)
            return rows
        
    except Exception as e:
        print(f"Exception analyzing {strategy_id}: {str(e)}")
    
    return None

def process_batch(strategies_batch, batch_num):
    """Process a batch of strategies"""
    results = []
    
    for i, strategy in enumerate(strategies_batch):
        if i % 10 == 0:
            print(f"  Batch {batch_num}: Processing strategy {i+1}/{len(strategies_batch)}")
        
        result = analyze_single_strategy(strategy)
        if result:
            results.extend(result)
    
    return results

def main():
    """Main analysis function"""
    print(f"Starting comprehensive analysis of ALL strategies")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Using {MAX_WORKERS} parallel workers")
    
    # Setup regime tables once
    if not setup_regime_tables():
        print("Failed to setup regime tables")
        return
    
    # Get all strategies
    print("\nLoading strategy list...")
    strategies_query = """
    SELECT 
        strategy_id,
        strategy_name,
        strategy_type,
        signal_file_path
    FROM analytics.strategies
    WHERE signal_file_path IS NOT NULL
    ORDER BY strategy_type, strategy_name;
    """
    
    result = subprocess.run(
        ['duckdb', ANALYTICS_DB, '-csv'],
        input=strategies_query,
        text=True,
        capture_output=True
    )
    
    if result.returncode != 0:
        print(f"Error loading strategies: {result.stderr}")
        return
    
    # Parse strategies
    from io import StringIO
    strategies_df = pd.read_csv(StringIO(result.stdout))
    print(f"Found {len(strategies_df)} strategies to analyze")
    
    # Convert to list of dicts
    strategies = strategies_df.to_dict('records')
    
    # Split into batches
    batch_size = len(strategies) // MAX_WORKERS + 1
    batches = [strategies[i:i + batch_size] for i in range(0, len(strategies), batch_size)]
    
    print(f"\nProcessing in {len(batches)} batches of ~{batch_size} strategies each")
    
    # Process batches in parallel
    all_results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(process_batch, batch, i): i 
            for i, batch in enumerate(batches)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_num = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                print(f"Completed batch {batch_num + 1}/{len(batches)}")
            except Exception as e:
                print(f"Batch {batch_num} failed: {str(e)}")
    
    # Save results
    if all_results:
        # Write CSV header
        csv_header = "strategy_id,strategy_name,strategy_type,current_regime,trading_days,avg_daily_return_pct,daily_volatility_pct,daily_sharpe_ratio,annualized_sharpe_ratio,total_return_pct,win_days_pct,avg_trades_per_day"
        
        output_file = RESULTS_DIR / f"all_strategies_regime_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(output_file, 'w') as f:
            f.write(csv_header + '\n')
            for row in all_results:
                f.write(row + '\n')
        
        print(f"\nResults saved to: {output_file}")
        
        # Generate summary
        print("\nGenerating summary...")
        generate_summary(output_file)
    
    elapsed = time.time() - start_time
    print(f"\nAnalysis complete in {elapsed:.1f} seconds")

def generate_summary(results_file):
    """Generate summary statistics from results"""
    
    summary_query = f"""
    -- Load results
    CREATE TABLE analysis_results AS 
    SELECT * FROM read_csv_auto('{results_file}');
    
    -- Top strategies by Sharpe
    SELECT 
        '=== TOP 20 STRATEGIES BY ANNUALIZED SHARPE RATIO ===' as header;
    
    SELECT 
        strategy_name,
        current_regime,
        annualized_sharpe_ratio as sharpe,
        total_return_pct as return_pct,
        win_days_pct,
        trading_days
    FROM analysis_results
    WHERE annualized_sharpe_ratio IS NOT NULL
    ORDER BY annualized_sharpe_ratio DESC
    LIMIT 20;
    
    -- Best strategy per regime
    SELECT 
        '=== BEST STRATEGY PER REGIME ===' as header;
    
    WITH ranked AS (
        SELECT 
            *,
            ROW_NUMBER() OVER (PARTITION BY current_regime ORDER BY annualized_sharpe_ratio DESC) as rank
        FROM analysis_results
        WHERE trading_days >= 20
    )
    SELECT 
        current_regime,
        strategy_name,
        strategy_type,
        annualized_sharpe_ratio as sharpe,
        total_return_pct as return_pct,
        win_days_pct
    FROM ranked
    WHERE rank <= 5
    ORDER BY current_regime, rank;
    
    -- Strategy type performance
    SELECT 
        '=== STRATEGY TYPE AVERAGE PERFORMANCE ===' as header;
    
    SELECT 
        strategy_type,
        COUNT(DISTINCT strategy_name) as strategies,
        ROUND(AVG(annualized_sharpe_ratio), 3) as avg_sharpe,
        ROUND(MAX(annualized_sharpe_ratio), 3) as best_sharpe,
        ROUND(AVG(total_return_pct), 2) as avg_return_pct
    FROM analysis_results
    WHERE trading_days >= 20
    GROUP BY strategy_type
    HAVING COUNT(DISTINCT strategy_name) >= 3
    ORDER BY avg_sharpe DESC;
    
    -- Summary stats
    SELECT 
        '=== OVERALL STATISTICS ===' as header;
    
    SELECT 
        COUNT(DISTINCT strategy_id) as total_strategies,
        COUNT(*) as total_regime_combinations,
        COUNT(CASE WHEN annualized_sharpe_ratio > 0 THEN 1 END) as positive_sharpe_count,
        COUNT(CASE WHEN total_return_pct > 0 THEN 1 END) as profitable_count,
        ROUND(MAX(annualized_sharpe_ratio), 3) as best_sharpe,
        ROUND(MIN(annualized_sharpe_ratio), 3) as worst_sharpe
    FROM analysis_results;
    """
    
    # Execute summary
    subprocess.run(
        ['duckdb', ANALYTICS_DB],
        input=summary_query,
        text=True
    )

if __name__ == "__main__":
    main()