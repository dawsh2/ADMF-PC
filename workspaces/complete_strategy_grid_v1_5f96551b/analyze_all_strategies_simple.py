#!/usr/bin/env python3
"""
Simple sequential analysis of all strategies
Constructs signal file paths from strategy names
"""

import subprocess
import os
from pathlib import Path
import time
from datetime import datetime

# Configuration
ANALYTICS_DB = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/analytics.duckdb"
WORKSPACE_PATH = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b"
SIGNAL_BASE = f"{WORKSPACE_PATH}/traces/SPY_1m/signals"
RESULTS_DIR = Path(WORKSPACE_PATH) / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Output file
OUTPUT_FILE = RESULTS_DIR / f"all_strategies_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

def construct_signal_path(strategy_type, strategy_name):
    """Construct signal file path from strategy type and name"""
    # Map strategy types to folder names
    folder_map = {
        'sma_crossover': 'sma_crossover_grid',
        'ema_crossover': 'ema_crossover_grid',
        'macd_crossover': 'macd_crossover_grid',
        'rsi_threshold': 'rsi_threshold_grid',
        'rsi_bands': 'rsi_bands_grid',
        'bollinger_breakout': 'bollinger_breakout_grid',
        'stochastic_rsi': 'stochastic_rsi_grid',
        'ultimate_oscillator': 'ultimate_oscillator_grid',
        'adx_trend_strength': 'adx_trend_strength_trend_grid',
        'cci_bands': 'cci_bands_grid',
        'cci_threshold': 'cci_threshold_grid',
        'momentum_breakout': 'momentum_breakout_grid',
        'roc_trend': 'roc_trend_grid',
        'roc_threshold': 'roc_threshold_grid',
        'trendline_bounces': 'trendline_bounces_slope_grid',
        'trendline_breaks': 'trendline_breaks_slope_grid',
        'pivot_channel_bounces': 'pivot_channel_bounces_grid',
        'pivot_channel_breaks': 'pivot_channel_breaks_grid',
        'support_resistance_breakout': 'support_resistance_breakout_grid',
        'elder_ray': 'elder_ray_grid',
        'atr_channel_breakout': 'atr_channel_breakout_grid',
        'mfi_bands': 'mfi_bands_grid',
        'williams_r': 'williams_r_grid',
        'dema_crossover': 'dema_crossover_grid',
        'dema_sma_crossover': 'dema_sma_crossover_grid',
        'ema_sma_crossover': 'ema_sma_crossover_grid',
        'tema_sma_crossover': 'tema_sma_crossover_grid',
        'stochastic_crossover': 'stochastic_crossover_grid',
        'keltner_breakout': 'keltner_breakout_grid',
        'aroon_crossover': 'aroon_crossover_grid',
        'aroon_oscillator': 'aroon_oscillator_grid',
        'donchian_breakout': 'donchian_breakout_grid',
        'vortex_crossover': 'vortex_crossover_grid',
        'vortex_trend': 'vortex_trend_grid',
        'parabolic_sar': 'parabolic_sar_grid',
        'linear_regression_slope': 'linear_regression_slope_grid',
        'ichimoku_cloud_position': 'ichimoku_cloud_position_grid',
        'vwap_deviation': 'vwap_deviation_grid',
        'supertrend': 'supertrend_grid',
        'chaikin_money_flow': 'chaikin_money_flow_grid',
        'accumulation_distribution': 'accumulation_distribution_threshold_grid',
        'obv_trend': 'obv_trend_threshold_grid',
        'price_action_swing': 'price_action_swing_grid',
        'pivot_points': 'pivot_points_bounces_grid',
        'fibonacci_retracement': 'fibonacci_retracement_bounces_grid'
    }
    
    folder = folder_map.get(strategy_type, strategy_type + '_grid')
    filename = f"SPY_{strategy_name}.parquet"
    return f"{SIGNAL_BASE}/{folder}/{filename}"

def analyze_strategy(strategy_id, strategy_name, strategy_type, signal_path):
    """Analyze a single strategy using DuckDB"""
    
    # Check if file exists
    if not Path(signal_path).exists():
        return None
    
    query = f"""
    -- Analyze strategy: {strategy_name}
    SET VARIABLE strategy_path = '{signal_path}';
    
    WITH 
    -- Load regime timeline (already in DB)
    regime_data AS (
        SELECT * FROM regime_timeline
    ),
    -- Load market prices (already in DB)
    price_data AS (
        SELECT * FROM market_prices
    ),
    -- Get strategy signals
    strategy_signals AS (
        SELECT 
            ts::timestamp as signal_time,
            val as signal_value,
            LAG(val) OVER (ORDER BY ts::timestamp) as prev_signal,
            LEAD(ts::timestamp) OVER (ORDER BY ts::timestamp) as next_signal_time
        FROM read_parquet('{signal_path}')
        WHERE ts::timestamp >= '2024-03-26 00:00:00'
          AND ts::timestamp <= '2025-01-17 20:00:00'
    ),
    -- Identify position changes
    position_changes AS (
        SELECT 
            signal_time,
            signal_value as position
        FROM strategy_signals
        WHERE prev_signal IS NULL OR prev_signal != signal_value
    ),
    -- Create position timeline
    position_timeline AS (
        SELECT 
            p.timestamp_est,
            p.close,
            r.current_regime,
            LAST_VALUE(pc.position IGNORE NULLS) OVER (
                ORDER BY p.timestamp_est 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as position
        FROM price_data p
        LEFT JOIN regime_data r ON p.timestamp_est = r.timestamp_est
        LEFT JOIN position_changes pc ON p.timestamp_est = pc.signal_time
    ),
    -- Calculate minute returns
    minute_returns AS (
        SELECT 
            DATE_TRUNC('day', timestamp_est) as trading_day,
            current_regime,
            CASE 
                WHEN position = 1 THEN (close - LAG(close) OVER (ORDER BY timestamp_est)) / LAG(close) OVER (ORDER BY timestamp_est)
                WHEN position = -1 THEN (LAG(close) OVER (ORDER BY timestamp_est) - close) / LAG(close) OVER (ORDER BY timestamp_est)
                ELSE 0
            END as minute_return
        FROM position_timeline
        WHERE position IS NOT NULL
    ),
    -- Aggregate to daily returns
    daily_returns AS (
        SELECT 
            trading_day,
            current_regime,
            SUM(minute_return) as daily_return
        FROM minute_returns
        WHERE minute_return IS NOT NULL
        GROUP BY trading_day, current_regime
    ),
    -- Calculate regime performance
    regime_performance AS (
        SELECT 
            current_regime,
            COUNT(*) as trading_days,
            AVG(daily_return) as avg_daily_return,
            STDDEV(daily_return) as daily_volatility,
            (AVG(daily_return) - 0.0) / NULLIF(STDDEV(daily_return), 0) * SQRT(252) as annualized_sharpe,
            SUM(daily_return) as total_return,
            SUM(CASE WHEN daily_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_days_pct
        FROM daily_returns
        GROUP BY current_regime
        HAVING COUNT(*) >= 5
    )
    SELECT 
        '{strategy_id}' as strategy_id,
        '{strategy_name}' as strategy_name,
        '{strategy_type}' as strategy_type,
        current_regime,
        trading_days,
        ROUND(avg_daily_return * 100, 4) as avg_daily_return_pct,
        ROUND(daily_volatility * 100, 4) as daily_volatility_pct,
        ROUND(annualized_sharpe, 3) as annualized_sharpe_ratio,
        ROUND(total_return * 100, 2) as total_return_pct,
        ROUND(win_days_pct, 1) as win_days_pct
    FROM regime_performance;
    """
    
    try:
        result = subprocess.run(
            ['duckdb', ANALYTICS_DB, '-csv', '-noheader'],
            input=query,
            text=True,
            capture_output=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')
        
    except Exception as e:
        print(f"Error analyzing {strategy_name}: {str(e)}")
    
    return None

def main():
    print(f"Starting sequential analysis of all strategies")
    print(f"Output file: {OUTPUT_FILE}")
    
    # First, ensure regime tables exist
    print("\nSetting up regime and market tables...")
    setup_query = """
    -- Check if tables exist, create if not
    CREATE TABLE IF NOT EXISTS regime_timeline AS
    WITH 
    regime_sparse AS (
        SELECT 
            ts::timestamp as regime_time,
            val as regime_state
        FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet')
        WHERE ts::timestamp >= '2024-03-26 00:00:00'
          AND ts::timestamp <= '2025-01-17 20:00:00'
    ),
    market_times AS (
        SELECT DISTINCT
            timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est
        FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
        WHERE timestamp >= TIMESTAMP '2024-03-26 00:00:00' - INTERVAL 4 HOUR
          AND timestamp <= TIMESTAMP '2025-01-17 20:00:00' + INTERVAL 4 HOUR
    )
    SELECT 
        mt.timestamp_est,
        LAST_VALUE(rs.regime_state IGNORE NULLS) OVER (
            ORDER BY mt.timestamp_est 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as current_regime
    FROM market_times mt
    LEFT JOIN regime_sparse rs ON mt.timestamp_est = rs.regime_time;
    
    CREATE TABLE IF NOT EXISTS market_prices AS
    SELECT 
        timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est,
        close
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE timestamp >= TIMESTAMP '2024-03-26 00:00:00' - INTERVAL 4 HOUR
      AND timestamp <= TIMESTAMP '2025-01-17 20:00:00' + INTERVAL 4 HOUR;
    
    SELECT 'Setup complete' as status;
    """
    
    subprocess.run(['duckdb', ANALYTICS_DB], input=setup_query, text=True)
    
    # Get all strategies
    print("\nLoading strategies...")
    strategies_query = """
    SELECT 
        strategy_id,
        strategy_name,
        strategy_type
    FROM analytics.strategies
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
    strategies = []
    lines = result.stdout.strip().split('\n')[1:]  # Skip header
    for line in lines:
        if line:
            parts = line.split(',')
            if len(parts) >= 3:
                strategies.append({
                    'strategy_id': parts[0],
                    'strategy_name': parts[1],
                    'strategy_type': parts[2]
                })
    
    print(f"Found {len(strategies)} strategies")
    
    # Write CSV header
    with open(OUTPUT_FILE, 'w') as f:
        f.write("strategy_id,strategy_name,strategy_type,current_regime,trading_days,avg_daily_return_pct,daily_volatility_pct,annualized_sharpe_ratio,total_return_pct,win_days_pct\n")
    
    # Analyze each strategy
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, strategy in enumerate(strategies):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(strategies) - i) / rate if rate > 0 else 0
            print(f"\nProgress: {i}/{len(strategies)} strategies ({i/len(strategies)*100:.1f}%) - {successful} successful")
            print(f"Rate: {rate:.1f} strategies/sec - ETA: {eta/60:.1f} minutes")
        
        # Construct signal path
        signal_path = construct_signal_path(strategy['strategy_type'], strategy['strategy_name'])
        
        # Analyze strategy
        results = analyze_strategy(
            strategy['strategy_id'],
            strategy['strategy_name'],
            strategy['strategy_type'],
            signal_path
        )
        
        if results:
            with open(OUTPUT_FILE, 'a') as f:
                for row in results:
                    f.write(row + '\n')
            successful += 1
        else:
            failed += 1
    
    elapsed = time.time() - start_time
    print(f"\n\nAnalysis complete!")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {OUTPUT_FILE}")
    
    # Generate summary
    print("\n=== GENERATING SUMMARY ===")
    summary_query = f"""
    CREATE TABLE results AS SELECT * FROM read_csv_auto('{OUTPUT_FILE}');
    
    -- Top 20 strategies
    SELECT 'TOP 20 STRATEGIES BY SHARPE' as section;
    SELECT 
        strategy_name,
        current_regime,
        annualized_sharpe_ratio,
        total_return_pct,
        win_days_pct
    FROM results
    ORDER BY annualized_sharpe_ratio DESC
    LIMIT 20;
    
    -- Best per regime
    SELECT 'BEST 5 PER REGIME' as section;
    WITH ranked AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY current_regime ORDER BY annualized_sharpe_ratio DESC) as rank
        FROM results
        WHERE trading_days >= 20
    )
    SELECT current_regime, strategy_name, strategy_type, annualized_sharpe_ratio, total_return_pct
    FROM ranked
    WHERE rank <= 5
    ORDER BY current_regime, rank;
    """
    
    subprocess.run(['duckdb', ANALYTICS_DB], input=summary_query, text=True)

if __name__ == "__main__":
    main()