#!/usr/bin/env python3
"""
Comprehensive analysis of all 1,235 strategies by regime
Processes entire 10-month training set
"""

import subprocess
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import time

# Configuration
ANALYTICS_DB = "analytics.duckdb"
MARKET_DATA_PATH = "/Users/daws/ADMF-PC/data/SPY_1m.parquet"
CLASSIFIER_PATH = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet"
SIGNAL_BASE_PATH = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals"

# Analysis period
START_DATE = "2024-03-26 00:00:00"
END_DATE = "2025-01-17 20:00:00"

def run_duckdb_query(query):
    """Execute DuckDB query and return results"""
    # Use command line interface since duckdb module not installed
    cmd = ['duckdb', ANALYTICS_DB, '-csv', '-c', query]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    # Parse CSV output
    if result.stdout:
        from io import StringIO
        return pd.read_csv(StringIO(result.stdout))
    return None

def get_strategies():
    """Get list of all strategies from database"""
    query = """
    SELECT 
        strategy_id,
        strategy_type,
        strategy_name,
        signal_file_path
    FROM analytics.strategies
    """
    return run_duckdb_query(query)

def analyze_strategy_batch(strategy_type, limit=None):
    """Analyze a batch of strategies of the same type"""
    print(f"\nAnalyzing {strategy_type} strategies...")
    
    # Build the analysis query
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    query = f"""
    WITH 
    -- Get strategies to analyze
    strategies_batch AS (
        SELECT 
            strategy_id,
            strategy_name,
            signal_file_path
        FROM analytics.strategies
        WHERE strategy_type = '{strategy_type}'
        {limit_clause}
    ),
    -- Build regime timeline once (forward-filled)
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
        WHERE timestamp >= '{START_DATE}'-4H
          AND timestamp <= '{END_DATE}'+4H
    ),
    regime_timeline AS (
        SELECT 
            mt.timestamp_est,
            LAST_VALUE(rs.regime_state IGNORE NULLS) OVER (
                ORDER BY mt.timestamp_est 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as current_regime
        FROM market_times mt
        LEFT JOIN regime_sparse rs ON mt.timestamp_est = rs.regime_time
    ),
    -- Get market prices once
    market_prices AS (
        SELECT 
            timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est,
            close
        FROM read_parquet('{MARKET_DATA_PATH}')
        WHERE timestamp >= '{START_DATE}'-4H
          AND timestamp <= '{END_DATE}'+4H
    ),
    -- Analyze each strategy
    strategy_trades AS (
        SELECT 
            s.strategy_id,
            s.strategy_name,
            sig.ts::timestamp as signal_time,
            sig.val as signal_value,
            LAG(sig.val) OVER (PARTITION BY s.strategy_id ORDER BY sig.ts) as prev_signal,
            LEAD(sig.ts::timestamp) OVER (PARTITION BY s.strategy_id ORDER BY sig.ts) as next_signal_time
        FROM strategies_batch s,
        LATERAL read_parquet(s.signal_file_path) sig
        WHERE sig.ts::timestamp >= '{START_DATE}'
          AND sig.ts::timestamp <= '{END_DATE}'
    ),
    valid_trades AS (
        SELECT 
            strategy_id,
            strategy_name,
            signal_time,
            signal_value,
            next_signal_time
        FROM strategy_trades
        WHERE prev_signal IS NOT NULL
          AND prev_signal != signal_value
          AND next_signal_time IS NOT NULL
    ),
    trades_with_data AS (
        SELECT 
            t.*,
            rt.current_regime as entry_regime,
            mp1.close as entry_price,
            mp2.close as exit_price,
            EXTRACT(EPOCH FROM (t.next_signal_time - t.signal_time)) / 60.0 as duration_minutes,
            CASE 
                WHEN t.signal_value = 1 THEN (mp2.close - mp1.close) / mp1.close
                WHEN t.signal_value = -1 THEN (mp1.close - mp2.close) / mp1.close
            END as trade_return
        FROM valid_trades t
        LEFT JOIN regime_timeline rt ON t.signal_time = rt.timestamp_est
        LEFT JOIN market_prices mp1 ON t.signal_time = mp1.timestamp_est
        LEFT JOIN market_prices mp2 ON t.next_signal_time = mp2.timestamp_est
        WHERE rt.current_regime IS NOT NULL
          AND mp1.close IS NOT NULL
          AND mp2.close IS NOT NULL
    )
    SELECT 
        strategy_id,
        strategy_name,
        entry_regime,
        COUNT(*) as trade_count,
        ROUND(AVG(trade_return) * 100, 3) as avg_return_pct,
        ROUND(SUM(trade_return) * 100, 3) as total_return_pct,
        ROUND(SUM(trade_return - 0.0005) * 100, 3) as net_return_pct,
        ROUND(AVG(trade_return) / NULLIF(STDDEV(trade_return), 0), 3) as sharpe_ratio,
        ROUND(SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate,
        ROUND(AVG(duration_minutes), 1) as avg_duration_min
    FROM trades_with_data
    GROUP BY strategy_id, strategy_name, entry_regime
    ORDER BY strategy_id, entry_regime
    """
    
    return run_duckdb_query(query)

def main():
    """Main analysis function"""
    print("Starting comprehensive strategy regime analysis...")
    
    # Get all strategies
    strategies_df = get_strategies()
    if strategies_df is None:
        print("Failed to get strategies")
        return
    
    print(f"Found {len(strategies_df)} strategies to analyze")
    
    # Get strategy types and counts
    strategy_types = strategies_df.groupby('strategy_type').size().sort_values(ascending=False)
    print("\nStrategy types:")
    for stype, count in strategy_types.items():
        print(f"  {stype}: {count} strategies")
    
    # Process each strategy type
    all_results = []
    
    for stype in strategy_types.index:
        # Sample first few strategies of each type to test
        batch_size = min(10, strategy_types[stype])  # Analyze up to 10 per type for initial test
        
        result = analyze_strategy_batch(stype, limit=batch_size)
        if result is not None and len(result) > 0:
            all_results.append(result)
            print(f"  Completed {len(result)} regime-strategy combinations")
        else:
            print(f"  No results for {stype}")
        
        # Sleep briefly to avoid overwhelming the system
        time.sleep(0.5)
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save detailed results
        combined_df.to_csv('all_strategies_regime_performance.csv', index=False)
        print(f"\nSaved detailed results to all_strategies_regime_performance.csv")
        
        # Create summary by regime
        regime_summary = combined_df.groupby('entry_regime').agg({
            'strategy_id': 'nunique',
            'trade_count': 'sum',
            'avg_return_pct': 'mean',
            'net_return_pct': 'mean',
            'win_rate': 'mean',
            'sharpe_ratio': 'mean'
        }).round(3)
        
        regime_summary.columns = ['strategies_analyzed', 'total_trades', 'avg_return_pct', 'avg_net_return_pct', 'avg_win_rate', 'avg_sharpe']
        
        print("\n=== REGIME PERFORMANCE SUMMARY ===")
        print(regime_summary)
        regime_summary.to_csv('regime_performance_summary.csv')
        
        # Find top strategies per regime
        print("\n=== TOP 5 STRATEGIES PER REGIME ===")
        for regime in combined_df['entry_regime'].unique():
            print(f"\n{regime.upper()} REGIME:")
            regime_data = combined_df[combined_df['entry_regime'] == regime]
            top_strategies = regime_data.nlargest(5, 'net_return_pct')[
                ['strategy_name', 'trade_count', 'net_return_pct', 'sharpe_ratio', 'win_rate']
            ]
            print(top_strategies.to_string())
        
        # Summary statistics
        print(f"\n=== ANALYSIS SUMMARY ===")
        print(f"Total strategies analyzed: {combined_df['strategy_id'].nunique()}")
        print(f"Total trades analyzed: {combined_df['trade_count'].sum():,}")
        print(f"Average trades per strategy: {combined_df.groupby('strategy_id')['trade_count'].sum().mean():.1f}")
        
        # Strategy type performance
        type_performance = combined_df.groupby('strategy_name').apply(
            lambda x: pd.Series({
                'total_trades': x['trade_count'].sum(),
                'avg_net_return': x['net_return_pct'].mean(),
                'best_regime': x.loc[x['net_return_pct'].idxmax(), 'entry_regime'] if len(x) > 0 else None,
                'best_regime_return': x['net_return_pct'].max()
            })
        ).round(3)
        
        type_performance = type_performance.sort_values('avg_net_return', ascending=False)
        print("\n=== TOP 10 STRATEGIES BY AVERAGE NET RETURN ===")
        print(type_performance.head(10))
        
        type_performance.to_csv('strategy_type_performance_summary.csv')
        
    else:
        print("No results obtained")

if __name__ == "__main__":
    main()