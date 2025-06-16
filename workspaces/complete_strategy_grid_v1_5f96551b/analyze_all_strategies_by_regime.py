#!/usr/bin/env python3
"""
Comprehensive analysis of all strategies by regime
Processes 1,235 strategies across 10 months of data
"""

import duckdb
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# Configuration
ANALYTICS_DB = "analytics.duckdb"
MARKET_DATA_PATH = "/Users/daws/ADMF-PC/data/SPY_1m.parquet"
CLASSIFIER_PATH = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_05_65_40.parquet"
SIGNAL_BASE_PATH = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/signals"

# Analysis period
START_DATE = "2024-03-26 00:00:00"
END_DATE = "2025-01-17 20:00:00"

def get_strategies(conn, strategy_type=None, limit=None):
    """Get list of strategies to analyze"""
    query = """
    SELECT 
        strategy_id,
        strategy_type,
        strategy_name,
        signal_file_path
    FROM analytics.strategies
    """
    if strategy_type:
        query += f" WHERE strategy_type = '{strategy_type}'"
    if limit:
        query += f" LIMIT {limit}"
    
    return conn.execute(query).df()

def build_regime_timeline(conn):
    """Build forward-filled regime timeline"""
    print("Building regime timeline...")
    
    query = f"""
    WITH 
    -- Get sparse regime data
    regime_sparse AS (
        SELECT 
            ts::timestamp as regime_time,
            val as regime_state
        FROM read_parquet('{CLASSIFIER_PATH}')
        WHERE ts::timestamp >= '{START_DATE}'
          AND ts::timestamp <= '{END_DATE}'
    ),
    -- Get market timestamps
    market_times AS (
        SELECT DISTINCT
            timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est
        FROM read_parquet('{MARKET_DATA_PATH}')
        WHERE timestamp >= '{START_DATE}'::timestamp with time zone - INTERVAL 4 HOUR
          AND timestamp <= '{END_DATE}'::timestamp with time zone
    ),
    -- Forward-fill regimes
    regime_timeline AS (
        SELECT 
            mt.timestamp_est,
            LAST_VALUE(rs.regime_state IGNORE NULLS) OVER (
                ORDER BY mt.timestamp_est 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as current_regime
        FROM market_times mt
        LEFT JOIN regime_sparse rs ON mt.timestamp_est = rs.regime_time
    )
    SELECT * FROM regime_timeline
    WHERE current_regime IS NOT NULL
    """
    
    return conn.execute(query).df()

def get_market_prices(conn):
    """Get market price data with timezone adjustment"""
    print("Loading market data...")
    
    query = f"""
    SELECT 
        timestamp::timestamp + INTERVAL 4 HOUR as timestamp_est,
        close
    FROM read_parquet('{MARKET_DATA_PATH}')
    WHERE timestamp >= '{START_DATE}'::timestamp with time zone - INTERVAL 4 HOUR
      AND timestamp <= '{END_DATE}'::timestamp with time zone
    """
    
    return conn.execute(query).df()

def analyze_strategy_performance(conn, strategy_row, regime_df, price_df):
    """Analyze single strategy performance by regime"""
    strategy_id = strategy_row['strategy_id']
    signal_path = strategy_row['signal_file_path']
    
    # Build proper file path
    if not signal_path:
        # Construct path from strategy info
        strategy_type = strategy_row['strategy_type']
        strategy_name = strategy_row['strategy_name']
        
        # Map strategy type to folder name
        folder_map = {
            'macd_crossover': 'macd_crossover_grid',
            'ema_crossover': 'ema_crossover_grid',
            'rsi_threshold': 'rsi_threshold_grid',
            'bollinger_breakout': 'bollinger_breakout_grid',
            'ultimate_oscillator': 'ultimate_oscillator_grid',
            'rsi_bands': 'rsi_bands_grid',
            'stochastic_rsi': 'stochastic_rsi_grid',
            'adx_trend_strength': 'adx_trend_strength_trend_grid',
            'cci_bands': 'cci_bands_grid',
            'cci_threshold': 'cci_threshold_grid'
        }
        
        if strategy_type in folder_map:
            folder = folder_map[strategy_type]
            filename = f"SPY_{strategy_name}.parquet"
            signal_path = f"{SIGNAL_BASE_PATH}/{folder}/{filename}"
        else:
            return None
    
    # Check if file exists
    if not Path(signal_path).exists():
        return None
    
    try:
        # Get signals
        signal_query = f"""
        SELECT 
            ts::timestamp as signal_time,
            val as signal_value,
            LAG(val) OVER (ORDER BY ts::timestamp) as prev_signal,
            LEAD(ts::timestamp) OVER (ORDER BY ts::timestamp) as next_signal_time
        FROM read_parquet('{signal_path}')
        WHERE ts::timestamp >= '{START_DATE}'
          AND ts::timestamp <= '{END_DATE}'
        """
        
        signals_df = conn.execute(signal_query).df()
        
        # Find trades (signal changes)
        trades_df = signals_df[
            (signals_df['prev_signal'].notna()) & 
            (signals_df['prev_signal'] != signals_df['signal_value']) &
            (signals_df['next_signal_time'].notna())
        ].copy()
        
        if len(trades_df) == 0:
            return None
        
        # Merge with regime data
        trades_df = pd.merge_asof(
            trades_df.sort_values('signal_time'),
            regime_df.sort_values('timestamp_est'),
            left_on='signal_time',
            right_on='timestamp_est',
            direction='backward'
        )
        
        # Merge with entry prices
        trades_df = pd.merge_asof(
            trades_df,
            price_df[['timestamp_est', 'close']].rename(columns={'close': 'entry_price'}),
            left_on='signal_time',
            right_on='timestamp_est',
            direction='nearest'
        )
        
        # Merge with exit prices
        trades_df = pd.merge_asof(
            trades_df,
            price_df[['timestamp_est', 'close']].rename(columns={'close': 'exit_price'}),
            left_on='next_signal_time',
            right_on='timestamp_est',
            direction='nearest'
        )
        
        # Calculate returns
        trades_df['trade_return'] = np.where(
            trades_df['signal_value'] == 1,
            (trades_df['exit_price'] - trades_df['entry_price']) / trades_df['entry_price'],
            np.where(
                trades_df['signal_value'] == -1,
                (trades_df['entry_price'] - trades_df['exit_price']) / trades_df['entry_price'],
                0
            )
        )
        
        # Group by regime
        regime_performance = trades_df.groupby('current_regime').agg({
            'trade_return': ['count', 'mean', 'sum', 'std'],
            'signal_value': lambda x: (x > 0).sum() / len(x)  # Long trade percentage
        }).round(4)
        
        regime_performance.columns = ['trade_count', 'avg_return', 'total_return', 'return_std', 'long_pct']
        regime_performance['win_rate'] = trades_df.groupby('current_regime')['trade_return'].apply(lambda x: (x > 0).sum() / len(x))
        regime_performance['sharpe'] = regime_performance['avg_return'] / regime_performance['return_std'].fillna(1)
        regime_performance['net_return'] = regime_performance['total_return'] - regime_performance['trade_count'] * 0.0005  # Transaction costs
        
        regime_performance['strategy_id'] = strategy_id
        regime_performance['strategy_type'] = strategy_row['strategy_type']
        regime_performance['strategy_name'] = strategy_row['strategy_name']
        
        return regime_performance.reset_index()
        
    except Exception as e:
        print(f"Error analyzing {strategy_id}: {str(e)}")
        return None

def main():
    """Main analysis function"""
    conn = duckdb.connect(ANALYTICS_DB)
    
    # Get regime timeline and market data
    regime_df = build_regime_timeline(conn)
    price_df = get_market_prices(conn)
    
    print(f"Loaded {len(regime_df)} regime data points and {len(price_df)} price points")
    
    # Get all strategies
    strategies_df = get_strategies(conn)
    print(f"Found {len(strategies_df)} strategies to analyze")
    
    # Process strategies in batches by type
    results = []
    
    for strategy_type in strategies_df['strategy_type'].unique():
        print(f"\nProcessing {strategy_type} strategies...")
        type_strategies = strategies_df[strategies_df['strategy_type'] == strategy_type]
        
        # Sample first 10 of each type for testing
        for idx, strategy in type_strategies.head(10).iterrows():
            result = analyze_strategy_performance(conn, strategy, regime_df, price_df)
            if result is not None:
                results.append(result)
                
        print(f"Completed {len(type_strategies.head(10))} {strategy_type} strategies")
    
    # Combine results
    if results:
        all_results = pd.concat(results, ignore_index=True)
        
        # Save detailed results
        all_results.to_csv('strategy_regime_performance_detailed.csv', index=False)
        
        # Create summary by regime
        regime_summary = all_results.groupby('current_regime').agg({
            'strategy_id': 'count',
            'trade_count': 'sum',
            'avg_return': 'mean',
            'net_return': 'mean',
            'win_rate': 'mean',
            'sharpe': 'mean'
        }).round(4)
        
        regime_summary.columns = ['strategies_analyzed', 'total_trades', 'avg_return_pct', 'avg_net_return', 'avg_win_rate', 'avg_sharpe']
        regime_summary['avg_return_pct'] *= 100
        regime_summary['avg_net_return'] *= 100
        regime_summary['avg_win_rate'] *= 100
        
        print("\n=== REGIME PERFORMANCE SUMMARY ===")
        print(regime_summary)
        
        # Create summary by strategy type and regime
        type_regime_summary = all_results.groupby(['strategy_type', 'current_regime']).agg({
            'net_return': ['mean', 'std', 'count'],
            'sharpe': 'mean',
            'win_rate': 'mean'
        }).round(4)
        
        print("\n=== TOP PERFORMING STRATEGY TYPES BY REGIME ===")
        for regime in all_results['current_regime'].unique():
            print(f"\n{regime.upper()} REGIME:")
            regime_data = all_results[all_results['current_regime'] == regime]
            top_strategies = regime_data.nlargest(5, 'net_return')[['strategy_name', 'trade_count', 'net_return', 'sharpe', 'win_rate']]
            print(top_strategies)
        
        # Save summaries
        regime_summary.to_csv('regime_performance_summary.csv')
        type_regime_summary.to_csv('strategy_type_regime_summary.csv')
        
        print(f"\nAnalysis complete. Results saved to CSV files.")
        print(f"Total strategies analyzed: {all_results['strategy_id'].nunique()}")
        print(f"Total trades analyzed: {all_results['trade_count'].sum()}")
    
    conn.close()

if __name__ == "__main__":
    main()