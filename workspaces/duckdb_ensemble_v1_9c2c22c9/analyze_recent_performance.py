#!/usr/bin/env python3
"""
Analyze the performance of adaptive ensemble strategies for the last 12,000 bars.
This represents approximately 12-13 trading days of 1-minute data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def load_signal_data(file_path):
    """Load signal data from parquet file."""
    df = pd.read_parquet(file_path)
    return df

def filter_last_n_bars(df, n=12000):
    """Filter dataframe to include only the last n bars."""
    # Convert timestamp to datetime if it's not already
    df['ts'] = pd.to_datetime(df['ts'])
    
    # Get the maximum index value (which represents bar index)
    max_idx = df['idx'].max()
    cutoff_idx = max_idx - n
    return df[df['idx'] > cutoff_idx].copy()

def reconstruct_full_signals(df):
    """Reconstruct the full signal series from compressed format."""
    # Create a complete range of indices
    min_idx = df['idx'].min()
    max_idx = df['idx'].max()
    full_index = pd.DataFrame({'idx': range(min_idx, max_idx + 1)})
    
    # Merge with signal data
    full_df = full_index.merge(df, on='idx', how='left')
    
    # Forward fill the signal values (val column contains the signal)
    full_df['val'] = full_df['val'].ffill()
    
    # Fill any remaining NaN at the beginning with 0
    full_df['val'] = full_df['val'].fillna(0)
    
    # Forward fill price data as well
    full_df['px'] = full_df['px'].ffill()
    
    return full_df

def calculate_returns(signals_df):
    """Calculate returns based on signals."""
    # Calculate price returns using the close column
    signals_df['price_return'] = signals_df['close'].pct_change()
    
    # Calculate strategy returns (signal from previous bar * current return)
    signals_df['strategy_return'] = signals_df['val'].shift(1) * signals_df['price_return']
    
    # Fill NaN values
    signals_df['price_return'] = signals_df['price_return'].fillna(0)
    signals_df['strategy_return'] = signals_df['strategy_return'].fillna(0)
    
    return signals_df

def calculate_performance_metrics(returns_df):
    """Calculate performance metrics for the strategy."""
    metrics = {}
    
    # Total return
    metrics['total_return'] = (1 + returns_df['strategy_return']).cumprod().iloc[-1] - 1
    metrics['buy_hold_return'] = (1 + returns_df['price_return']).cumprod().iloc[-1] - 1
    
    # Annualization factor (252 trading days * 390 minutes per day)
    minutes_per_year = 252 * 390
    n_minutes = len(returns_df)
    annualization_factor = minutes_per_year / n_minutes
    
    # Sharpe ratio (annualized)
    if returns_df['strategy_return'].std() > 0:
        metrics['sharpe_ratio'] = (returns_df['strategy_return'].mean() * np.sqrt(annualization_factor)) / returns_df['strategy_return'].std()
    else:
        metrics['sharpe_ratio'] = 0
    
    # Maximum drawdown
    cumulative_returns = (1 + returns_df['strategy_return']).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    
    # Trading statistics
    signal_changes = returns_df['val'].diff().abs()
    metrics['num_trades'] = int(signal_changes.sum() / 2)  # Divide by 2 for round trips
    
    # Win rate
    winning_trades = returns_df[returns_df['strategy_return'] > 0]['strategy_return'].count()
    losing_trades = returns_df[returns_df['strategy_return'] < 0]['strategy_return'].count()
    total_trades_with_returns = winning_trades + losing_trades
    
    if total_trades_with_returns > 0:
        metrics['win_rate'] = winning_trades / total_trades_with_returns
    else:
        metrics['win_rate'] = 0
    
    # Average return per trade
    if metrics['num_trades'] > 0:
        metrics['avg_return_per_trade'] = metrics['total_return'] / metrics['num_trades']
    else:
        metrics['avg_return_per_trade'] = 0
    
    return metrics

def load_price_data():
    """Load price data from the original SPY_1m.csv file."""
    price_df = pd.read_csv('/Users/daws/ADMF-PC/data/SPY_1m.csv')
    
    # Convert timestamp to datetime with UTC
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)
    
    # Create bar index (0-based to match signal data)
    price_df['idx'] = range(len(price_df))
    
    # Rename Close to close for consistency
    price_df['close'] = price_df['Close']
    
    return price_df[['idx', 'timestamp', 'close']]

def main():
    """Main analysis function."""
    workspace_path = Path('/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_v1_9c2c22c9')
    
    # Define file paths
    default_path = workspace_path / 'traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet'
    custom_path = workspace_path / 'traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_custom.parquet'
    
    # Load price data
    print("Loading price data...")
    price_data = load_price_data()
    print(f"Loaded {len(price_data):,} price bars")
    
    results = {}
    
    for strategy_name, file_path in [('Default Ensemble', default_path), ('Custom Ensemble', custom_path)]:
        print(f"\nAnalyzing {strategy_name}...")
        
        # Load data
        df = load_signal_data(file_path)
        print(f"Total bars in file: {len(df):,}")
        
        # Filter to last 12,000 bars
        recent_df = filter_last_n_bars(df, n=12000)
        print(f"Bars in recent period: {len(recent_df):,}")
        
        # Reconstruct full signals
        full_signals = reconstruct_full_signals(recent_df)
        
        # Filter price data to last 12,000 bars for merging
        max_idx = price_data['idx'].max()
        recent_price_data = price_data[price_data['idx'] > (max_idx - 12000)].copy()
        
        # Merge with price data
        full_signals = full_signals.merge(recent_price_data, on='idx', how='left')
        
        # Check if we have price data in the dataframe
        if 'close' in full_signals.columns and full_signals['close'].notna().any():
            # Calculate returns
            returns_df = calculate_returns(full_signals)
            
            # Calculate metrics
            metrics = calculate_performance_metrics(returns_df)
            
            # Add additional signal statistics
            metrics['signal_changes'] = len(recent_df)
            metrics['avg_signal'] = full_signals['val'].mean()
            metrics['signal_std'] = full_signals['val'].std()
            metrics['positive_signals'] = (full_signals['val'] > 0).sum()
            metrics['negative_signals'] = (full_signals['val'] < 0).sum()
            metrics['zero_signals'] = (full_signals['val'] == 0).sum()
            
            # Add time range info
            if 'ts' in recent_df.columns:
                metrics['start_time'] = recent_df['ts'].min()
                metrics['end_time'] = recent_df['ts'].max()
        else:
            print(f"Warning: No price data found in {strategy_name} file. Calculating signal statistics only.")
            metrics = {
                'total_signals': len(recent_df),
                'unique_bar_indices': recent_df['idx'].nunique(),
                'signal_changes': len(recent_df),
                'avg_signal': full_signals['val'].mean(),
                'signal_std': full_signals['val'].std(),
                'positive_signals': (full_signals['val'] > 0).sum(),
                'negative_signals': (full_signals['val'] < 0).sum(),
                'zero_signals': (full_signals['val'] == 0).sum(),
            }
        
        results[strategy_name] = metrics
        
        # Print metrics
        print(f"\n{strategy_name} Performance Metrics (Last 12,000 bars):")
        print("-" * 50)
        for metric, value in metrics.items():
            if isinstance(value, float):
                if 'return' in metric or 'drawdown' in metric:
                    print(f"{metric}: {value:.2%}")
                else:
                    print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
    
    # Compare strategies
    if len(results) == 2:
        print("\n\nComparison Summary:")
        print("=" * 50)
        
        default_metrics = results['Default Ensemble']
        custom_metrics = results['Custom Ensemble']
        
        # Find common metrics
        common_metrics = set(default_metrics.keys()) & set(custom_metrics.keys())
        
        for metric in sorted(common_metrics):
            default_val = default_metrics[metric]
            custom_val = custom_metrics[metric]
            
            if isinstance(default_val, (int, float)) and isinstance(custom_val, (int, float)):
                diff = custom_val - default_val
                if 'return' in metric or 'drawdown' in metric or 'rate' in metric:
                    print(f"{metric}:")
                    print(f"  Default: {default_val:.2%}")
                    print(f"  Custom:  {custom_val:.2%}")
                    print(f"  Difference: {diff:.2%}")
                else:
                    print(f"{metric}:")
                    print(f"  Default: {default_val:.4f}")
                    print(f"  Custom:  {custom_val:.4f}")
                    print(f"  Difference: {diff:.4f}")
            else:
                print(f"{metric}:")
                print(f"  Default: {default_val}")
                print(f"  Custom:  {custom_val}")
            print()
    
    # Save results to CSV
    results_df = pd.DataFrame(results).T
    results_df.to_csv(workspace_path / 'recent_performance_analysis.csv')
    print(f"\nResults saved to: {workspace_path / 'recent_performance_analysis.csv'}")

if __name__ == "__main__":
    main()