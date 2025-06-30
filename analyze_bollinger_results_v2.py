#!/usr/bin/env python3
"""Analyze Bollinger Band optimization results from the latest run."""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Path to the latest Bollinger results
RESULTS_DIR = Path("/Users/daws/ADMF-PC/config/bollinger/results/20250623_062931")
TRACES_DIR = RESULTS_DIR / "traces" / "bollinger_bands"
METADATA_FILE = RESULTS_DIR / "metadata.json"

def load_metadata():
    """Load and parse the metadata file."""
    with open(METADATA_FILE, 'r') as f:
        return json.load(f)

def calculate_trade_metrics(df):
    """Calculate detailed trade metrics from signal data."""
    signals = df['val'].values
    prices = df['px'].values
    timestamps = pd.to_datetime(df['ts'])
    
    trades = []
    current_position = 0
    entry_price = None
    entry_time = None
    entry_idx = None
    
    for i in range(len(signals)):
        # Entry signal
        if signals[i] != 0 and current_position == 0:
            current_position = signals[i]
            entry_price = prices[i]
            entry_time = timestamps[i]
            entry_idx = i
        
        # Exit signal (position changes or goes to 0)
        elif current_position != 0 and (signals[i] == 0 or signals[i] == -current_position):
            exit_price = prices[i]
            exit_time = timestamps[i]
            
            # Calculate return
            if current_position == 1:  # Long
                returns = (exit_price - entry_price) / entry_price
            else:  # Short
                returns = (entry_price - exit_price) / entry_price
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': current_position,
                'returns': returns,
                'duration_minutes': (exit_time - entry_time).total_seconds() / 60
            })
            
            # Reset position
            current_position = signals[i] if signals[i] != 0 else 0
            if current_position != 0:
                entry_price = prices[i]
                entry_time = timestamps[i]
                entry_idx = i
    
    return trades

def analyze_strategy_performance(signal_file):
    """Analyze performance of a single strategy from parquet file."""
    try:
        # Load the parquet file
        df = pd.read_parquet(signal_file)
        
        if df.empty or 'val' not in df.columns:
            return None
        
        # Get signals
        signals = df['val'].values
        
        # Basic signal statistics
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        neutral_signals = (signals == 0).sum()
        
        # Calculate trades
        trades = calculate_trade_metrics(df)
        
        # Extract strategy ID from filename
        strategy_id = int(signal_file.stem.split('_')[-1])
        
        # Calculate performance metrics
        if trades:
            returns = [t['returns'] for t in trades]
            durations = [t['duration_minutes'] for t in trades]
            
            # Win rate
            winning_trades = sum(1 for r in returns if r > 0)
            win_rate = winning_trades / len(trades) if trades else 0
            
            # Average return
            avg_return = np.mean(returns) if returns else 0
            
            # Sharpe ratio (simplified - annualized assuming 252 trading days)
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = (np.mean(returns) * np.sqrt(252 * 78)) / np.std(returns)  # 78 = 5-min bars per day
            else:
                sharpe = 0
            
            result = {
                'strategy_id': strategy_id,
                'total_signals': len(df),
                'long_signals': long_signals,
                'short_signals': short_signals,
                'neutral_signals': neutral_signals,
                'signal_ratio': (long_signals + short_signals) / len(df) if len(df) > 0 else 0,
                'num_trades': len(trades),
                'win_rate': win_rate,
                'avg_return_pct': avg_return * 100,
                'total_return_pct': sum(returns) * 100,
                'sharpe_ratio': sharpe,
                'avg_duration_minutes': np.mean(durations) if durations else 0,
                'max_duration_minutes': max(durations) if durations else 0,
                'min_duration_minutes': min(durations) if durations else 0
            }
        else:
            result = {
                'strategy_id': strategy_id,
                'total_signals': len(df),
                'long_signals': long_signals,
                'short_signals': short_signals,
                'neutral_signals': neutral_signals,
                'signal_ratio': (long_signals + short_signals) / len(df) if len(df) > 0 else 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_return_pct': 0,
                'total_return_pct': 0,
                'sharpe_ratio': 0,
                'avg_duration_minutes': 0,
                'max_duration_minutes': 0,
                'min_duration_minutes': 0
            }
        
        return result
        
    except Exception as e:
        print(f"Error processing {signal_file}: {e}")
        return None

def main():
    print("Bollinger Band Optimization Results Analysis")
    print("=" * 60)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load metadata
    print("Loading metadata...")
    metadata = load_metadata()
    
    # Extract configuration info
    if 'config' in metadata:
        config = metadata['config']
        print("\nConfiguration Summary:")
        print(f"- Mode: {config.get('mode', 'N/A')}")
        if 'data_config' in config:
            print(f"- Symbols: {config['data_config'].get('symbols', 'N/A')}")
            print(f"- Date range: {config['data_config'].get('start_date', 'N/A')} to {config['data_config'].get('end_date', 'N/A')}")
    
    # Analyze all strategy files
    print("\nAnalyzing strategy performance...")
    parquet_files = sorted(list(TRACES_DIR.glob("*.parquet")))[:100]  # Analyze first 100 for speed
    print(f"Analyzing first {len(parquet_files)} strategies...")
    
    results = []
    for i, file in enumerate(parquet_files):
        if i % 10 == 0:
            print(f"Processing strategy {i}/{len(parquet_files)}...")
        
        result = analyze_strategy_performance(file)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results found!")
        return
    
    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(results)
    
    print(f"\nAnalyzed {len(df_results)} strategies successfully")
    print("\nPerformance Summary:")
    print("-" * 60)
    
    # Overall statistics
    print(f"Average trades per strategy: {df_results['num_trades'].mean():.2f}")
    print(f"Max trades: {df_results['num_trades'].max()}")
    print(f"Min trades: {df_results['num_trades'].min()}")
    print(f"Strategies with trades: {(df_results['num_trades'] > 0).sum()} ({(df_results['num_trades'] > 0).sum() / len(df_results) * 100:.1f}%)")
    
    # Filter to strategies with trades
    df_with_trades = df_results[df_results['num_trades'] > 0]
    
    if len(df_with_trades) > 0:
        print(f"\nFor strategies with trades:")
        print(f"  Average win rate: {df_with_trades['win_rate'].mean():.2%}")
        print(f"  Average return per trade: {df_with_trades['avg_return_pct'].mean():.3f}%")
        print(f"  Average total return: {df_with_trades['total_return_pct'].mean():.2f}%")
        print(f"  Average Sharpe ratio: {df_with_trades['sharpe_ratio'].mean():.2f}")
        print(f"  Average trade duration: {df_with_trades['avg_duration_minutes'].mean():.1f} minutes")
    
    # Top performing strategies
    print("\nTop 10 Strategies by Total Return:")
    print("-" * 80)
    print(f"{'ID':>6} {'Trades':>8} {'Win Rate':>10} {'Avg Ret':>10} {'Total Ret':>12} {'Sharpe':>8} {'Avg Dur':>10}")
    print("-" * 80)
    
    top_by_return = df_results.nlargest(10, 'total_return_pct')
    for _, row in top_by_return.iterrows():
        print(f"{row['strategy_id']:>6} {row['num_trades']:>8} {row['win_rate']:>10.1%} "
              f"{row['avg_return_pct']:>10.3f}% {row['total_return_pct']:>12.2f}% "
              f"{row['sharpe_ratio']:>8.2f} {row['avg_duration_minutes']:>10.1f}")
    
    print("\nTop 10 Strategies by Sharpe Ratio (min 10 trades):")
    print("-" * 80)
    df_min_trades = df_results[df_results['num_trades'] >= 10]
    if len(df_min_trades) > 0:
        top_by_sharpe = df_min_trades.nlargest(10, 'sharpe_ratio')
        for _, row in top_by_sharpe.iterrows():
            print(f"{row['strategy_id']:>6} {row['num_trades']:>8} {row['win_rate']:>10.1%} "
                  f"{row['avg_return_pct']:>10.3f}% {row['total_return_pct']:>12.2f}% "
                  f"{row['sharpe_ratio']:>8.2f} {row['avg_duration_minutes']:>10.1f}")
    
    # Signal distribution
    print("\nSignal Distribution Analysis:")
    print(f"  Average long signals: {df_results['long_signals'].mean():.1f}")
    print(f"  Average short signals: {df_results['short_signals'].mean():.1f}")
    print(f"  Average signal ratio: {df_results['signal_ratio'].mean():.3f}")
    
    # Save results
    output_file = "bollinger_performance_analysis.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Try to extract and match parameters
    print("\nAttempting to extract strategy parameters from metadata...")
    if 'strategies' in metadata:
        # Create parameter lookup
        param_lookup = {}
        for strat_key, strat_info in metadata['strategies'].items():
            if 'config' in strat_info:
                strat_id = int(strat_key.split('_')[-1])
                param_lookup[strat_id] = strat_info['config']
        
        # Add parameters to results
        for i, row in df_results.iterrows():
            if row['strategy_id'] in param_lookup:
                params = param_lookup[row['strategy_id']]
                df_results.at[i, 'period'] = params.get('period', 'N/A')
                df_results.at[i, 'std_dev'] = params.get('std_dev', 'N/A')
                df_results.at[i, 'exit_at_band'] = params.get('exit_at_band', 'N/A')
                df_results.at[i, 'exit_at_mid'] = params.get('exit_at_mid', 'N/A')
        
        # Show best parameters
        if 'period' in df_results.columns:
            print("\nBest Parameter Combinations (by total return):")
            best_params = df_results.nlargest(5, 'total_return_pct')
            for _, row in best_params.iterrows():
                print(f"\nStrategy {row['strategy_id']}:")
                print(f"  Parameters: Period={row.get('period', 'N/A')}, StdDev={row.get('std_dev', 'N/A')}")
                print(f"  Exit: Band={row.get('exit_at_band', 'N/A')}, Mid={row.get('exit_at_mid', 'N/A')}")
                print(f"  Performance: {row['num_trades']} trades, {row['total_return_pct']:.2f}% return, Sharpe={row['sharpe_ratio']:.2f}")

if __name__ == "__main__":
    main()