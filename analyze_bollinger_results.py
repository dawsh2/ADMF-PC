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

def analyze_strategy_performance(signal_file):
    """Analyze performance of a single strategy from parquet file."""
    try:
        # Load the parquet file
        df = pd.read_parquet(signal_file)
        
        if df.empty or 'signal' not in df.columns:
            return None
            
        # Get signals
        signals = df['signal'].values
        
        # Count trades (transitions from 0 to non-zero)
        trades = 0
        in_position = False
        
        for i in range(1, len(signals)):
            if signals[i] != 0 and not in_position:
                trades += 1
                in_position = True
            elif signals[i] == 0 and in_position:
                in_position = False
        
        # Extract strategy ID from filename
        strategy_id = int(signal_file.stem.split('_')[-1])
        
        return {
            'strategy_id': strategy_id,
            'total_signals': len(df),
            'non_zero_signals': (signals != 0).sum(),
            'signal_ratio': (signals != 0).sum() / len(df) if len(df) > 0 else 0,
            'num_trades': trades,
            'unique_signals': np.unique(signals).tolist()
        }
        
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
        print(f"- Symbols: {config.get('data_config', {}).get('symbols', 'N/A')}")
        print(f"- Date range: {config.get('data_config', {}).get('start_date', 'N/A')} to {config.get('data_config', {}).get('end_date', 'N/A')}")
        
        # Extract Bollinger parameters if available
        if 'strategies' in config:
            for strategy in config['strategies']:
                if strategy.get('type') == 'bollinger_bands':
                    print(f"\nBollinger Band Parameters:")
                    print(f"- Period range: {strategy.get('period', 'N/A')}")
                    print(f"- Std dev range: {strategy.get('std_dev', 'N/A')}")
                    print(f"- Exit conditions: {strategy.get('exit_conditions', 'N/A')}")
                    break
    
    # Analyze all strategy files
    print("\nAnalyzing strategy performance...")
    parquet_files = list(TRACES_DIR.glob("*.parquet"))
    print(f"Found {len(parquet_files)} strategy files to analyze")
    
    results = []
    for i, file in enumerate(parquet_files):
        if i % 100 == 0:
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
    print("-" * 40)
    
    # Overall statistics
    print(f"Average trades per strategy: {df_results['num_trades'].mean():.2f}")
    print(f"Max trades: {df_results['num_trades'].max()}")
    print(f"Min trades: {df_results['num_trades'].min()}")
    print(f"Strategies with >0 trades: {(df_results['num_trades'] > 0).sum()} ({(df_results['num_trades'] > 0).sum() / len(df_results) * 100:.1f}%)")
    
    print(f"\nAverage signal ratio: {df_results['signal_ratio'].mean():.4f}")
    print(f"Max signal ratio: {df_results['signal_ratio'].max():.4f}")
    
    # Top performing strategies by trade count
    print("\nTop 10 Strategies by Trade Count:")
    print("-" * 40)
    top_strategies = df_results.nlargest(10, 'num_trades')
    for _, row in top_strategies.iterrows():
        print(f"Strategy {row['strategy_id']}: {row['num_trades']} trades, signal ratio: {row['signal_ratio']:.4f}")
    
    # Save detailed results
    output_file = "bollinger_analysis_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Additional analysis - parameter extraction from metadata
    if 'experiment_results' in metadata and 'strategies' in metadata['experiment_results']:
        print("\nExtracting strategy parameters...")
        strategies_data = []
        
        for strategy_id, strategy_info in metadata['experiment_results']['strategies'].items():
            if 'config' in strategy_info:
                config = strategy_info['config']
                strategies_data.append({
                    'strategy_id': int(strategy_id),
                    'period': config.get('period', 'N/A'),
                    'std_dev': config.get('std_dev', 'N/A'),
                    'exit_at_band': config.get('exit_at_band', 'N/A'),
                    'exit_at_mid': config.get('exit_at_mid', 'N/A')
                })
        
        if strategies_data:
            df_params = pd.DataFrame(strategies_data)
            
            # Merge with performance data
            df_combined = df_results.merge(df_params, on='strategy_id', how='left')
            
            # Find best parameters
            best_by_trades = df_combined.nlargest(5, 'num_trades')
            print("\nTop 5 Parameter Combinations by Trade Count:")
            print("-" * 60)
            for _, row in best_by_trades.iterrows():
                print(f"Strategy {row['strategy_id']}:")
                print(f"  - Period: {row['period']}, Std Dev: {row['std_dev']}")
                print(f"  - Exit at band: {row['exit_at_band']}, Exit at mid: {row['exit_at_mid']}")
                print(f"  - Trades: {row['num_trades']}, Signal ratio: {row['signal_ratio']:.4f}")
                print()
            
            # Save combined results
            combined_output = "bollinger_analysis_with_params.csv"
            df_combined.to_csv(combined_output, index=False)
            print(f"Combined results with parameters saved to: {combined_output}")

if __name__ == "__main__":
    main()