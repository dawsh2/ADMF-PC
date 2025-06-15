#!/usr/bin/env python3
"""
Analyze grid search results with regime-aware performance
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import os

def load_strategy_signals(workspace_path: str) -> pd.DataFrame:
    """Load all strategy signals into a combined DataFrame."""
    
    signals_dir = Path(workspace_path) / "traces" / "SPY_1m" / "signals"
    
    all_signals = []
    
    for strategy_dir in signals_dir.iterdir():
        if not strategy_dir.is_dir():
            continue
            
        strategy_type = strategy_dir.name
        
        for file_path in strategy_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(file_path)
                df['strategy_type'] = strategy_type
                df['file_name'] = file_path.name
                all_signals.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    if not all_signals:
        return pd.DataFrame()
    
    combined = pd.concat(all_signals, ignore_index=True)
    
    # Rename columns to standard names
    combined = combined.rename(columns={
        'val': 'signal_value',
        'strat': 'strategy_id', 
        'ts': 'timestamp',
        'sym': 'symbol',
        'px': 'price'
    })
    
    return combined

def load_classifier_regimes(workspace_path: str) -> pd.DataFrame:
    """Load all classifier regime data."""
    
    classifiers_dir = Path(workspace_path) / "traces" / "SPY_1m" / "classifiers"
    
    all_regimes = []
    
    for classifier_dir in classifiers_dir.iterdir():
        if not classifier_dir.is_dir():
            continue
            
        classifier_type = classifier_dir.name
        
        for file_path in classifier_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(file_path)
                df['classifier_type'] = classifier_type
                df['file_name'] = file_path.name
                all_regimes.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    if not all_regimes:
        return pd.DataFrame()
    
    combined = pd.concat(all_regimes, ignore_index=True)
    
    # Rename columns
    combined = combined.rename(columns={
        'val': 'regime',
        'strat': 'classifier_id',
        'ts': 'timestamp', 
        'sym': 'symbol',
        'px': 'price'
    })
    
    return combined

def calculate_signal_performance(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate performance metrics for each strategy."""
    
    performance_results = []
    
    for (strategy_type, strategy_id), group in signals_df.groupby(['strategy_type', 'strategy_id']):
        
        # Calculate basic metrics
        total_signals = len(group)
        
        if total_signals == 0:
            continue
            
        # Signal distribution
        signal_counts = group['signal_value'].value_counts()
        buy_signals = signal_counts.get(1, 0)
        sell_signals = signal_counts.get(-1, 0) 
        neutral_signals = signal_counts.get(0, 0)
        
        # Calculate simple returns (assuming we can act on signals)
        group_sorted = group.sort_values('timestamp')
        
        # Simple momentum calculation - if signal is 1, assume next period return is positive
        returns = []
        for i in range(len(group_sorted) - 1):
            signal = group_sorted.iloc[i]['signal_value']
            if signal != 0:  # Only consider non-zero signals
                # Very simplified return calculation
                current_price = group_sorted.iloc[i]['price']
                next_price = group_sorted.iloc[i+1]['price'] 
                
                if current_price > 0:
                    ret = (next_price - current_price) / current_price
                    # Apply signal direction
                    directional_return = ret * signal
                    returns.append(directional_return)
        
        if len(returns) > 10:  # Need minimum trades
            returns = np.array(returns)
            
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            win_rate = np.mean(returns > 0) * 100
            
            performance_results.append({
                'strategy_type': strategy_type,
                'strategy_id': strategy_id,
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'neutral_signals': neutral_signals,
                'active_signals': buy_signals + sell_signals,
                'num_trades': len(returns),
                'avg_return_pct': avg_return * 100,
                'return_std': std_return * 100,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate
            })
    
    return pd.DataFrame(performance_results)

def analyze_regime_performance(signals_df: pd.DataFrame, regimes_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze strategy performance by regime."""
    
    # Merge signals with regimes based on timestamp proximity
    regime_performance = []
    
    print("Analyzing regime-specific performance...")
    print(f"Signals: {len(signals_df)}, Regimes: {len(regimes_df)}")
    
    # For each classifier
    for classifier_type in regimes_df['classifier_type'].unique():
        classifier_regimes = regimes_df[regimes_df['classifier_type'] == classifier_type]
        
        print(f"\nAnalyzing classifier: {classifier_type}")
        print(f"Regime distribution: {classifier_regimes['regime'].value_counts().to_dict()}")
        
        # For each regime in this classifier
        for regime in classifier_regimes['regime'].unique():
            regime_data = classifier_regimes[classifier_regimes['regime'] == regime]
            
            # Get time ranges for this regime
            regime_times = set(regime_data['timestamp'])
            
            # Find signals that occurred during this regime
            regime_signals = signals_df[signals_df['timestamp'].isin(regime_times)]
            
            if len(regime_signals) > 0:
                # Calculate performance for each strategy in this regime
                perf_df = calculate_signal_performance(regime_signals)
                
                if len(perf_df) > 0:
                    perf_df['classifier'] = classifier_type
                    perf_df['regime'] = regime
                    regime_performance.append(perf_df)
    
    if regime_performance:
        return pd.concat(regime_performance, ignore_index=True)
    else:
        return pd.DataFrame()

def print_summary_report(signals_df: pd.DataFrame, regimes_df: pd.DataFrame, 
                        performance_df: pd.DataFrame, regime_perf_df: pd.DataFrame):
    """Print comprehensive summary report."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE GRID SEARCH ANALYSIS")
    print("="*80)
    
    # Overall statistics
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total signals generated: {len(signals_df):,}")
    print(f"  Total regime classifications: {len(regimes_df):,}")
    print(f"  Strategy types tested: {signals_df['strategy_type'].nunique()}")
    print(f"  Individual strategies: {signals_df['strategy_id'].nunique()}")
    print(f"  Classifier types: {regimes_df['classifier_type'].nunique()}")
    
    # Strategy performance
    if len(performance_df) > 0:
        print(f"\nSTRATEGY PERFORMANCE SUMMARY:")
        top_strategies = performance_df.nlargest(10, 'sharpe_ratio')
        
        print(f"\nTop 10 Strategies by Sharpe Ratio:")
        for i, (_, row) in enumerate(top_strategies.iterrows(), 1):
            print(f"  {i:2d}. {row['strategy_type']:<25} → Sharpe: {row['sharpe_ratio']:.3f}, "
                  f"Win Rate: {row['win_rate']:.1f}%, Trades: {row['num_trades']}")
    
    # Regime analysis
    if len(regime_perf_df) > 0:
        print(f"\nREGIME-SPECIFIC PERFORMANCE:")
        
        # Best strategy for each regime
        print(f"\nBest Strategy per Regime:")
        for (classifier, regime), group in regime_perf_df.groupby(['classifier', 'regime']):
            if len(group) > 0:
                best = group.loc[group['sharpe_ratio'].idxmax()]
                print(f"  {classifier}:{regime:<15} → {best['strategy_type']:<25} "
                      f"(Sharpe: {best['sharpe_ratio']:.3f})")
    
    # Strategy type breakdown
    print(f"\nSTRATEGY TYPE BREAKDOWN:")
    strategy_summary = signals_df.groupby('strategy_type').agg({
        'strategy_id': 'nunique',
        'signal_value': ['count', lambda x: (x != 0).sum()]
    }).round(2)
    
    strategy_summary.columns = ['num_configs', 'total_signals', 'active_signals']
    strategy_summary['activity_rate'] = (strategy_summary['active_signals'] / 
                                       strategy_summary['total_signals'] * 100).round(1)
    
    for strategy_type, row in strategy_summary.iterrows():
        print(f"  {strategy_type:<30} → {int(row['num_configs']):2d} configs, "
              f"{int(row['total_signals']):6,d} signals ({row['activity_rate']:4.1f}% active)")

def main():
    workspace_path = "workspaces/expansive_grid_search_0397bd70"
    
    print("Loading strategy signals...")
    signals_df = load_strategy_signals(workspace_path)
    
    print("Loading classifier regimes...")
    regimes_df = load_classifier_regimes(workspace_path)
    
    if len(signals_df) == 0:
        print("No strategy signals found!")
        return
    
    if len(regimes_df) == 0:
        print("No classifier regimes found!")
        return
    
    print("Calculating strategy performance...")
    performance_df = calculate_signal_performance(signals_df)
    
    print("Analyzing regime-specific performance...")
    regime_perf_df = analyze_regime_performance(signals_df, regimes_df)
    
    # Print comprehensive report
    print_summary_report(signals_df, regimes_df, performance_df, regime_perf_df)
    
    # Save results
    if len(performance_df) > 0:
        performance_df.to_csv('strategy_performance_results.csv', index=False)
        print(f"\nStrategy performance saved to: strategy_performance_results.csv")
    
    if len(regime_perf_df) > 0:
        regime_perf_df.to_csv('regime_performance_results.csv', index=False)
        print(f"Regime performance saved to: regime_performance_results.csv")

if __name__ == "__main__":
    main()