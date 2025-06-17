#!/usr/bin/env python3
"""
Analyze strategy performance under selected regimes using the best balanced classifiers.

Based on the classifier analysis, we'll use the most balanced classifiers available
and analyze strategy performance during different market regimes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from decimal import Decimal

# Import from our existing calculate_log_returns script
def calculate_log_return_pnl_with_costs(df):
    """
    Calculate P&L using log returns with execution costs (simplified version).
    """
    if df.empty:
        return {
            'total_log_return': 0,
            'percentage_return': 0,
            'trades': [],
            'num_trades': 0,
            'win_rate': 0,
            'avg_trade_log_return': 0,
            'max_drawdown_pct': 0
        }
    
    trades = []
    current_position = 0
    entry_price = None
    entry_bar_idx = None
    total_log_return = 0
    log_return_curve = []
    
    for idx, row in df.iterrows():
        bar_idx = row['bar_idx']
        signal = row.get('signal_value', 0)
        price = row['price']
        
        # Track cumulative log return for drawdown calculation
        log_return_curve.append(total_log_return)
        
        if current_position == 0:
            # No position, check if we should open one
            if signal != 0:
                current_position = signal
                entry_price = price
                entry_bar_idx = bar_idx
        else:
            # We have a position
            if signal == 0 or signal != current_position:
                # Close position (either to flat or flip)
                if entry_price > 0 and price > 0:  # Avoid log(0) or log(negative)
                    # Calculate return
                    trade_log_return = float(np.log(float(price) / float(entry_price)) * current_position)
                    
                    trades.append({
                        'entry_bar': entry_bar_idx,
                        'exit_bar': bar_idx,
                        'entry_price': float(entry_price),
                        'exit_price': float(price),
                        'signal': current_position,
                        'log_return': trade_log_return,
                        'bars_held': bar_idx - entry_bar_idx
                    })
                    
                    total_log_return += trade_log_return
                
                # Reset position
                current_position = 0
                entry_price = None
                entry_bar_idx = None
                
                # If signal flip (not to zero), open new position
                if signal != 0:
                    current_position = signal
                    entry_price = price
                    entry_bar_idx = bar_idx
    
    # Calculate performance metrics
    if not trades:
        return {
            'total_log_return': 0,
            'percentage_return': 0,
            'trades': [],
            'num_trades': 0,
            'win_rate': 0,
            'avg_trade_log_return': 0,
            'max_drawdown_pct': 0
        }
    
    # Convert total log return to percentage return
    percentage_return = np.exp(total_log_return) - 1
    
    winning_trades = [t for t in trades if t['log_return'] > 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0
    avg_trade_log_return = total_log_return / len(trades) if trades else 0
    
    # Calculate maximum drawdown in percentage terms
    log_return_curve = np.array(log_return_curve)
    percentage_curve = np.exp(log_return_curve) - 1  # Convert to percentage returns
    running_max = np.maximum.accumulate(1 + percentage_curve)  # Running max of (1 + return)
    drawdown = (1 + percentage_curve) / running_max - 1  # Drawdown as fraction
    max_drawdown_pct = np.min(drawdown)
    
    return {
        'total_log_return': total_log_return,
        'percentage_return': percentage_return,
        'trades': trades,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_trade_log_return': avg_trade_log_return,
        'max_drawdown_pct': max_drawdown_pct
    }

def load_selected_classifiers():
    """Load the most balanced classifiers for analysis."""
    
    # Based on the analysis, select the top balanced classifiers
    # These have the lowest balance scores and reasonable state distributions
    selected_classifiers = [
        "SPY_volatility_momentum_grid_12_75_30",
        "SPY_volatility_momentum_grid_16_75_30", 
        "SPY_volatility_momentum_grid_20_75_30",
        "SPY_volatility_momentum_grid_16_80_20",
        "SPY_volatility_momentum_grid_20_75_25",
        "SPY_volatility_momentum_grid_12_80_20"
    ]
    
    print(f"Selected {len(selected_classifiers)} most balanced classifiers:")
    for classifier in selected_classifiers:
        print(f"  • {classifier}")
    
    return selected_classifiers

def load_classifier_states(workspace_path, classifier_name):
    """Load classifier state data from parquet file."""
    
    classifier_file = workspace_path / "traces" / "SPY_1m" / "classifiers" / "volatility_momentum_grid" / f"{classifier_name}.parquet"
    
    if not classifier_file.exists():
        print(f"Classifier file not found: {classifier_file}")
        return None
    
    try:
        df = pd.read_parquet(classifier_file)
        # Rename columns to standard format
        df = df.rename(columns={
            'idx': 'bar_idx',
            'val': 'state',
            'px': 'price'
        })
        return df[['bar_idx', 'state', 'price']].sort_values('bar_idx')
    except Exception as e:
        print(f"Error loading classifier {classifier_name}: {e}")
        return None

def find_strategy_files(workspace_path, limit=20):
    """Find strategy signal files for analysis."""
    
    signals_dir = workspace_path / "traces" / "SPY_1m" / "signals"
    
    if not signals_dir.exists():
        print(f"Signals directory not found: {signals_dir}")
        return []
    
    # Find all strategy parquet files
    strategy_files = list(signals_dir.rglob("*.parquet"))
    
    if len(strategy_files) > limit:
        print(f"Found {len(strategy_files)} strategy files, limiting to first {limit} for analysis")
        strategy_files = strategy_files[:limit]
    
    return strategy_files

def load_strategy_signals(file_path):
    """Load strategy signal data from parquet file."""
    
    try:
        df = pd.read_parquet(file_path)
        # Rename columns to standard format expected by calculate_log_returns
        df = df.rename(columns={
            'idx': 'bar_idx',
            'val': 'signal_value',
            'px': 'price'
        })
        return df[['bar_idx', 'signal_value', 'price']].sort_values('bar_idx')
    except Exception as e:
        print(f"Error loading strategy file {file_path}: {e}")
        return None

def filter_signals_by_regime(signals_df, classifier_df, target_regime):
    """Filter strategy signals to only include those during a specific regime."""
    
    # Merge signals with classifier states
    merged = pd.merge_asof(
        signals_df.sort_values('bar_idx'),
        classifier_df.sort_values('bar_idx'),
        on='bar_idx',
        direction='backward',
        suffixes=('_signal', '_classifier')
    )
    
    # Filter to target regime
    regime_signals = merged[merged['state'] == target_regime].copy()
    
    # Use signal price if available, otherwise classifier price
    if 'price_signal' in regime_signals.columns:
        regime_signals['price'] = regime_signals['price_signal'].fillna(regime_signals['price_classifier'])
    else:
        regime_signals['price'] = regime_signals['price_classifier']
    
    return regime_signals[['bar_idx', 'signal_value', 'price']].dropna()

def analyze_strategy_performance_by_regime(workspace_path, classifier_name, strategy_files, max_strategies=10):
    """Analyze strategy performance broken down by market regime."""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING PERFORMANCE BY REGIME - CLASSIFIER: {classifier_name}")
    print(f"{'='*80}")
    
    # Load classifier states
    classifier_df = load_classifier_states(workspace_path, classifier_name)
    
    if classifier_df is None:
        print(f"Could not load classifier {classifier_name}")
        return {}
    
    # Get unique states and their distributions
    state_distribution = classifier_df['state'].value_counts()
    print(f"\nClassifier state distribution:")
    for state, count in state_distribution.items():
        pct = (count / len(classifier_df)) * 100
        print(f"  {state}: {count:,} bars ({pct:.1f}%)")
    
    results = {
        'classifier': classifier_name,
        'state_distribution': state_distribution.to_dict(),
        'strategies': {}
    }
    
    # Analyze each strategy
    for i, strategy_file in enumerate(strategy_files[:max_strategies]):
        strategy_name = strategy_file.stem
        print(f"\n{'-'*60}")
        print(f"Strategy {i+1}/{min(len(strategy_files), max_strategies)}: {strategy_name}")
        print(f"{'-'*60}")
        
        # Load strategy signals
        signals_df = load_strategy_signals(strategy_file)
        
        if signals_df is None:
            print(f"Could not load strategy {strategy_name}")
            continue
        
        print(f"Total signals: {len(signals_df)}")
        
        strategy_results = {
            'file_path': str(strategy_file),
            'total_signals': len(signals_df),
            'regimes': {}
        }
        
        # Analyze performance in each regime
        for state in state_distribution.index:
            regime_signals = filter_signals_by_regime(signals_df, classifier_df, state)
            
            if len(regime_signals) == 0:
                print(f"  {state}: No signals during this regime")
                strategy_results['regimes'][state] = {
                    'signal_count': 0,
                    'performance': None
                }
                continue
            
            print(f"  {state}: {len(regime_signals)} signals")
            
            # Calculate performance using log returns
            performance = calculate_log_return_pnl_with_costs(regime_signals)
            
            strategy_results['regimes'][state] = {
                'signal_count': len(regime_signals),
                'performance': {
                    'total_log_return': performance['total_log_return'],
                    'percentage_return': performance['percentage_return'],
                    'num_trades': performance['num_trades'],
                    'win_rate': performance['win_rate'],
                    'max_drawdown_pct': performance['max_drawdown_pct']
                }
            }
            
            print(f"    Log return: {performance['total_log_return']:.4f}")
            print(f"    Percentage return: {performance['percentage_return']:.2%}")
            print(f"    Trades: {performance['num_trades']}")
            print(f"    Win rate: {performance['win_rate']:.2%}")
        
        results['strategies'][strategy_name] = strategy_results
    
    return results

def print_regime_performance_summary(results):
    """Print a summary of strategy performance by regime."""
    
    print(f"\n{'='*80}")
    print(f"REGIME PERFORMANCE SUMMARY - {results['classifier']}")
    print(f"{'='*80}")
    
    # Collect performance data by regime
    regime_performance = {}
    
    for strategy_name, strategy_data in results['strategies'].items():
        for regime, regime_data in strategy_data['regimes'].items():
            if regime_data['performance'] is None:
                continue
                
            if regime not in regime_performance:
                regime_performance[regime] = []
            
            regime_performance[regime].append({
                'strategy': strategy_name,
                'log_return': regime_data['performance']['total_log_return'],
                'percentage_return': regime_data['performance']['percentage_return'],
                'num_trades': regime_data['performance']['num_trades'],
                'win_rate': regime_data['performance']['win_rate']
            })
    
    # Print summary for each regime
    for regime in regime_performance.keys():
        regime_data = regime_performance[regime]
        
        if not regime_data:
            continue
            
        print(f"\n{regime.upper()} REGIME:")
        print(f"  Strategies analyzed: {len(regime_data)}")
        
        # Calculate aggregate statistics
        log_returns = [s['log_return'] for s in regime_data]
        pct_returns = [s['percentage_return'] for s in regime_data]
        trades = [s['num_trades'] for s in regime_data]
        win_rates = [s['win_rate'] for s in regime_data if s['num_trades'] > 0]
        
        print(f"  Average log return: {np.mean(log_returns):.4f}")
        print(f"  Average percentage return: {np.mean(pct_returns):.2%}")
        print(f"  Average trades per strategy: {np.mean(trades):.1f}")
        if win_rates:
            print(f"  Average win rate: {np.mean(win_rates):.2%}")
        
        # Show top performers
        top_performers = sorted(regime_data, key=lambda x: x['log_return'], reverse=True)[:3]
        print(f"  Top 3 performers:")
        for i, perf in enumerate(top_performers):
            print(f"    {i+1}. {perf['strategy']}: {perf['log_return']:.4f} log return "
                  f"({perf['percentage_return']:.2%}, {perf['num_trades']} trades)")

def save_results(results, output_file):
    """Save analysis results to JSON file."""
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_file}")

def main():
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    
    if not workspace_path.exists():
        print(f"Workspace not found: {workspace_path}")
        return
    
    print("STRATEGY PERFORMANCE ANALYSIS BY MARKET REGIME")
    print("="*50)
    
    # Load selected classifiers
    selected_classifiers = load_selected_classifiers()
    
    # Find strategy files to analyze
    strategy_files = find_strategy_files(workspace_path, limit=10)  # Limit for initial analysis
    
    if not strategy_files:
        print("No strategy files found")
        return
    
    print(f"\nFound {len(strategy_files)} strategy files for analysis")
    
    # Analyze performance with the best classifier
    primary_classifier = selected_classifiers[0]  # Use the most balanced one
    
    results = analyze_strategy_performance_by_regime(
        workspace_path, 
        primary_classifier, 
        strategy_files,
        max_strategies=10
    )
    
    if results:
        # Print summary
        print_regime_performance_summary(results)
        
        # Save results
        output_file = workspace_path / f"regime_performance_analysis_{primary_classifier}.json"
        save_results(results, output_file)

if __name__ == "__main__":
    main()