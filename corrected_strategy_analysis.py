#!/usr/bin/env python3
"""
Corrected strategy performance analysis using proper linear P&L then log transformation.

Implements the correct approach:
1. Calculate linear P&L: t_i = (price_exit - price_entry) * signal_value - execution_cost
2. Sum all trades: Total_PnL = sum(t_i)
3. Convert to percentage: percentage_return = log(1 + Total_PnL/initial_capital)
4. Attribute trades to regime where position was OPENED
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from decimal import Decimal

def calculate_linear_pnl_with_costs(df, initial_capital=10000):
    """
    Calculate P&L using proper linear calculation then log transformation.
    
    For each trade: t_i = (price_exit - price_entry) * signal_value - execution_cost
    Total P&L = sum of all t_i
    Percentage return = log(1 + Total_PnL / initial_capital)
    """
    if df.empty:
        return {
            'total_pnl': 0,
            'percentage_return': 0,
            'log_return': 0,
            'trades': [],
            'num_trades': 0,
            'win_rate': 0,
            'avg_trade_pnl': 0
        }
    
    # Execution costs (simplified for now)
    COMMISSION_PER_TRADE = 1.0  # $1 per trade
    SLIPPAGE_BPS = 1  # 1 basis point slippage
    
    trades = []
    current_position = 0
    entry_price = None
    entry_bar_idx = None
    total_pnl = 0
    
    for idx, row in df.iterrows():
        bar_idx = row['bar_idx']
        signal = row.get('signal_value', 0)
        price = float(row['price'])
        
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
                if entry_price is not None and price > 0:
                    # Calculate linear P&L
                    price_diff = price - entry_price
                    gross_pnl = price_diff * current_position
                    
                    # Apply execution costs
                    commission_cost = COMMISSION_PER_TRADE * 2  # Entry + exit
                    slippage_cost = (entry_price + price) * (SLIPPAGE_BPS / 10000) * abs(current_position)
                    execution_cost = commission_cost + slippage_cost
                    
                    net_pnl = gross_pnl - execution_cost
                    
                    trades.append({
                        'entry_bar': entry_bar_idx,
                        'exit_bar': bar_idx,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'signal': current_position,
                        'gross_pnl': gross_pnl,
                        'execution_cost': execution_cost,
                        'net_pnl': net_pnl,
                        'bars_held': bar_idx - entry_bar_idx
                    })
                    
                    total_pnl += net_pnl
                
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
            'total_pnl': 0,
            'percentage_return': 0,
            'log_return': 0,
            'trades': [],
            'num_trades': 0,
            'win_rate': 0,
            'avg_trade_pnl': 0
        }
    
    # Convert to percentage and log returns
    percentage_return = total_pnl / initial_capital
    log_return = np.log(1 + percentage_return) if (1 + percentage_return) > 0 else -10
    
    winning_trades = [t for t in trades if t['net_pnl'] > 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0
    avg_trade_pnl = total_pnl / len(trades) if trades else 0
    
    return {
        'total_pnl': total_pnl,
        'percentage_return': percentage_return,
        'log_return': log_return,
        'trades': trades,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_trade_pnl': avg_trade_pnl,
        'initial_capital': initial_capital
    }

def load_classifier_state_changes(workspace_path, classifier_name):
    """Load classifier state changes (sparse data)."""
    
    classifier_file = workspace_path / "traces" / "SPY_1m" / "classifiers" / "volatility_momentum_grid" / f"{classifier_name}.parquet"
    
    if not classifier_file.exists():
        print(f"Classifier file not found: {classifier_file}")
        return None
    
    try:
        df = pd.read_parquet(classifier_file)
        df = df.rename(columns={
            'idx': 'bar_idx',
            'val': 'state',
            'px': 'price'
        })
        return df[['bar_idx', 'state']].sort_values('bar_idx')
    except Exception as e:
        print(f"Error loading classifier {classifier_name}: {e}")
        return None

def determine_trade_regime(trade, classifier_changes):
    """
    Determine which regime a trade belongs to based on where it was OPENED.
    Uses sparse classifier state changes to find the active regime.
    """
    entry_bar = trade['entry_bar']
    
    # Find the most recent classifier state change before or at entry
    relevant_changes = classifier_changes[classifier_changes['bar_idx'] <= entry_bar]
    
    if len(relevant_changes) == 0:
        return 'unknown'  # No classifier data before this trade
    
    # Get the most recent state change
    most_recent = relevant_changes.iloc[-1]
    return most_recent['state']

def load_strategy_signals(file_path):
    """Load strategy signal data."""
    
    try:
        df = pd.read_parquet(file_path)
        df = df.rename(columns={
            'idx': 'bar_idx',
            'val': 'signal_value',
            'px': 'price'
        })
        return df[['bar_idx', 'signal_value', 'price']].sort_values('bar_idx')
    except Exception as e:
        print(f"Error loading strategy file {file_path}: {e}")
        return None

def analyze_strategy_performance_corrected(workspace_path, classifier_name, strategy_files, max_strategies=10):
    """Analyze strategy performance with corrected P&L calculation and regime attribution."""
    
    print(f"\n{'='*80}")
    print(f"CORRECTED STRATEGY ANALYSIS - CLASSIFIER: {classifier_name}")
    print(f"{'='*80}")
    print("Using proper linear P&L calculation and regime attribution to opening bar")
    
    # Load classifier state changes (sparse)
    classifier_changes = load_classifier_state_changes(workspace_path, classifier_name)
    
    if classifier_changes is None:
        print(f"Could not load classifier {classifier_name}")
        return {}
    
    print(f"\nClassifier has {len(classifier_changes)} state changes")
    state_counts = classifier_changes['state'].value_counts()
    print("State change frequency:")
    for state, count in state_counts.items():
        print(f"  {state}: {count} changes")
    
    results = {
        'classifier': classifier_name,
        'state_changes': len(classifier_changes),
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
        
        # Calculate overall performance
        overall_perf = calculate_linear_pnl_with_costs(signals_df)
        
        print(f"Overall Performance:")
        print(f"  Total P&L: ${overall_perf['total_pnl']:.2f}")
        print(f"  Percentage Return: {overall_perf['percentage_return']:.2%}")
        print(f"  Log Return: {overall_perf['log_return']:.4f}")
        print(f"  Total Trades: {overall_perf['num_trades']}")
        print(f"  Win Rate: {overall_perf['win_rate']:.2%}")
        
        # Attribute trades to regimes based on opening bar
        regime_performance = {}
        
        for trade in overall_perf['trades']:
            regime = determine_trade_regime(trade, classifier_changes)
            
            if regime not in regime_performance:
                regime_performance[regime] = {
                    'trades': [],
                    'total_pnl': 0,
                    'count': 0
                }
            
            regime_performance[regime]['trades'].append(trade)
            regime_performance[regime]['total_pnl'] += trade['net_pnl']
            regime_performance[regime]['count'] += 1
        
        # Calculate regime-specific metrics
        print(f"\nPerformance by Regime (attributed to opening bar):")
        strategy_regime_results = {}
        
        for regime, data in regime_performance.items():
            if data['count'] == 0:
                continue
                
            trades = data['trades']
            total_pnl = data['total_pnl']
            
            winning_trades = [t for t in trades if t['net_pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_pnl = total_pnl / len(trades) if trades else 0
            
            # Convert to percentage return for this regime
            pct_return = total_pnl / overall_perf['initial_capital']
            log_return = np.log(1 + pct_return) if (1 + pct_return) > 0 else -10
            
            strategy_regime_results[regime] = {
                'trade_count': len(trades),
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'win_rate': win_rate,
                'percentage_return': pct_return,
                'log_return': log_return
            }
            
            print(f"  {regime}: {len(trades)} trades, ${total_pnl:.2f} P&L ({pct_return:.2%}), {win_rate:.1%} win rate")
        
        results['strategies'][strategy_name] = {
            'file_path': str(strategy_file),
            'overall': {
                'total_pnl': overall_perf['total_pnl'],
                'percentage_return': overall_perf['percentage_return'],
                'log_return': overall_perf['log_return'],
                'num_trades': overall_perf['num_trades'],
                'win_rate': overall_perf['win_rate']
            },
            'regimes': strategy_regime_results
        }
    
    return results

def print_corrected_regime_summary(results):
    """Print summary of corrected regime performance analysis."""
    
    print(f"\n{'='*80}")
    print(f"CORRECTED REGIME PERFORMANCE SUMMARY - {results['classifier']}")
    print(f"{'='*80}")
    print("Using proper linear P&L calculation and regime attribution")
    
    # Collect performance data by regime
    regime_performance = {}
    
    for strategy_name, strategy_data in results['strategies'].items():
        for regime, regime_data in strategy_data['regimes'].items():
            if regime not in regime_performance:
                regime_performance[regime] = []
            
            regime_performance[regime].append({
                'strategy': strategy_name,
                'total_pnl': regime_data['total_pnl'],
                'percentage_return': regime_data['percentage_return'],
                'log_return': regime_data['log_return'],
                'num_trades': regime_data['trade_count'],
                'win_rate': regime_data['win_rate']
            })
    
    # Print summary for each regime
    for regime in regime_performance.keys():
        regime_data = regime_performance[regime]
        
        if not regime_data:
            continue
            
        print(f"\n{regime.upper()} REGIME:")
        print(f"  Strategies analyzed: {len(regime_data)}")
        
        # Calculate aggregate statistics
        total_pnls = [s['total_pnl'] for s in regime_data]
        pct_returns = [s['percentage_return'] for s in regime_data]
        log_returns = [s['log_return'] for s in regime_data]
        trades = [s['num_trades'] for s in regime_data]
        win_rates = [s['win_rate'] for s in regime_data if s['num_trades'] > 0]
        
        print(f"  Average P&L: ${np.mean(total_pnls):.2f}")
        print(f"  Average percentage return: {np.mean(pct_returns):.2%}")
        print(f"  Average log return: {np.mean(log_returns):.4f}")
        print(f"  Average trades per strategy: {np.mean(trades):.1f}")
        if win_rates:
            print(f"  Average win rate: {np.mean(win_rates):.2%}")
        
        # Show top performers
        top_performers = sorted(regime_data, key=lambda x: x['total_pnl'], reverse=True)[:3]
        print(f"  Top 3 performers:")
        for i, perf in enumerate(top_performers):
            print(f"    {i+1}. {perf['strategy']}: ${perf['total_pnl']:.2f} "
                  f"({perf['percentage_return']:.2%}, {perf['num_trades']} trades)")

def main():
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    
    if not workspace_path.exists():
        print(f"Workspace not found: {workspace_path}")
        return
    
    print("CORRECTED STRATEGY PERFORMANCE ANALYSIS")
    print("="*50)
    print("Using proper linear P&L calculation and regime attribution")
    
    # Use the best classifier from previous analysis
    classifier_name = "SPY_volatility_momentum_grid_12_75_30"
    
    # Find strategy files (limit to MACD for comparison)
    signals_dir = workspace_path / "traces" / "SPY_1m" / "signals"
    strategy_files = list(signals_dir.rglob("*macd_crossover*.parquet"))[:10]
    
    if not strategy_files:
        print("No MACD strategy files found")
        return
    
    print(f"Found {len(strategy_files)} MACD strategy files for analysis")
    
    # Analyze performance
    results = analyze_strategy_performance_corrected(
        workspace_path, 
        classifier_name, 
        strategy_files,
        max_strategies=10
    )
    
    if results:
        # Print summary
        print_corrected_regime_summary(results)
        
        # Save results
        output_file = workspace_path / f"corrected_regime_performance_{classifier_name}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Corrected results saved to: {output_file}")

if __name__ == "__main__":
    main()