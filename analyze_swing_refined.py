#!/usr/bin/env python3
"""Analyze Swing Pivot Bounce refined optimization results."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_optimization_results(workspace_path):
    """Analyze all strategy results from optimization run."""
    # Load metadata
    metadata_path = Path(workspace_path) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get all strategy components
    components = metadata.get('components', {})
    
    # Analyze each strategy variant
    results = []
    
    for strategy_name, strategy_data in components.items():
        if strategy_data.get('component_type') != 'strategy':
            continue
            
        # Extract parameters from strategy name or metadata
        params = strategy_data.get('parameters', {})
        
        # Load signal file
        signal_file = Path(workspace_path) / strategy_data['signal_file_path']
        if not signal_file.exists():
            continue
            
        df = pd.read_parquet(signal_file)
        
        # Calculate metrics
        trades = []
        current_position = None
        
        for i in range(len(df)):
            row = df.iloc[i]
            signal = row['val']
            price = row['px']
            bar_idx = row['idx']
            
            if current_position is None and signal != 0:
                current_position = {
                    'entry_price': price,
                    'entry_bar': bar_idx,
                    'direction': signal
                }
            elif current_position is not None and (signal == 0 or signal != current_position['direction']):
                exit_price = price
                entry_price = current_position['entry_price']
                
                if current_position['direction'] > 0:
                    gross_return = (exit_price / entry_price) - 1
                else:
                    gross_return = (entry_price / exit_price) - 1
                
                net_return = gross_return - 0.0001  # 1bp round trip
                
                trades.append({
                    'return': net_return,
                    'bars_held': bar_idx - current_position['entry_bar']
                })
                
                if signal != 0:
                    current_position = {
                        'entry_price': price,
                        'entry_bar': bar_idx,
                        'direction': signal
                    }
                else:
                    current_position = None
        
        if trades:
            trades_df = pd.DataFrame(trades)
            num_trades = len(trades_df)
            win_rate = len(trades_df[trades_df['return'] > 0]) / num_trades
            total_return = (1 + trades_df['return']).prod() - 1
            avg_return = trades_df['return'].mean()
            
            # Annualize (assume 306 days)
            annual_return = (1 + total_return) ** (365.25 / 306) - 1
            
            # Extract parameters from compiled strategy name
            try:
                strategy_idx = int(strategy_name.split('_')[-1])
            except:
                strategy_idx = 0
            
            results.append({
                'strategy_idx': strategy_idx,
                'strategy_name': strategy_name,
                'trades': num_trades,
                'trades_per_day': num_trades / 306,
                'win_rate': win_rate,
                'total_return': total_return,
                'annual_return': annual_return,
                'avg_return': avg_return,
                'avg_bars_held': trades_df['bars_held'].mean() if len(trades_df) > 0 else 0
            })
    
    return pd.DataFrame(results)

def extract_parameters(strategy_idx, param_space):
    """Extract actual parameter values from strategy index."""
    # Parameter order from refined config:
    # sr_period: [30, 35, 40, 45, 50] = 5 values
    # min_touches: [3, 4, 5] = 3 values  
    # bounce_threshold: [0.0035, 0.004, 0.0045, 0.005] = 4 values
    # exit_threshold: [0.0012, 0.0014, 0.0015, 0.0016, 0.0018] = 5 values
    # Total: 5 * 3 * 4 * 5 = 300 combinations
    
    sr_periods = [30, 35, 40, 45, 50]
    min_touches = [3, 4, 5]
    bounce_thresholds = [0.0035, 0.004, 0.0045, 0.005]
    exit_thresholds = [0.0012, 0.0014, 0.0015, 0.0016, 0.0018]
    
    # Calculate indices
    exit_idx = strategy_idx % 5
    strategy_idx //= 5
    bounce_idx = strategy_idx % 4
    strategy_idx //= 4
    touch_idx = strategy_idx % 3
    strategy_idx //= 3
    sr_idx = strategy_idx % 5
    
    return {
        'sr_period': sr_periods[sr_idx],
        'min_touches': min_touches[touch_idx],
        'bounce_threshold': bounce_thresholds[bounce_idx],
        'exit_threshold': exit_thresholds[exit_idx]
    }

def main():
    """Analyze optimization results."""
    workspace_path = Path('/Users/daws/ADMF-PC/workspaces/signal_generation_55957a3c')
    
    print(f"Analyzing optimization workspace: {workspace_path.name}")
    
    results_df = analyze_optimization_results(workspace_path)
    
    if results_df.empty:
        print("No results found!")
        return
    
    # Add parameter columns
    for idx, row in results_df.iterrows():
        params = extract_parameters(int(row['strategy_idx']), None)
        for key, value in params.items():
            results_df.loc[idx, key] = value
    
    print(f"\nTotal parameter combinations tested: {len(results_df)}")
    
    # Find profitable strategies
    profitable = results_df[results_df['annual_return'] > 0].sort_values('annual_return', ascending=False)
    
    print(f"\nProfitable strategies: {len(profitable)} out of {len(results_df)} ({len(profitable)/len(results_df)*100:.1f}%)")
    
    if len(profitable) > 0:
        print("\nTop 10 Profitable Parameter Sets:")
        print("-" * 120)
        print(f"{'SR':<4} {'Touch':<6} {'Bounce':<8} {'Exit':<6} {'Trades':<8} {'T/Day':<8} {'Win%':<8} {'Annual':<10} {'Avg Ret':<10}")
        print("-" * 120)
        
        for _, row in profitable.head(10).iterrows():
            print(f"{row['sr_period']:<4.0f} {row['min_touches']:<6.0f} {row['bounce_threshold']:<8.4f} "
                  f"{row['exit_threshold']:<6.4f} {row['trades']:<8.0f} {row['trades_per_day']:<8.2f} "
                  f"{row['win_rate']*100:<8.1f} {row['annual_return']*100:<10.2f} {row['avg_return']*100:<10.3f}")
    
    # Analyze parameter impacts
    print("\n=== PARAMETER ANALYSIS ===")
    
    # Group by each parameter
    for param in ['sr_period', 'min_touches', 'bounce_threshold', 'exit_threshold']:
        print(f"\n{param.upper()} Impact on Annual Return:")
        param_stats = results_df.groupby(param)['annual_return'].agg(['mean', 'std', 'count'])
        param_stats['mean_pct'] = param_stats['mean'] * 100
        print(param_stats[['mean_pct', 'count']].sort_values('mean_pct', ascending=False))
    
    # Find best parameter ranges
    print("\n=== OPTIMAL PARAMETER RANGES ===")
    
    # Focus on strategies with reasonable trade frequency (0.5 - 3.0 trades/day)
    reasonable_freq = results_df[(results_df['trades_per_day'] >= 0.5) & (results_df['trades_per_day'] <= 3.0)]
    profitable_reasonable = reasonable_freq[reasonable_freq['annual_return'] > 0]
    
    if len(profitable_reasonable) > 0:
        print(f"\nStrategies with reasonable frequency and positive returns: {len(profitable_reasonable)}")
        
        # Find common parameter values
        print("\nMost common parameter values in profitable strategies:")
        for param in ['sr_period', 'min_touches', 'bounce_threshold', 'exit_threshold']:
            value_counts = profitable_reasonable[param].value_counts()
            print(f"\n{param}:")
            for value, count in value_counts.items():
                print(f"  {value}: {count} occurrences ({count/len(profitable_reasonable)*100:.1f}%)")
    
    # Best overall strategy
    if len(profitable) > 0:
        best = profitable.iloc[0]
        print(f"\n=== BEST STRATEGY ===")
        print(f"SR Period: {best['sr_period']}")
        print(f"Min Touches: {best['min_touches']}")
        print(f"Bounce Threshold: {best['bounce_threshold']:.4f}")
        print(f"Exit Threshold: {best['exit_threshold']:.4f}")
        print(f"\nPerformance:")
        print(f"Annual Return: {best['annual_return']*100:.2f}%")
        print(f"Win Rate: {best['win_rate']*100:.1f}%")
        print(f"Trades per Day: {best['trades_per_day']:.2f}")
        print(f"Average Return per Trade: {best['avg_return']*100:.3f}%")
    
    # Save results
    output_file = 'swing_refined_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")

if __name__ == "__main__":
    main()