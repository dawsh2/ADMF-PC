#!/usr/bin/env python3
"""Analyze Swing Pivot Bounce Target-based exit optimization results."""

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
    # Parameter order from target test config:
    # sr_period: [30, 35, 40] = 3 values
    # min_touches: [3, 4, 5] = 3 values  
    # bounce_threshold: [0.003, 0.004, 0.005] = 3 values
    # target_threshold: [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005] = 8 values
    # Total: 3 * 3 * 3 * 8 = 216 combinations
    
    sr_periods = [30, 35, 40]
    min_touches = [3, 4, 5]
    bounce_thresholds = [0.003, 0.004, 0.005]
    target_thresholds = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005]
    
    # Calculate indices
    target_idx = strategy_idx % 8
    strategy_idx //= 8
    bounce_idx = strategy_idx % 3
    strategy_idx //= 3
    touch_idx = strategy_idx % 3
    strategy_idx //= 3
    sr_idx = strategy_idx % 3
    
    return {
        'sr_period': sr_periods[sr_idx],
        'min_touches': min_touches[touch_idx],
        'bounce_threshold': bounce_thresholds[bounce_idx],
        'target_threshold': target_thresholds[target_idx]
    }

def main():
    """Analyze optimization results."""
    workspace_path = Path('/Users/daws/ADMF-PC/workspaces/signal_generation_4957a25e')
    
    print(f"Analyzing optimization workspace: {workspace_path.name}")
    print("Strategy: swing_pivot_bounce_target (exits near opposite S/R level)")
    
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
        print(f"{'SR':<4} {'Touch':<6} {'Bounce':<8} {'Target':<8} {'Trades':<8} {'T/Day':<8} {'Win%':<8} {'Annual':<10} {'Avg Ret':<10}")
        print("-" * 120)
        
        for _, row in profitable.head(10).iterrows():
            print(f"{row['sr_period']:<4.0f} {row['min_touches']:<6.0f} {row['bounce_threshold']:<8.4f} "
                  f"{row['target_threshold']:<8.4f} {row['trades']:<8.0f} {row['trades_per_day']:<8.2f} "
                  f"{row['win_rate']*100:<8.1f} {row['annual_return']*100:<10.2f} {row['avg_return']*100:<10.3f}")
    else:
        # Show best performers even if not profitable
        print("\nTop 10 Best Performers (though not profitable):")
        best = results_df.nlargest(10, 'annual_return')
        print("-" * 120)
        print(f"{'SR':<4} {'Touch':<6} {'Bounce':<8} {'Target':<8} {'Trades':<8} {'T/Day':<8} {'Win%':<8} {'Annual':<10} {'Avg Ret':<10}")
        print("-" * 120)
        
        for _, row in best.iterrows():
            print(f"{row['sr_period']:<4.0f} {row['min_touches']:<6.0f} {row['bounce_threshold']:<8.4f} "
                  f"{row['target_threshold']:<8.4f} {row['trades']:<8.0f} {row['trades_per_day']:<8.2f} "
                  f"{row['win_rate']*100:<8.1f} {row['annual_return']*100:<10.2f} {row['avg_return']*100:<10.3f}")
    
    # Analyze parameter impacts
    print("\n=== PARAMETER ANALYSIS ===")
    
    # Group by each parameter
    for param in ['sr_period', 'min_touches', 'bounce_threshold', 'target_threshold']:
        print(f"\n{param.upper()} Impact on Annual Return:")
        param_stats = results_df.groupby(param)['annual_return'].agg(['mean', 'std', 'count'])
        param_stats['mean_pct'] = param_stats['mean'] * 100
        print(param_stats[['mean_pct', 'count']].sort_values('mean_pct', ascending=False))
    
    # Compare with original midpoint exit
    print("\n=== COMPARISON WITH MIDPOINT EXIT ===")
    print("Original midpoint exit best result: -0.52% annual return")
    print(f"Target-based exit best result: {results_df['annual_return'].max()*100:.2f}% annual return")
    improvement = results_df['annual_return'].max() - (-0.0052)
    print(f"Improvement: {improvement*100:.2f} percentage points")
    
    # Save results
    output_file = 'swing_target_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")

if __name__ == "__main__":
    main()