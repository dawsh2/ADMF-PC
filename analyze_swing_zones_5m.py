#!/usr/bin/env python3
"""Analyze Swing Pivot Bounce Zones 5m optimization results."""

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
            
            # Calculate per-trade metrics in basis points
            avg_return_bps = avg_return * 10000
            
            results.append({
                'strategy_idx': strategy_idx,
                'strategy_name': strategy_name,
                'trades': num_trades,
                'trades_per_day': num_trades / 306,
                'win_rate': win_rate,
                'total_return': total_return,
                'annual_return': annual_return,
                'avg_return': avg_return,
                'avg_return_bps': avg_return_bps,
                'avg_bars_held': trades_df['bars_held'].mean() if len(trades_df) > 0 else 0
            })
    
    return pd.DataFrame(results)

def extract_parameters(strategy_idx):
    """Extract actual parameter values from strategy index."""
    # Parameter order from zones 5m config:
    # sr_period: [12, 18, 24, 30, 36] = 5 values
    # min_touches: [3, 4, 5] = 3 values  
    # entry_zone: [0.002, 0.0025, 0.003, 0.0035, 0.004] = 5 values
    # exit_zone: [0.002, 0.0025, 0.003, 0.0035, 0.004] = 5 values
    # min_range: [0.003, 0.004, 0.005, 0.006] = 4 values
    # Total: 5 * 3 * 5 * 5 * 4 = 1500 combinations
    
    sr_periods = [12, 18, 24, 30, 36]
    min_touches = [3, 4, 5]
    entry_zones = [0.002, 0.0025, 0.003, 0.0035, 0.004]
    exit_zones = [0.002, 0.0025, 0.003, 0.0035, 0.004]
    min_ranges = [0.003, 0.004, 0.005, 0.006]
    
    # Calculate indices
    min_range_idx = strategy_idx % 4
    strategy_idx //= 4
    exit_zone_idx = strategy_idx % 5
    strategy_idx //= 5
    entry_zone_idx = strategy_idx % 5
    strategy_idx //= 5
    touch_idx = strategy_idx % 3
    strategy_idx //= 3
    sr_idx = strategy_idx % 5
    
    return {
        'sr_period': sr_periods[sr_idx],
        'min_touches': min_touches[touch_idx],
        'entry_zone': entry_zones[entry_zone_idx],
        'exit_zone': exit_zones[exit_zone_idx],
        'min_range': min_ranges[min_range_idx]
    }

def main():
    """Analyze optimization results."""
    workspace_path = Path('/Users/daws/ADMF-PC/workspaces/signal_generation_bfe07b48')
    
    print(f"Analyzing 5m zones optimization workspace: {workspace_path.name}")
    print("Strategy: swing_pivot_bounce_zones on 5-minute data")
    print("Total combinations: 1500\n")
    
    results_df = analyze_optimization_results(workspace_path)
    
    if results_df.empty:
        print("No results found!")
        return
    
    # Add parameter columns
    for idx, row in results_df.iterrows():
        params = extract_parameters(int(row['strategy_idx']))
        for key, value in params.items():
            results_df.loc[idx, key] = value
    
    print(f"Successfully analyzed: {len(results_df)} strategies")
    
    # Find profitable strategies
    profitable = results_df[results_df['annual_return'] > 0].sort_values('annual_return', ascending=False)
    
    print(f"\nProfitable strategies: {len(profitable)} out of {len(results_df)} ({len(profitable)/len(results_df)*100:.1f}%)")
    
    # Show top performers
    print("\nTop 20 Performers:")
    print("-" * 150)
    print(f"{'SR':<4} {'Touch':<6} {'Entry':<7} {'Exit':<7} {'MinRng':<8} {'Trades':<8} {'T/Day':<8} {'Win%':<8} {'Annual':<10} {'Avg(bps)':<10} {'Bars':<8}")
    print("-" * 150)
    
    top_20 = results_df.nlargest(20, 'annual_return')
    for _, row in top_20.iterrows():
        print(f"{row['sr_period']:<4.0f} {row['min_touches']:<6.0f} {row['entry_zone']:<7.4f} "
              f"{row['exit_zone']:<7.4f} {row['min_range']:<8.4f} {row['trades']:<8.0f} "
              f"{row['trades_per_day']:<8.2f} {row['win_rate']*100:<8.1f} "
              f"{row['annual_return']*100:<10.2f} {row['avg_return_bps']:<10.1f} "
              f"{row['avg_bars_held']:<8.1f}")
    
    # Parameter analysis
    print("\n=== PARAMETER IMPACT ON RETURNS ===")
    
    for param in ['sr_period', 'min_touches', 'entry_zone', 'exit_zone', 'min_range']:
        print(f"\n{param.upper()}:")
        param_stats = results_df.groupby(param).agg({
            'annual_return': ['mean', 'std', 'count'],
            'avg_return_bps': 'mean',
            'trades': 'mean'
        }).round(4)
        
        # Flatten column names
        param_stats.columns = ['_'.join(col).strip() for col in param_stats.columns]
        param_stats['annual_mean_pct'] = param_stats['annual_return_mean'] * 100
        
        print(param_stats[['annual_mean_pct', 'avg_return_bps_mean', 'trades_mean']].sort_values('annual_mean_pct', ascending=False))
    
    # Trade frequency analysis
    print("\n=== TRADE FREQUENCY ANALYSIS ===")
    print(f"Average trades per day: {results_df['trades_per_day'].mean():.2f}")
    print(f"Min trades per day: {results_df['trades_per_day'].min():.2f}")
    print(f"Max trades per day: {results_df['trades_per_day'].max():.2f}")
    
    # Best edge strategies (highest bps per trade)
    print("\n=== BEST EDGE (BPS PER TRADE) ===")
    best_edge = results_df.nlargest(10, 'avg_return_bps')
    for _, row in best_edge.iterrows():
        print(f"SR:{row['sr_period']:.0f} Touch:{row['min_touches']:.0f} "
              f"Entry:{row['entry_zone']:.3f} Exit:{row['exit_zone']:.3f} "
              f"→ {row['avg_return_bps']:.1f} bps/trade, "
              f"{row['trades']:.0f} trades, {row['annual_return']*100:.1f}% annual")
    
    # Save results
    output_file = 'swing_zones_5m_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")

if __name__ == "__main__":
    main()