#!/usr/bin/env python3
"""
Investigate the discrepancy between group average (0.68 bps) and individual analysis (-0.38 bps).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def investigate_discrepancy():
    print("=== INVESTIGATING 2826 STRATEGY DISCREPANCY ===\n")
    
    # Load the filter group analysis
    filter_df = pd.read_csv("keltner_filter_group_analysis.csv")
    group_2826 = filter_df[filter_df['signal_count'] == 2826].iloc[0]
    
    print("GROUP AVERAGE (from filter analysis):")
    print(f"  Return: {group_2826['avg_return_bps']:.2f} bps")
    print(f"  Trades: {group_2826['avg_trades']:.0f}")
    print(f"  Win rate: {group_2826['avg_win_rate']*100:.1f}%")
    
    # Now let's analyze multiple 2826 strategies to understand the variation
    workspace = "/Users/daws/ADMF-PC/config/keltner/results/latest"
    
    # Get all 2826 strategies
    with open(Path(workspace) / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    strategies_2826 = []
    for name, comp in metadata['components'].items():
        if comp.get('signal_changes') == 2826:
            strategy_num = int(name.split('_')[-1])
            strategies_2826.append(strategy_num)
    
    print(f"\nFound {len(strategies_2826)} strategies with 2826 signals")
    print(f"Analyzing first 5: {strategies_2826[:5]}")
    
    # Analyze each strategy
    results = []
    for strategy_num in strategies_2826[:5]:  # Sample first 5
        print(f"\nAnalyzing strategy {strategy_num}...")
        result = analyze_single_strategy(workspace, strategy_num)
        if result:
            results.append(result)
    
    # Compare results
    if results:
        df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("INDIVIDUAL STRATEGY RESULTS:")
        print("="*80)
        print(f"{'Strategy':<10} {'Trades':<10} {'RPT (bps)':<12} {'Win Rate':<10} {'Long RPT':<10} {'Short RPT':<10}")
        print("-"*80)
        
        for _, row in df.iterrows():
            print(f"{row['strategy']:<10} {row['trades']:<10} {row['return_bps']:<12.2f} "
                  f"{row['win_rate']*100:<10.1f} {row['long_return']:<10.2f} {row['short_return']:<10.2f}")
        
        print(f"\nAVERAGE: {df['trades'].mean():<10.0f} {df['return_bps'].mean():<12.2f} "
              f"{df['win_rate'].mean()*100:<10.1f}")
        
        # Check calculation method differences
        print("\n" + "="*60)
        print("CALCULATION METHOD ANALYSIS:")
        print("="*60)
        
        # Method 1: Simple position counting
        print("\nMethod 1: Count position changes (used in filter analysis)")
        sample_strategy = strategies_2826[0]
        signals_file = Path(workspace) / "traces" / "keltner_bands" / f"SPY_5m_compiled_strategy_{sample_strategy}.parquet"
        signals_df = pd.read_parquet(signals_file)
        
        # Count trades by position changes
        position_changes = 0
        last_val = 0
        for _, row in signals_df.iterrows():
            if row['val'] != last_val:
                position_changes += 1
                last_val = row['val']
        
        print(f"  Signal changes: {len(signals_df)}")
        print(f"  Position changes: {position_changes}")
        print(f"  Trades (changes/2): {position_changes/2:.0f}")
        
        # Method 2: Full simulation
        print("\nMethod 2: Full trade simulation (what we're doing)")
        print(f"  Actual trades simulated: {df.iloc[0]['trades']}")
        print(f"  Difference: {df.iloc[0]['trades'] - position_changes/2:.0f}")
        
        print("\nKEY INSIGHT:")
        print("The filter analysis likely used a simplified calculation")
        print("that doesn't account for:")
        print("- Exact entry/exit timing")
        print("- Signal filtering logic")
        print("- Position management rules")

def analyze_single_strategy(workspace, strategy_num):
    """Analyze a single strategy's actual performance."""
    try:
        strategy_name = f"SPY_5m_compiled_strategy_{strategy_num}"
        signals_file = Path(workspace) / "traces" / "keltner_bands" / f"{strategy_name}.parquet"
        
        signals_df = pd.read_parquet(signals_file)
        
        # Calculate trades
        trades = []
        current_position = None
        
        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            signal = row['val']
            price = row['px']
            
            if signal != 0:
                if current_position is not None:
                    # Close existing position
                    if current_position['direction'] == 'long':
                        ret = np.log(price / current_position['entry_price']) * 10000
                    else:
                        ret = -np.log(price / current_position['entry_price']) * 10000
                    
                    trades.append({
                        'return_bps': ret,
                        'direction': current_position['direction']
                    })
                
                # Open new position
                current_position = {
                    'entry_price': price,
                    'direction': 'long' if signal > 0 else 'short'
                }
            elif signal == 0 and current_position is not None:
                # Exit signal
                if current_position['direction'] == 'long':
                    ret = np.log(price / current_position['entry_price']) * 10000
                else:
                    ret = -np.log(price / current_position['entry_price']) * 10000
                
                trades.append({
                    'return_bps': ret,
                    'direction': current_position['direction']
                })
                current_position = None
        
        if not trades:
            return None
        
        trades_df = pd.DataFrame(trades)
        
        # Apply costs
        trades_df['return_net'] = trades_df['return_bps'] - 0.5
        
        # Calculate metrics
        long_trades = trades_df[trades_df['direction'] == 'long']
        short_trades = trades_df[trades_df['direction'] == 'short']
        
        return {
            'strategy': strategy_num,
            'trades': len(trades_df),
            'return_bps': trades_df['return_net'].mean(),
            'win_rate': (trades_df['return_net'] > 0).mean(),
            'long_return': long_trades['return_net'].mean() if len(long_trades) > 0 else 0,
            'short_return': short_trades['return_net'].mean() if len(short_trades) > 0 else 0,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades)
        }
        
    except Exception as e:
        print(f"Error analyzing strategy {strategy_num}: {e}")
        return None

if __name__ == "__main__":
    investigate_discrepancy()