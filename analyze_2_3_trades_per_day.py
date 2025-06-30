#!/usr/bin/env python3
"""
Find and analyze strategies that produce exactly 2-3 trades per day.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_2_3_trades_strategies():
    print("=== STRATEGIES WITH 2-3 TRADES PER DAY ===\n")
    
    # Load filter group analysis
    df = pd.read_csv("keltner_filter_group_analysis.csv")
    
    # Add trades per day
    df['trades_per_day'] = df['avg_trades'] / 252
    
    # Filter for 2-3 trades per day (with some tolerance)
    target_strategies = df[(df['trades_per_day'] >= 1.8) & (df['trades_per_day'] <= 3.2)]
    
    # Sort by return
    target_strategies = target_strategies.sort_values('avg_return_bps', ascending=False)
    
    print("STRATEGIES WITH 2-3 TRADES/DAY:")
    print("="*120)
    print(f"{'Signals':<10} {'Trades/Day':<12} {'RPT (bps)':<12} {'Annual %':<10} {'Win Rate':<10} {'Sharpe':<10} {'Long bps':<10} {'Short bps':<10}")
    print("-"*120)
    
    for _, row in target_strategies.iterrows():
        annual_return = row['avg_return_bps'] * row['avg_trades'] / 100
        print(f"{row['signal_count']:<10.0f} {row['trades_per_day']:<12.2f} "
              f"{row['avg_return_bps']:<12.2f} {annual_return:<10.1f} "
              f"{row['avg_win_rate']*100:<10.1f} {row['avg_sharpe']:<10.2f} "
              f"{row['avg_long_bps']:<10.2f} {row['avg_short_bps']:<10.2f}")
    
    # Detailed analysis of top performers
    print("\n" + "="*80)
    print("DETAILED ANALYSIS OF TOP PERFORMERS:")
    print("="*80)
    
    if len(target_strategies) > 0:
        # Best overall
        best_idx = target_strategies['avg_return_bps'].idxmax()
        best = target_strategies.loc[best_idx]
        
        print(f"\n1. BEST RETURN PER TRADE ({best['signal_count']:.0f} signals):")
        print(f"   - Return: {best['avg_return_bps']:.2f} bps/trade")
        print(f"   - Trades/day: {best['trades_per_day']:.2f}")
        print(f"   - Annual return: {best['avg_return_bps'] * best['avg_trades'] / 100:.1f}%")
        print(f"   - Win rate: {best['avg_win_rate']*100:.1f}%")
        print(f"   - Filter reduction: {best['filter_reduction']:.1f}%")
        print(f"   - Direction: Long {best['avg_long_bps']:.2f} vs Short {best['avg_short_bps']:.2f}")
        
        # Most balanced
        positive_returns = target_strategies[target_strategies['avg_return_bps'] > 0]
        if len(positive_returns) > 0:
            # Find most balanced long/short
            positive_returns['balance'] = abs(positive_returns['avg_long_bps'] + positive_returns['avg_short_bps'])
            balanced_idx = positive_returns['balance'].idxmin()
            balanced = positive_returns.loc[balanced_idx]
            
            print(f"\n2. MOST BALANCED LONG/SHORT ({balanced['signal_count']:.0f} signals):")
            print(f"   - Return: {balanced['avg_return_bps']:.2f} bps/trade")
            print(f"   - Trades/day: {balanced['trades_per_day']:.2f}")
            print(f"   - Long: {balanced['avg_long_bps']:.2f} bps")
            print(f"   - Short: {balanced['avg_short_bps']:.2f} bps")
            print(f"   - Balance score: {balanced['balance']:.2f}")
        
        # Best Sharpe
        sharpe_idx = target_strategies['avg_sharpe'].idxmax()
        sharpe = target_strategies.loc[sharpe_idx]
        
        print(f"\n3. BEST RISK-ADJUSTED ({sharpe['signal_count']:.0f} signals):")
        print(f"   - Sharpe ratio: {sharpe['avg_sharpe']:.3f}")
        print(f"   - Return: {sharpe['avg_return_bps']:.2f} bps/trade")
        print(f"   - Trades/day: {sharpe['trades_per_day']:.2f}")
        print(f"   - Win rate: {sharpe['avg_win_rate']*100:.1f}%")
    
    # Filter type analysis
    print("\n" + "="*80)
    print("LIKELY FILTER TYPES:")
    print("="*80)
    
    filter_map = {
        1500: "Directional/Long-only filter",
        1510: "Time + Volume filter", 
        1535: "Light volatility filter",
        1155: "RSI + Trend filter",
        1202: "Volume spike filter",
        1315: "VWAP distance filter"
    }
    
    for _, row in target_strategies.iterrows():
        signal_count = row['signal_count']
        filter_type = filter_map.get(signal_count, "Unknown combination")
        print(f"\n{signal_count:.0f} signals: {filter_type}")
        print(f"  Performance: {row['avg_return_bps']:.2f} bps/trade")
        print(f"  Characteristics: ", end="")
        
        if row['avg_long_bps'] > row['avg_short_bps'] * 2:
            print("Strong long bias", end="")
        elif row['avg_short_bps'] > row['avg_long_bps'] * 2:
            print("Strong short bias", end="")
        else:
            print("Balanced directional", end="")
            
        if row['avg_win_rate'] > 0.65:
            print(", High win rate", end="")
        
        print()
    
    # Summary recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR 2-3 TRADES/DAY TARGET:")
    print("="*80)
    
    viable = target_strategies[target_strategies['avg_return_bps'] > 0.3]
    
    if len(viable) > 0:
        print(f"\nFound {len(viable)} viable strategies with positive returns:")
        
        for i, (_, row) in enumerate(viable.head(3).iterrows(), 1):
            print(f"\n{i}. Signal count {row['signal_count']:.0f}:")
            print(f"   - {row['avg_return_bps']:.2f} bps/trade")
            print(f"   - {row['trades_per_day']:.2f} trades/day")
            print(f"   - {row['avg_return_bps'] * row['avg_trades'] / 100:.1f}% annual return")
            
            if row['avg_long_bps'] > row['avg_short_bps'] * 1.5:
                print(f"   - Consider long-only implementation")
    else:
        print("\n⚠️  No strategies with >0.3 bps/trade in the 2-3 trades/day range")
        print("Consider relaxing frequency requirements or accepting lower returns")

if __name__ == "__main__":
    analyze_2_3_trades_strategies()