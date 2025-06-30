#!/usr/bin/env python3
"""
Analyze why filters aren't improving Keltner strategy performance.
Focus on strategies with 2-3+ trades per day requirement.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_filter_analysis():
    """Load the filter group analysis data."""
    return pd.read_csv("keltner_filter_group_analysis.csv")

def analyze_filter_issues():
    print("=== KELTNER FILTER PERFORMANCE INVESTIGATION ===\n")
    
    # Load data
    df = load_filter_analysis()
    
    # Add trades per day
    df['trades_per_day'] = df['avg_trades'] / 252  # Assuming 252 trading days
    
    # 1. Identify strategies meeting frequency requirement (2-3+ trades/day)
    min_trades_per_day = 2.0
    frequent_strategies = df[df['trades_per_day'] >= min_trades_per_day]
    
    print(f"1. STRATEGIES WITH {min_trades_per_day}+ TRADES/DAY:")
    print("="*80)
    print(f"{'Signals':<10} {'Trades/Day':<12} {'RPT (bps)':<12} {'Win Rate':<10} {'Filter %':<10}")
    print("-"*80)
    
    for _, row in frequent_strategies.iterrows():
        print(f"{row['signal_count']:<10.0f} {row['trades_per_day']:<12.1f} "
              f"{row['avg_return_bps']:<12.2f} {row['avg_win_rate']*100:<10.1f} "
              f"{row['filter_reduction']:<10.1f}")
    
    # 2. Analyze the performance paradox
    print("\n2. PERFORMANCE PARADOX ANALYSIS:")
    print("="*60)
    
    # Calculate correlation between filter reduction and performance
    corr_filter_return = df['filter_reduction'].corr(df['avg_return_bps'])
    print(f"Correlation between filter reduction and returns: {corr_filter_return:.3f}")
    
    # Best performers by return
    print("\nTop 5 by returns:")
    top_performers = df.nlargest(5, 'avg_return_bps')
    for _, row in top_performers.iterrows():
        print(f"  {row['signal_count']:>4.0f} signals: {row['avg_return_bps']:>6.2f} bps, "
              f"{row['trades_per_day']:.1f} trades/day")
    
    # 3. Filter effectiveness by category
    print("\n3. FILTER CATEGORY BREAKDOWN:")
    print("="*60)
    
    categories = [
        ("No Filter", df[df['filter_reduction'] < 10]),
        ("Light (10-50%)", df[(df['filter_reduction'] >= 10) & (df['filter_reduction'] < 50)]),
        ("Moderate (50-90%)", df[(df['filter_reduction'] >= 50) & (df['filter_reduction'] < 90)]),
        ("Heavy (90-98%)", df[(df['filter_reduction'] >= 90) & (df['filter_reduction'] < 98)]),
        ("Extreme (>98%)", df[df['filter_reduction'] >= 98])
    ]
    
    for name, group in categories:
        if len(group) > 0:
            avg_return = group['avg_return_bps'].mean()
            avg_trades = group['avg_trades'].mean()
            avg_win = group['avg_win_rate'].mean()
            print(f"\n{name}:")
            print(f"  Avg return: {avg_return:.2f} bps")
            print(f"  Avg trades: {avg_trades:.0f} ({avg_trades/252:.1f}/day)")
            print(f"  Avg win rate: {avg_win*100:.1f}%")
            print(f"  Count: {len(group)} strategies")
    
    # 4. Why aren't filters working?
    print("\n4. POTENTIAL ISSUES:")
    print("="*60)
    
    # Check if filtering is removing good trades
    baseline = df[df['filter_reduction'] < 10]
    if len(baseline) > 0:
        baseline_return = baseline['avg_return_bps'].mean()
        baseline_win_rate = baseline['avg_win_rate'].mean()
        
        print(f"\nBaseline performance (no filter):")
        print(f"  Return: {baseline_return:.2f} bps")
        print(f"  Win rate: {baseline_win_rate*100:.1f}%")
        
        # Compare with filtered strategies
        filtered = df[df['filter_reduction'] > 50]
        if len(filtered) > 0:
            filtered_return = filtered['avg_return_bps'].mean()
            filtered_win_rate = filtered['avg_win_rate'].mean()
            
            print(f"\nFiltered performance (>50% reduction):")
            print(f"  Return: {filtered_return:.2f} bps")
            print(f"  Win rate: {filtered_win_rate*100:.1f}%")
            
            print(f"\nDifference:")
            print(f"  Return: {filtered_return - baseline_return:+.2f} bps")
            print(f"  Win rate: {(filtered_win_rate - baseline_win_rate)*100:+.1f}%")
    
    # 5. Look for sweet spot
    print("\n5. SEARCHING FOR SWEET SPOT:")
    print("="*60)
    
    # Find strategies with good balance
    good_balance = df[(df['trades_per_day'] >= 2) & (df['avg_return_bps'] > 0)]
    
    if len(good_balance) > 0:
        print(f"\nStrategies with 2+ trades/day AND positive returns:")
        print(f"Found {len(good_balance)} strategies\n")
        
        best_balanced = good_balance.nlargest(3, 'avg_return_bps')
        for _, row in best_balanced.iterrows():
            print(f"Signal count {row['signal_count']:.0f}:")
            print(f"  Return: {row['avg_return_bps']:.2f} bps")
            print(f"  Trades/day: {row['trades_per_day']:.1f}")
            print(f"  Annual return: {row['avg_return_bps'] * row['avg_trades'] / 100:.1f}%")
            print(f"  Filter reduction: {row['filter_reduction']:.1f}%")
            print()
    
    # 6. Long/Short analysis
    print("6. LONG/SHORT BREAKDOWN:")
    print("="*60)
    
    # Find strategies where one direction dominates
    df['long_short_diff'] = abs(df['avg_long_bps'] - df['avg_short_bps'])
    directional = df.nlargest(5, 'long_short_diff')
    
    print("\nMost directionally biased strategies:")
    for _, row in directional.iterrows():
        direction = "Long" if row['avg_long_bps'] > row['avg_short_bps'] else "Short"
        print(f"  {row['signal_count']:>4.0f} signals: {direction} bias, "
              f"L: {row['avg_long_bps']:.2f}, S: {row['avg_short_bps']:.2f}")
    
    # 7. Recommendations
    print("\n7. RECOMMENDATIONS:")
    print("="*60)
    
    # Find best strategy meeting requirements
    viable = df[(df['trades_per_day'] >= 2) & (df['avg_return_bps'] > 0.5)]
    
    if len(viable) > 0:
        best = viable.loc[viable['avg_return_bps'].idxmax()]
        print(f"\nBest strategy meeting requirements:")
        print(f"  Signal count: {best['signal_count']:.0f}")
        print(f"  Return: {best['avg_return_bps']:.2f} bps/trade")
        print(f"  Trades/day: {best['trades_per_day']:.1f}")
        print(f"  Annual return: {best['avg_return_bps'] * best['avg_trades'] / 100:.1f}%")
    else:
        print("\n⚠️  NO strategies meet both requirements (2+ trades/day AND >0.5 bps/trade)")
        
        # Relax requirements
        print("\nRelaxing requirements to 1.5+ trades/day:")
        viable_relaxed = df[(df['trades_per_day'] >= 1.5) & (df['avg_return_bps'] > 0.5)]
        
        if len(viable_relaxed) > 0:
            for _, row in viable_relaxed.head(3).iterrows():
                print(f"\n  Signal count {row['signal_count']:.0f}:")
                print(f"    Return: {row['avg_return_bps']:.2f} bps")
                print(f"    Trades/day: {row['trades_per_day']:.1f}")
                print(f"    Annual: {row['avg_return_bps'] * row['avg_trades'] / 100:.1f}%")

if __name__ == "__main__":
    analyze_filter_issues()