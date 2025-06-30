#!/usr/bin/env python3
"""
Show performance breakdown of all strategy groups with long/short analysis.
"""

import pandas as pd
import numpy as np

def analyze_all_groups():
    print("=== COMPLETE STRATEGY GROUP PERFORMANCE BREAKDOWN ===\n")
    
    # Load filter group analysis
    df = pd.read_csv("keltner_filter_group_analysis.csv")
    
    # Add additional calculated columns
    df['trades_per_day'] = df['avg_trades'] / 252
    df['annual_return_gross'] = df['avg_return_bps'] * df['avg_trades'] / 100
    df['annual_return_net'] = (df['avg_return_bps'] - 0.5) * df['avg_trades'] / 100  # After 0.5 bps costs
    df['long_short_ratio'] = df['avg_long_bps'] / df['avg_short_bps'].replace(0, np.nan)
    df['long_edge'] = df['avg_long_bps'] > df['avg_short_bps'] * 2  # Strong long bias
    df['short_edge'] = df['avg_short_bps'] > df['avg_long_bps'] * 2  # Strong short bias
    
    # Sort by return per trade
    df = df.sort_values('avg_return_bps', ascending=False)
    
    # Main performance table
    print("STRATEGY GROUP PERFORMANCE (Sorted by Return per Trade):")
    print("="*140)
    print(f"{'Signals':<8} {'Trades':<8} {'T/Day':<6} {'RPT':<8} {'Net RPT':<8} {'Win%':<6} {'Annual%':<8} {'Net Ann%':<8} {'Long':<8} {'Short':<8} {'L/S':<6} {'Filter%':<8}")
    print("-"*140)
    
    for _, row in df.iterrows():
        ls_ratio = f"{row['long_short_ratio']:.1f}" if abs(row['long_short_ratio']) < 100 else "N/A"
        net_rpt = row['avg_return_bps'] - 0.5
        
        # Color coding for direction
        direction = ""
        if row['long_edge']:
            direction = "L"
        elif row['short_edge']:
            direction = "S"
            
        print(f"{row['signal_count']:<8.0f} {row['avg_trades']:<8.0f} {row['trades_per_day']:<6.1f} "
              f"{row['avg_return_bps']:<8.2f} {net_rpt:<8.2f} {row['avg_win_rate']*100:<6.1f} "
              f"{row['annual_return_gross']:<8.1f} {row['annual_return_net']:<8.1f} "
              f"{row['avg_long_bps']:<8.2f} {row['avg_short_bps']:<8.2f} {ls_ratio:<6} "
              f"{row['filter_reduction']:<8.1f} {direction}")
    
    # Detailed breakdowns
    print("\n" + "="*100)
    print("LONG vs SHORT DETAILED ANALYSIS:")
    print("="*100)
    print(f"{'Signals':<8} {'Long RPT':<10} {'Long Win%':<10} {'Short RPT':<10} {'Short Win%':<10} {'Best Side':<10} {'Recommendation':<30}")
    print("-"*100)
    
    for _, row in df.head(10).iterrows():
        long_net = row['avg_long_bps'] - 0.5
        short_net = row['avg_short_bps'] - 0.5
        
        best_side = "Long" if row['avg_long_bps'] > row['avg_short_bps'] else "Short"
        
        # Recommendation based on performance
        if long_net > 0 and short_net < 0:
            rec = "Long-only"
        elif short_net > 0 and long_net < 0:
            rec = "Short-only"
        elif long_net > 0 and short_net > 0:
            rec = "Both directions"
        elif row['avg_return_bps'] > 0:
            rec = "Both (marginal)"
        else:
            rec = "Avoid"
            
        print(f"{row['signal_count']:<8.0f} {row['avg_long_bps']:<10.2f} {row['long_win_rate']*100:<10.1f} "
              f"{row['avg_short_bps']:<10.2f} {row['short_win_rate']*100:<10.1f} {best_side:<10} {rec:<30}")
    
    # Filter effectiveness analysis
    print("\n" + "="*80)
    print("FILTER EFFECTIVENESS BY CATEGORY:")
    print("="*80)
    
    categories = [
        ("Heavy (>90%)", df[df['filter_reduction'] > 90]),
        ("Strong (70-90%)", df[(df['filter_reduction'] >= 70) & (df['filter_reduction'] < 90)]),
        ("Moderate (40-70%)", df[(df['filter_reduction'] >= 40) & (df['filter_reduction'] < 70)]),
        ("Light (10-40%)", df[(df['filter_reduction'] >= 10) & (df['filter_reduction'] < 40)]),
        ("Minimal (<10%)", df[df['filter_reduction'] < 10])
    ]
    
    print(f"{'Category':<20} {'Count':<8} {'Avg RPT':<10} {'Avg T/Day':<10} {'Avg Annual%':<12} {'Best RPT':<10}")
    print("-"*80)
    
    for name, group in categories:
        if len(group) > 0:
            print(f"{name:<20} {len(group):<8} {group['avg_return_bps'].mean():<10.2f} "
                  f"{group['trades_per_day'].mean():<10.1f} {group['annual_return_net'].mean():<12.1f} "
                  f"{group['avg_return_bps'].max():<10.2f}")
    
    # Trading frequency analysis
    print("\n" + "="*80)
    print("PERFORMANCE BY TRADING FREQUENCY:")
    print("="*80)
    
    freq_groups = [
        ("Ultra Low (<1/day)", df[df['trades_per_day'] < 1]),
        ("Low (1-2/day)", df[(df['trades_per_day'] >= 1) & (df['trades_per_day'] < 2)]),
        ("Medium (2-4/day)", df[(df['trades_per_day'] >= 2) & (df['trades_per_day'] < 4)]),
        ("High (4-6/day)", df[(df['trades_per_day'] >= 4) & (df['trades_per_day'] < 6)]),
        ("Very High (>6/day)", df[df['trades_per_day'] >= 6])
    ]
    
    print(f"{'Frequency':<20} {'Count':<8} {'Avg RPT':<10} {'Best RPT':<10} {'Avg Win%':<10} {'Avg Annual%':<12}")
    print("-"*80)
    
    for name, group in freq_groups:
        if len(group) > 0:
            print(f"{name:<20} {len(group):<8} {group['avg_return_bps'].mean():<10.2f} "
                  f"{group['avg_return_bps'].max():<10.2f} {group['avg_win_rate'].mean()*100:<10.1f} "
                  f"{group['annual_return_net'].mean():<12.1f}")
    
    # Best strategies for different objectives
    print("\n" + "="*80)
    print("BEST STRATEGIES FOR DIFFERENT OBJECTIVES:")
    print("="*80)
    
    # Best absolute return
    best_return = df.loc[df['avg_return_bps'].idxmax()]
    print(f"\n1. HIGHEST RETURN PER TRADE:")
    print(f"   Signals: {best_return['signal_count']:.0f}")
    print(f"   Return: {best_return['avg_return_bps']:.2f} bps/trade")
    print(f"   Frequency: {best_return['trades_per_day']:.1f} trades/day")
    print(f"   Annual: {best_return['annual_return_net']:.1f}% net")
    print(f"   Issue: Only {best_return['avg_trades']:.0f} trades total")
    
    # Best with minimum frequency
    min_freq = 2.0  # trades/day
    frequent = df[df['trades_per_day'] >= min_freq]
    if len(frequent) > 0:
        best_frequent = frequent.loc[frequent['avg_return_bps'].idxmax()]
        print(f"\n2. BEST WITH {min_freq}+ TRADES/DAY:")
        print(f"   Signals: {best_frequent['signal_count']:.0f}")
        print(f"   Return: {best_frequent['avg_return_bps']:.2f} bps/trade")
        print(f"   Frequency: {best_frequent['trades_per_day']:.1f} trades/day")
        print(f"   Annual: {best_frequent['annual_return_net']:.1f}% net")
    
    # Best annual return
    best_annual = df.loc[df['annual_return_net'].idxmax()]
    print(f"\n3. HIGHEST ANNUAL RETURN:")
    print(f"   Signals: {best_annual['signal_count']:.0f}")
    print(f"   Return: {best_annual['avg_return_bps']:.2f} bps/trade")
    print(f"   Frequency: {best_annual['trades_per_day']:.1f} trades/day")
    print(f"   Annual: {best_annual['annual_return_net']:.1f}% net")
    
    # Best Sharpe
    positive_sharpe = df[df['avg_sharpe'] > 0]
    if len(positive_sharpe) > 0:
        best_sharpe = positive_sharpe.loc[positive_sharpe['avg_sharpe'].idxmax()]
        print(f"\n4. BEST RISK-ADJUSTED (SHARPE):")
        print(f"   Signals: {best_sharpe['signal_count']:.0f}")
        print(f"   Return: {best_sharpe['avg_return_bps']:.2f} bps/trade")
        print(f"   Sharpe: {best_sharpe['avg_sharpe']:.3f}")
        print(f"   Annual: {best_sharpe['annual_return_net']:.1f}% net")
    
    # Best long-only candidate
    long_candidates = df[df['avg_long_bps'] > 0.5]
    if len(long_candidates) > 0:
        best_long = long_candidates.loc[long_candidates['avg_long_bps'].idxmax()]
        print(f"\n5. BEST LONG-ONLY CANDIDATE:")
        print(f"   Signals: {best_long['signal_count']:.0f}")
        print(f"   Long return: {best_long['avg_long_bps']:.2f} bps/trade")
        print(f"   vs Short: {best_long['avg_short_bps']:.2f} bps/trade")
        print(f"   Frequency: {best_long['trades_per_day']:.1f} trades/day")

if __name__ == "__main__":
    analyze_all_groups()