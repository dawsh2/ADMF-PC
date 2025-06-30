#!/usr/bin/env python3
"""Analyze true robustness - consistent profits across different market conditions"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Load the comprehensive analysis from the original 2,750 strategy run
df = pd.read_csv('config/keltner/results/20250622_180858/comprehensive_analysis.csv')

print("TRUE ROBUSTNESS ANALYSIS - CONSISTENT PROFITABLE PERFORMANCE")
print("="*80)

# Decode parameters
def decode_base_param(param_id):
    period_idx = param_id % 5
    multiplier_idx = param_id // 5
    periods = [10, 15, 20, 30, 50]
    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
    return periods[period_idx], multipliers[multiplier_idx]

# Group by base parameters to analyze consistency
param_analysis = {}
for param_id in range(25):
    strategies = df[df['base_param_id'] == param_id]
    if len(strategies) < 5:  # Need enough data points
        continue
    
    period, mult = decode_base_param(param_id)
    
    # Calculate robustness metrics
    returns = strategies['avg_return_per_trade']
    win_rates = strategies['win_rate']
    
    # True robustness criteria:
    # 1. Consistently profitable (what % of filters give positive returns)
    # 2. Low downside (worst case scenario)
    # 3. Reasonable upside
    # 4. Consistent win rate
    # 5. Adequate trading frequency
    
    profitable_count = (returns > 0).sum()
    profitable_pct = profitable_count / len(strategies) * 100
    
    # Only consider if profitable in majority of cases
    if profitable_pct >= 60:
        param_analysis[param_id] = {
            'period': period,
            'multiplier': mult,
            'n_strategies': len(strategies),
            'profitable_pct': profitable_pct,
            'mean_return': returns.mean(),
            'median_return': returns.median(),
            'worst_return': returns.min(),
            'best_return': returns.max(),
            'return_std': returns.std(),
            'downside_risk': returns[returns < 0].mean() if (returns < 0).any() else 0,
            'mean_win_rate': win_rates.mean(),
            'win_rate_std': win_rates.std(),
            'mean_trades': strategies['trade_count'].mean(),
            'signal_freq': strategies['signal_frequency'].mean()
        }
        
        # Calculate Sortino ratio (return / downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            param_analysis[param_id]['sortino'] = returns.mean() / downside_std if downside_std > 0 else float('inf')
        else:
            param_analysis[param_id]['sortino'] = float('inf')
        
        # Calculate consistency score
        # High score = profitable most of the time + limited downside + consistent returns
        consistency_score = (
            profitable_pct / 100 * 0.4 +  # 40% weight on profitability rate
            (1 - abs(param_analysis[param_id]['worst_return']) / 0.1) * 0.3 +  # 30% weight on limited downside
            (1 - returns.std() / returns.mean() if returns.mean() > 0 else 0) * 0.3  # 30% weight on consistency
        )
        param_analysis[param_id]['consistency_score'] = consistency_score

print("\n1. PARAMETERS WITH CONSISTENT PROFITABILITY (>60% of filters profitable)")
print("-"*80)
print(f"{'Period':<8} {'Mult':<6} {'Profitable':<12} {'Mean Ret':<10} {'Worst':<10} {'Best':<10} {'Consistency':<12}")
print("-"*80)

# Sort by consistency score
sorted_params = sorted(param_analysis.items(), key=lambda x: x[1]['consistency_score'], reverse=True)

for param_id, data in sorted_params[:10]:
    print(f"{data['period']:<8} {data['multiplier']:<6.1f} {data['profitable_pct']:<11.0f}% "
          f"{data['mean_return']:<9.3f}% {data['worst_return']:<9.3f}% "
          f"{data['best_return']:<9.3f}% {data['consistency_score']:<11.2f}")

# Analyze robustness across different filter types
print("\n\n2. ROBUSTNESS ACROSS FILTER CATEGORIES")
print("-"*80)

# Group filters into categories (rough approximation based on filter ID patterns)
filter_categories = {
    'No Filter': [0],
    'RSI-based': list(range(1, 37)),
    'Volume-based': list(range(37, 55)),
    'Combined': list(range(55, 85)),
    'Other': list(range(85, 110))
}

for param_id, param_data in sorted_params[:5]:
    print(f"\nPeriod={param_data['period']}, Multiplier={param_data['multiplier']}:")
    
    for category, filter_ids in filter_categories.items():
        category_strategies = df[(df['base_param_id'] == param_id) & (df['filter_id'].isin(filter_ids))]
        if len(category_strategies) > 0:
            profitable = (category_strategies['avg_return_per_trade'] > 0).sum()
            avg_return = category_strategies['avg_return_per_trade'].mean()
            print(f"  {category:<15}: {profitable}/{len(category_strategies)} profitable, "
                  f"avg return: {avg_return:.3f}%")

# Find parameters that work well without filters
print("\n\n3. BEST PARAMETERS WITHOUT FILTERS (baseline performance)")
print("-"*80)

no_filter_strategies = df[df['filter_id'] == 0].copy()
no_filter_strategies['period'] = no_filter_strategies['base_param_id'].apply(lambda x: decode_base_param(x)[0])
no_filter_strategies['multiplier'] = no_filter_strategies['base_param_id'].apply(lambda x: decode_base_param(x)[1])

print(f"{'Period':<8} {'Mult':<6} {'Return':<10} {'Win Rate':<10} {'Trades':<10}")
print("-"*80)

for idx, row in no_filter_strategies.nlargest(5, 'avg_return_per_trade').iterrows():
    print(f"{row['period']:<8} {row['multiplier']:<6.1f} {row['avg_return_per_trade']:<9.3f}% "
          f"{row['win_rate']*100:<9.1f}% {row['trade_count']:<10.0f}")

# Stability analysis - which parameters maintain performance with any filter
print("\n\n4. MOST STABLE PARAMETERS (smallest performance range)")
print("-"*80)

stability_analysis = []
for param_id, param_data in param_analysis.items():
    if param_data['n_strategies'] >= 10:  # Need enough data
        performance_range = param_data['best_return'] - param_data['worst_return']
        stability_analysis.append({
            'param_id': param_id,
            'period': param_data['period'],
            'multiplier': param_data['multiplier'],
            'range': performance_range,
            'mean_return': param_data['mean_return'],
            'worst_return': param_data['worst_return'],
            'profitable_pct': param_data['profitable_pct']
        })

stability_df = pd.DataFrame(stability_analysis)
if len(stability_df) > 0:
    stability_df = stability_df[stability_df['mean_return'] > 0.005]  # Only positive performers
    stability_df = stability_df.sort_values('range')

print(f"{'Period':<8} {'Mult':<6} {'Return Range':<15} {'Mean Ret':<10} {'Worst':<10}")
print("-"*80)

for idx, row in stability_df.head(5).iterrows():
    print(f"{row['period']:<8} {row['multiplier']:<6.1f} {row['range']:<14.3f}% "
          f"{row['mean_return']:<9.3f}% {row['worst_return']:<9.3f}%")

# Final recommendations
print("\n\n5. RECOMMENDED ROBUST PARAMETERS")
print("="*80)

# Find parameters that score well on multiple criteria
recommendations = []

for param_id, param_data in param_analysis.items():
    if (param_data['profitable_pct'] >= 70 and  # Profitable with most filters
        param_data['worst_return'] > -0.01 and   # Limited downside
        param_data['mean_return'] > 0.005 and    # Decent returns
        param_data['mean_trades'] > 100):        # Adequate trading frequency
        
        recommendations.append({
            'period': param_data['period'],
            'multiplier': param_data['multiplier'],
            'score': param_data['consistency_score'],
            'profitable_pct': param_data['profitable_pct'],
            'mean_return': param_data['mean_return'],
            'worst_return': param_data['worst_return'],
            'mean_trades': param_data['mean_trades']
        })

recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)

if recommendations:
    print("\nTOP 3 TRULY ROBUST PARAMETER COMBINATIONS:")
    for i, rec in enumerate(recommendations[:3]):
        print(f"\n{i+1}. Period={rec['period']}, Multiplier={rec['multiplier']}")
        print(f"   - Profitable with {rec['profitable_pct']:.0f}% of filters")
        print(f"   - Average return: {rec['mean_return']:.3f}% per trade")
        print(f"   - Worst case: {rec['worst_return']:.3f}% per trade")
        print(f"   - Average trades: {rec['mean_trades']:.0f}")
        print(f"   - Consistency score: {rec['score']:.2f}")
else:
    print("\nNo parameters meet all robustness criteria.")
    print("Consider relaxing constraints or using ensemble approach.")

print("\n" + "="*80)
print("CONCLUSION:")
print("True robustness = Consistent profitability across different market conditions")
print("NOT just compatibility with many filters!")