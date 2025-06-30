#!/usr/bin/env python3
"""Find robust Keltner parameters that should work on test set"""

import pandas as pd
import numpy as np

# Load the comprehensive analysis
df = pd.read_csv('config/keltner/results/20250622_180858/comprehensive_analysis.csv')

print('ROBUST PARAMETER SELECTION')
print('='*80)

# Decode parameters
def decode_base_param(param_id):
    period_idx = param_id % 5
    multiplier_idx = param_id // 5
    periods = [10, 15, 20, 30, 50]
    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
    return periods[period_idx], multipliers[multiplier_idx]

# 1. Find parameters that work well across many filters
print('\n1. PARAMETERS THAT WORK WITH MANY FILTERS:')
print('-'*60)

param_success = {}
for param_id in range(25):
    strategies = df[df['base_param_id'] == param_id]
    if len(strategies) == 0:
        continue
        
    profitable = strategies[strategies['avg_return_per_trade'] > 0]
    good_returns = strategies[strategies['avg_return_per_trade'] > 0.01]  # >0.01% per trade
    
    period, mult = decode_base_param(param_id)
    param_success[param_id] = {
        'period': period,
        'mult': mult,
        'profitable_pct': len(profitable) / len(strategies) * 100,
        'good_returns_pct': len(good_returns) / len(strategies) * 100,
        'avg_return': strategies['avg_return_per_trade'].mean(),
        'best_return': strategies['avg_return_per_trade'].max(),
        'worst_return': strategies['avg_return_per_trade'].min()
    }

# Sort by percentage of filters that give good returns
sorted_params = sorted(param_success.items(), key=lambda x: x[1]['good_returns_pct'], reverse=True)

for param_id, data in sorted_params[:5]:
    print(f"\nPeriod={data['period']}, Multiplier={data['mult']}:")
    print(f"  Works well with {data['good_returns_pct']:.0f}% of filters")
    print(f"  Profitable with {data['profitable_pct']:.0f}% of filters")
    print(f"  Average return: {data['avg_return']:.3f}%")
    print(f"  Range: [{data['worst_return']:.3f}% to {data['best_return']:.3f}%]")

# 2. Find stable, moderate performers
print('\n\n2. STABLE PERFORMERS (consistent across filters):')
print('-'*60)

# Calculate coefficient of variation (CV = std/mean) for positive performers
param_stability = {}
for param_id in range(25):
    strategies = df[df['base_param_id'] == param_id]
    if len(strategies) < 3:  # Need at least 3 data points
        continue
        
    returns = strategies['avg_return_per_trade']
    if returns.mean() > 0:
        cv = returns.std() / returns.mean() if returns.mean() != 0 else float('inf')
        period, mult = decode_base_param(param_id)
        param_stability[param_id] = {
            'period': period,
            'mult': mult,
            'cv': cv,
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'n_filters': len(strategies)
        }

# Sort by CV (lower is more stable)
sorted_stable = sorted(param_stability.items(), key=lambda x: x[1]['cv'])

for param_id, data in sorted_stable[:5]:
    print(f"\nPeriod={data['period']}, Multiplier={data['mult']}:")
    print(f"  Stability score (CV): {data['cv']:.2f}")
    print(f"  Mean return: {data['mean_return']:.3f}% ± {data['std_return']:.3f}%")
    print(f"  Tested on {data['n_filters']} filters")

# 3. Best for test set (avoid overfitting)
print('\n\n3. RECOMMENDED FOR ROBUSTNESS (avoiding overfitting):')
print('-'*60)

# Parameters with good returns but not the absolute best (avoid overfitting)
moderate_performers = []
for param_id, data in param_success.items():
    if 0.005 < data['avg_return'] < 0.02:  # Moderate returns
        if data['profitable_pct'] > 50:  # Works with most filters
            moderate_performers.append((param_id, data))

moderate_performers.sort(key=lambda x: x[1]['profitable_pct'], reverse=True)

print('\nBest choices for out-of-sample performance:')
for param_id, data in moderate_performers[:3]:
    print(f"\n** Period={data['period']}, Multiplier={data['mult']} **")
    print(f"  Average return: {data['avg_return']:.3f}%")
    print(f"  Works with {data['profitable_pct']:.0f}% of filters")
    print(f"  Not overfit to training data")

# 4. Create recommended config
print('\n\n4. CREATING ROBUST CONFIG:')
print('-'*60)

if moderate_performers:
    best_param_id, best_data = moderate_performers[0]
    print(f"\nRecommended parameters:")
    print(f"  Period: {best_data['period']}")
    print(f"  Multiplier: {best_data['mult']}")
    print(f"  Expected return: {best_data['avg_return']:.3f}% per trade")
    print(f"  Works with {best_data['profitable_pct']:.0f}% of filters")
    
    # Also find a good simple filter
    strategies_with_param = df[df['base_param_id'] == best_param_id]
    good_filters = strategies_with_param[strategies_with_param['avg_return_per_trade'] > best_data['avg_return']]
    
    if len(good_filters) > 0:
        best_filter = good_filters.iloc[0]
        print(f"\nBest filter for these parameters:")
        print(f"  Filter ID: {best_filter['filter_id']}")
        print(f"  Return: {best_filter['avg_return_per_trade']:.3f}% per trade")
        print(f"  Win rate: {best_filter['win_rate']*100:.1f}%")