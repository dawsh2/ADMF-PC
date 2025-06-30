#!/usr/bin/env python3
"""Parameter sensitivity analysis for Keltner strategies"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Load all strategy results
results_dir = Path("config/keltner/results/20250622_180858")
metadata_path = results_dir / "metadata.json"
traces_dir = results_dir / "traces" / "keltner_bands"

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

print("KELTNER PARAMETER SENSITIVITY ANALYSIS")
print("="*80)

# Decode all strategies
def decode_strategy_id(strategy_id):
    base_id = strategy_id % 25
    period_idx = base_id % 5
    multiplier_idx = base_id // 5
    filter_id = strategy_id // 25
    
    periods = [10, 15, 20, 30, 50]
    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    return {
        'period': periods[period_idx],
        'multiplier': multipliers[multiplier_idx],
        'filter_id': filter_id
    }

# Analyze a subset of strategies for performance
results = []
strategies_to_analyze = range(0, 2750, 10)  # Every 10th strategy for speed

print(f"Analyzing {len(list(strategies_to_analyze))} strategies...")

for strategy_id in strategies_to_analyze:
    if strategy_id % 100 == 0:
        print(f"  Processing strategy {strategy_id}...")
    
    trace_file = traces_dir / f"SPY_5m_compiled_strategy_{strategy_id}.parquet"
    if not trace_file.exists():
        continue
        
    try:
        df = pd.read_parquet(trace_file)
        
        # Quick performance calculation
        trades = 0
        returns = []
        in_trade = False
        
        for i in range(len(df) - 1):
            signal = df.iloc[i]['val']
            if not in_trade and signal != 0:
                trades += 1
                in_trade = True
                entry = df.iloc[i]['px']
                direction = signal
            elif in_trade and signal == 0:
                in_trade = False
                exit = df.iloc[i]['px']
                if direction > 0:
                    ret = (exit - entry) / entry
                else:
                    ret = (entry - exit) / entry
                returns.append(ret)
        
        if returns:
            params = decode_strategy_id(strategy_id)
            total_return = sum(returns) * 100
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            
            # Annualized return (simple)
            trading_days = 213
            annual_return = (total_return / trading_days) * 252
            
            results.append({
                'strategy_id': strategy_id,
                'period': params['period'],
                'multiplier': params['multiplier'],
                'filter_id': params['filter_id'],
                'trades': trades,
                'total_return': total_return,
                'annual_return': annual_return,
                'win_rate': win_rate
            })
    except:
        continue

# Convert to DataFrame
df_results = pd.DataFrame(results)
print(f"\nAnalyzed {len(df_results)} strategies successfully")

# SENSITIVITY ANALYSIS
print("\n" + "="*80)
print("PARAMETER SENSITIVITY (No Filter - Base Parameters Only)")
print("="*80)

# Analyze base strategies (no filter)
base_strategies = df_results[df_results['filter_id'] == 0]

if len(base_strategies) > 0:
    # Create pivot table for heatmap
    pivot = base_strategies.pivot_table(
        values='annual_return', 
        index='multiplier', 
        columns='period'
    )
    
    print("\nAnnual Return by Period and Multiplier (%):")
    print(pivot.round(2))
    
    print("\n" + "-"*60)
    print("ROBUSTNESS ANALYSIS:")
    print("-"*60)
    
    # Find most stable parameters (low variance across nearby values)
    for period in [10, 15, 20, 30, 50]:
        period_data = base_strategies[base_strategies['period'] == period]
        if len(period_data) > 0:
            mean_return = period_data['annual_return'].mean()
            std_return = period_data['annual_return'].std()
            print(f"Period {period}: Mean={mean_return:6.2f}%, Std={std_return:6.2f}%")
    
    print("\nBy Multiplier:")
    for mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
        mult_data = base_strategies[base_strategies['multiplier'] == mult]
        if len(mult_data) > 0:
            mean_return = mult_data['annual_return'].mean()
            std_return = mult_data['annual_return'].std()
            print(f"Multiplier {mult}: Mean={mean_return:6.2f}%, Std={std_return:6.2f}%")

# Analyze filter effectiveness
print("\n" + "="*80)
print("FILTER EFFECTIVENESS ANALYSIS")
print("="*80)

filter_summary = df_results.groupby('filter_id').agg({
    'annual_return': ['mean', 'std', 'max', 'count'],
    'win_rate': 'mean'
}).round(2)

print("\nTop 10 Filters by Average Annual Return:")
top_filters = filter_summary.sort_values(('annual_return', 'mean'), ascending=False).head(10)
print(top_filters)

# Find robust parameter combinations
print("\n" + "="*80)
print("MOST ROBUST PARAMETER COMBINATIONS")
print("="*80)

# Group by base parameters and look at consistency across filters
param_groups = df_results.groupby(['period', 'multiplier']).agg({
    'annual_return': ['mean', 'std', 'min', 'max', 'count'],
    'win_rate': 'mean'
})

# Calculate robustness score (high mean, low std)
param_groups['robustness'] = (
    param_groups[('annual_return', 'mean')] / 
    (param_groups[('annual_return', 'std')] + 1)  # Add 1 to avoid division by zero
)

# Sort by robustness
robust_params = param_groups.sort_values('robustness', ascending=False).head(10)

print("\nTop 10 Most Robust Parameter Combinations:")
print("(High return with low variance across different filters)")
for (period, mult), row in robust_params.iterrows():
    mean_ret = row[('annual_return', 'mean')]
    std_ret = row[('annual_return', 'std')]
    min_ret = row[('annual_return', 'min')]
    max_ret = row[('annual_return', 'max')]
    count = row[('annual_return', 'count')]
    
    print(f"\nPeriod={period}, Multiplier={mult}:")
    print(f"  Mean Return: {mean_ret:.2f}% (±{std_ret:.2f}%)")
    print(f"  Range: {min_ret:.2f}% to {max_ret:.2f}%")
    print(f"  Tested on {count} filter variations")

# Save results
output_file = results_dir / "parameter_sensitivity_analysis.csv"
df_results.to_csv(output_file, index=False)
print(f"\nDetailed results saved to: {output_file}")