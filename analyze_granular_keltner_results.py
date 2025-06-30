#!/usr/bin/env python3
"""Analyze results from granular Keltner parameter sweep"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# Results directory
results_dir = Path("config/keltner/results/20250622_215020")
traces_dir = results_dir / "traces" / "keltner_bands"
metadata_file = results_dir / "metadata.json"

print("GRANULAR KELTNER PARAMETER SWEEP ANALYSIS")
print("="*80)

# Load metadata
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

# Analyze all strategies
results = []
trace_files = sorted(traces_dir.glob("*.parquet"))
print(f"Found {len(trace_files)} trace files")

for trace_file in trace_files:
    # Extract strategy number from filename
    filename = trace_file.stem  # e.g., "SPY_5m_compiled_strategy_41"
    strategy_num = int(filename.split('_')[-1])
        
    # Load trace
    df = pd.read_parquet(trace_file)
    
    # Get strategy info from metadata
    comp_name = filename  # Use the actual filename
    comp_info = metadata['components'].get(comp_name, {})
    
    # Extract parameters - they're directly in the parameters field
    params = comp_info.get('parameters', {})
    period = params.get('period')
    multiplier = params.get('multiplier')
    
    # Quick performance calc
    trades = 0
    trade_returns = []
    in_trade = False
    
    for idx in range(len(df)):
        signal = df.iloc[idx]['val']
        price = df.iloc[idx]['px']
        
        if not in_trade and signal != 0:
            trades += 1
            in_trade = True
            entry_price = price
            trade_direction = signal
        elif in_trade and signal == 0:
            in_trade = False
            if 'entry_price' in locals():
                if trade_direction > 0:
                    ret = (price - entry_price) / entry_price
                else:
                    ret = (entry_price - price) / entry_price
                trade_returns.append(ret)
    
    # Calculate metrics
    if trade_returns:
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100
        avg_return = np.mean(trade_returns)
        total_return = sum(trade_returns)
        
        # Simple annualization
        total_bars = metadata['total_bars']
        years = (total_bars / 78) / 252
        annual_return = (total_return / years) if years > 0 else 0
        
        # Compound annualization
        cumulative = 1.0
        for r in trade_returns:
            cumulative *= (1 + r)
        
        if years > 0 and cumulative > 0:
            compound_annual = ((cumulative ** (1/years)) - 1) * 100
        else:
            compound_annual = 0
    else:
        win_rate = avg_return = total_return = annual_return = compound_annual = 0
    
    results.append({
        'strategy_id': strategy_num,
        'period': period,
        'multiplier': multiplier,
        'trades': trades,
        'win_rate': win_rate,
        'avg_return_pct': avg_return * 100,
        'total_return_pct': total_return * 100,
        'annual_return_pct': annual_return * 100,
        'compound_annual_pct': compound_annual
    })

# Convert to DataFrame for analysis
df_results = pd.DataFrame(results)

# Remove any strategies with missing parameters
df_results = df_results.dropna(subset=['period', 'multiplier'])

print(f"Analyzed {len(df_results)} strategies with valid parameters")
print(f"Period range: {df_results['period'].min()} to {df_results['period'].max()}")
print(f"Multiplier range: {df_results['multiplier'].min()} to {df_results['multiplier'].max()}")

# Find top performers
print("\nTOP 20 PERFORMERS (by compound annual return):")
print("-"*80)
top_20 = df_results.nlargest(20, 'compound_annual_pct')
print(f"{'Rank':<6} {'Period':<8} {'Mult':<6} {'Trades':<8} {'Win%':<8} {'Avg%':<10} {'Annual%':<10} {'Compound%':<10}")
print("-"*80)

for idx, row in enumerate(top_20.itertuples(), 1):
    print(f"{idx:<6} {int(row.period):<8} {row.multiplier:<6.1f} {row.trades:<8} "
          f"{row.win_rate:<8.1f} {row.avg_return_pct:<10.4f} "
          f"{row.annual_return_pct:<10.2f} {row.compound_annual_pct:<10.2f}")

# Analyze by parameter
print("\nPARAMETER ANALYSIS:")
print("="*80)

# Best periods (aggregate across all multipliers)
period_perf = df_results.groupby('period').agg({
    'compound_annual_pct': ['mean', 'std', 'max'],
    'trades': 'mean',
    'win_rate': 'mean'
}).round(2)

print("\nBest Periods (by average compound return across all multipliers):")
best_periods = period_perf.sort_values(('compound_annual_pct', 'mean'), ascending=False).head(10)
print(best_periods)

# Best multipliers (aggregate across all periods)
mult_perf = df_results.groupby('multiplier').agg({
    'compound_annual_pct': ['mean', 'std', 'max'],
    'trades': 'mean',
    'win_rate': 'mean'
}).round(2)

print("\nBest Multipliers (by average compound return across all periods):")
best_mults = mult_perf.sort_values(('compound_annual_pct', 'mean'), ascending=False)
print(best_mults)

# Create heatmap data
print("\nCREATING PERFORMANCE HEATMAP...")
heatmap = df_results.pivot_table(
    values='compound_annual_pct',
    index='period',
    columns='multiplier',
    aggfunc='first'
)

# Find sweet spots
print("\nSWEET SPOTS (consistent high performance regions):")
print("-"*80)

# Look for regions with consistently good performance
for period in [10, 15, 20, 23, 25, 30, 35, 40, 45, 50]:
    if period in heatmap.index:
        row_data = heatmap.loc[period]
        best_mult = row_data.idxmax()
        best_return = row_data.max()
        avg_return = row_data.mean()
        print(f"Period {period}: Best at M={best_mult:.1f} ({best_return:.2f}%), Avg across multipliers: {avg_return:.2f}%")

# Check our known good parameters
print("\nKNOWN GOOD PARAMETERS:")
print("-"*80)
known_good = [
    (20, 3.0, "Robust in training"),
    (23, 3.0, "Best on test set"),
    (10, 3.0, "Previous analysis"),
    (15, 3.0, "Previous analysis"),
    (50, 1.0, "Top training performer")
]

for period, mult, desc in known_good:
    match = df_results[(df_results['period'] == period) & (df_results['multiplier'] == mult)]
    if not match.empty:
        row = match.iloc[0]
        print(f"P={period}, M={mult} ({desc}): "
              f"{row['compound_annual_pct']:.2f}% annual, "
              f"{row['trades']} trades, "
              f"{row['win_rate']:.1f}% win rate")

# Save detailed results
df_results.to_csv("keltner_granular_results.csv", index=False)
print(f"\nDetailed results saved to keltner_granular_results.csv")

# Parameter sensitivity around P=23, M=3.0
print("\nPARAMETER SENSITIVITY AROUND P=23, M=3.0:")
print("-"*80)
for p in range(20, 27):
    for m in [2.5, 3.0, 3.5]:
        match = df_results[(df_results['period'] == p) & (df_results['multiplier'] == m)]
        if not match.empty:
            row = match.iloc[0]
            print(f"P={p}, M={m}: {row['compound_annual_pct']:>6.2f}%")