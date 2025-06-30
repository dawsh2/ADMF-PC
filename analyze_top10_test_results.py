#!/usr/bin/env python3
"""Analyze top 10 Keltner test results"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Results directory
results_dir = Path("config/keltner/test_top10/results/20250622_220133")
traces_dir = results_dir / "traces" / "mean_reversion"
metadata_file = results_dir / "metadata.json"

print("TOP 10 KELTNER STRATEGIES - TEST SET RESULTS")
print("="*80)

# Load metadata
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

# Analyze each strategy
results = []
trace_files = sorted(traces_dir.glob("*.parquet"))

for trace_file in trace_files:
    # Load trace
    df = pd.read_parquet(trace_file)
    
    # Get strategy info from metadata
    comp_name = trace_file.stem
    comp_info = metadata['components'].get(comp_name, {})
    
    # Extract strategy name and parameters
    strategy_name = comp_name.replace('SPY_5m_', '')
    
    # Get parameters directly from metadata
    params = comp_info.get('parameters', {})
    period = params.get('period')
    multiplier = params.get('multiplier')
    
    # Calculate performance
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
    total_bars = metadata['total_bars']
    trading_days = total_bars / 78
    
    if trade_returns:
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100
        avg_return = np.mean(trade_returns)
        total_return = sum(trade_returns)
        
        # Simple annualization
        annual_return = (total_return / trading_days) * 252 if trading_days > 0 else 0
        
        # Compound annualization
        cumulative = 1.0
        for r in trade_returns:
            cumulative *= (1 + r)
        
        years = trading_days / 252
        if years > 0 and cumulative > 0:
            compound_annual = ((cumulative ** (1/years)) - 1) * 100
        else:
            compound_annual = 0
            
        # Sharpe ratio approximation
        if len(trade_returns) > 1:
            daily_returns = [trade_returns[i] for i in range(len(trade_returns))]
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        else:
            sharpe = 0
    else:
        win_rate = avg_return = total_return = annual_return = compound_annual = sharpe = 0
    
    # Trades per day
    trades_per_day = trades / trading_days if trading_days > 0 else 0
    
    results.append({
        'strategy': strategy_name,
        'period': period,
        'multiplier': multiplier,
        'trades': trades,
        'trades_per_day': trades_per_day,
        'win_rate': win_rate,
        'avg_return_pct': avg_return * 100,
        'total_return_pct': total_return * 100,
        'annual_return_pct': annual_return * 100,
        'compound_annual_pct': compound_annual,
        'sharpe': sharpe
    })

# Sort by compound annual return
results = sorted(results, key=lambda x: x['compound_annual_pct'], reverse=True)

# Display results
print(f"{'Strategy':<20} {'P':<4} {'M':<5} {'Trades':<8} {'T/Day':<6} {'Win%':<6} {'Avg%':<8} {'Annual%':<10} {'Compound%':<10} {'Sharpe':<8}")
print("-"*110)

for r in results:
    print(f"{r['strategy']:<20} {r['period']:<4} {r['multiplier']:<5.1f} {r['trades']:<8} "
          f"{r['trades_per_day']:<6.2f} {r['win_rate']:<6.1f} {r['avg_return_pct']:<8.4f} "
          f"{r['annual_return_pct']:<10.2f} {r['compound_annual_pct']:<10.2f} {r['sharpe']:<8.2f}")

# Compare to training performance
print("\n" + "="*80)
print("TRAINING vs TEST COMPARISON:")
print("="*80)

training_perf = {
    (22, 0.5): 29.23,
    (21, 0.5): 28.80,
    (26, 0.5): 26.79,
    (23, 0.5): 24.65,
    (25, 0.5): 23.05,
    (27, 1.0): 12.81
}

print(f"{'Parameters':<15} {'Training%':<12} {'Test%':<12} {'Difference':<12} {'Degradation%':<12}")
print("-"*60)

for r in results:
    if r['period'] and r['multiplier']:
        key = (r['period'], r['multiplier'])
        if key in training_perf:
            train_perf = training_perf[key]
            test_perf = r['compound_annual_pct']
            diff = test_perf - train_perf
            degrade = (diff / train_perf) * 100 if train_perf != 0 else 0
            
            print(f"P={r['period']}, M={r['multiplier']:<3.1f} {train_perf:<12.2f} {test_perf:<12.2f} "
                  f"{diff:<12.2f} {degrade:<12.1f}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY:")
print("="*80)

if results:
    print(f"Best performer: {results[0]['strategy']} (P={results[0]['period']}, M={results[0]['multiplier']}) "
          f"with {results[0]['compound_annual_pct']:.2f}% annual return")
    
    # Check if any are profitable
    profitable = [r for r in results if r['compound_annual_pct'] > 0]
    print(f"Profitable strategies: {len(profitable)} out of {len(results)}")
    
    # Average performance by multiplier
    mult_groups = {}
    for r in results:
        if r['multiplier']:
            if r['multiplier'] not in mult_groups:
                mult_groups[r['multiplier']] = []
            mult_groups[r['multiplier']].append(r['compound_annual_pct'])
    
    print("\nAverage performance by multiplier:")
    for mult in sorted(mult_groups.keys()):
        avg_perf = np.mean(mult_groups[mult])
        print(f"  M={mult}: {avg_perf:.2f}%")

# Transaction cost analysis
print("\n" + "="*80)
print("TRANSACTION COST IMPACT:")
print("="*80)

commission_per_trade = 0.0001  # 1 basis point per trade (buy + sell = 2bp round trip)

for r in results[:5]:  # Top 5
    gross_return = r['compound_annual_pct']
    trades_per_year = r['trades_per_day'] * 252
    cost_per_year = trades_per_year * commission_per_trade * 100  # Convert to percentage
    net_return = gross_return - cost_per_year
    
    print(f"{r['strategy']}: {gross_return:.2f}% gross - {cost_per_year:.2f}% costs = {net_return:.2f}% net")