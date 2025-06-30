#!/usr/bin/env python3
"""
Analyze filter trade-offs between edge and frequency.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# Previous results from progressive_filter_relaxation.py
filter_results = [
    # (Filter Name, Trades, T/Day, Edge_bps, Win%)
    ("SMA>0.3% + Vol>85 (Original)", 10.0, 0.03, 11.20, 70.0),
    ("SMA>0.3% + Vol>80", 11.0, 0.04, 11.08, 72.7),
    ("SMA>0.3% + Vol>75", 13.0, 0.04, 9.04, 69.2),
    ("SMA>0.3% + Vol>70", 15.0, 0.05, 6.59, 60.0),
    ("SMA>0.25% + Vol>80", 17.0, 0.06, 6.70, 58.8),
    ("SMA>0.2% + Vol>85", 17.0, 0.06, 6.86, 58.8),
    ("SMA>0.2% + Vol>75", 21.0, 0.07, 5.53, 57.1),
    ("SMA>0.15% + Vol>70", 27.0, 0.09, 2.94, 48.1),
    
    ("Shorts + Vol>70 (Original)", 15.0, 0.05, 6.51, 60.0),
    ("Shorts + Vol>65", 15.0, 0.05, 6.51, 60.0),
    ("Shorts + Vol>60", 15.0, 0.05, 6.51, 60.0),
    ("Shorts + Vol>55", 16.0, 0.05, 7.23, 62.5),
    ("Shorts + Vol>50", 16.0, 0.05, 7.23, 62.5),
    
    ("Volume>1.2x (Original)", 8.0, 0.03, 9.42, 50.0),
    ("Volume>1.15x", 10.0, 0.03, 3.97, 40.0),
    ("Volume>1.1x", 13.0, 0.04, 2.35, 38.5),
    ("Volume>1.05x", 14.0, 0.05, 2.22, 42.9),
    ("Volume>1.1x + Vol>50", 11.0, 0.04, 3.07, 36.4),
    ("Volume>1.05x + Vol>60", 10.0, 0.03, 4.38, 40.0),
    
    ("Strict OR (SMA|Short|Vol)", 25.0, 0.08, 7.36, 60.0),
    ("Relaxed OR", 32.0, 0.10, 3.41, 50.0),
    ("Any: Vol>70 OR SMA>0.2% OR Vol>1.1x", 41.0, 0.13, 1.66, 48.8),
]

# Convert to DataFrame
df = pd.DataFrame(filter_results, columns=['Filter', 'Trades', 'TPD', 'Edge', 'WinRate'])

# Calculate expected returns
df['Daily_bps'] = df['Edge'] * df['TPD']
df['Annual_bps'] = df['Daily_bps'] * 252
df['Annual_pct'] = df['Annual_bps'] / 10000 * 100

# Sort by annual return
df = df.sort_values('Annual_pct', ascending=False)

print("="*100)
print("FILTER TRADE-OFF ANALYSIS: EDGE vs FREQUENCY")
print("="*100)
print("\nRanked by Expected Annual Return:")
print("-"*100)
print("Filter                              | Edge  | T/Day | Daily | Annual | Win%  | Risk/Reward")
print("------------------------------------|-------|-------|-------|--------|-------|------------")

for _, row in df.iterrows():
    # Risk/Reward assessment
    if row['Annual_pct'] > 5 and row['TPD'] > 0.1:
        risk_reward = "EXCELLENT"
    elif row['Annual_pct'] > 3 and row['TPD'] > 0.05:
        risk_reward = "GOOD"
    elif row['Annual_pct'] > 1:
        risk_reward = "MODERATE"
    else:
        risk_reward = "POOR"
    
    print(f"{row['Filter']:35s} | {row['Edge']:5.2f} | {row['TPD']:5.2f} | "
          f"{row['Daily_bps']:5.2f} | {row['Annual_pct']:6.2f}% | {row['WinRate']:5.1f} | {risk_reward}")

# Find optimal trade-offs
print("\n\nOPTIMAL TRADE-OFFS:")
print("="*80)

# Group 1: High edge, low frequency
high_edge = df[df['Edge'] > 5]
print("\n1. HIGH EDGE STRATEGIES (>5 bps per trade):")
for _, row in high_edge.head(5).iterrows():
    print(f"   {row['Filter']:35s}: {row['Edge']:5.2f} bps @ {row['TPD']:4.2f} t/day = {row['Annual_pct']:5.2f}% annual")

# Group 2: Balanced
balanced = df[(df['Edge'] > 2) & (df['TPD'] > 0.05)]
print("\n2. BALANCED STRATEGIES (>2 bps edge, >0.05 t/day):")
for _, row in balanced.head(5).iterrows():
    print(f"   {row['Filter']:35s}: {row['Edge']:5.2f} bps @ {row['TPD']:4.2f} t/day = {row['Annual_pct']:5.2f}% annual")

# Group 3: Higher frequency
higher_freq = df[df['TPD'] > 0.08]
print("\n3. HIGHER FREQUENCY (>0.08 t/day):")
for _, row in higher_freq.head(5).iterrows():
    print(f"   {row['Filter']:35s}: {row['Edge']:5.2f} bps @ {row['TPD']:4.2f} t/day = {row['Annual_pct']:5.2f}% annual")

# Calculate Sharpe estimates
print("\n\nSHARPE RATIO ESTIMATES:")
print("="*80)
print("(Assuming volatility scales with sqrt(trades))")

for _, row in df.head(10).iterrows():
    # Rough Sharpe estimate
    # Annual return / (volatility * sqrt(annual trades))
    # Assume per-trade vol of 50bps
    per_trade_vol = 50  # bps
    annual_trades = row['TPD'] * 252
    annual_vol = per_trade_vol * np.sqrt(annual_trades) / 100  # Convert to %
    
    if annual_vol > 0:
        sharpe = row['Annual_pct'] / annual_vol
        print(f"{row['Filter']:35s}: Sharpe ~{sharpe:5.2f}")

print("\n\nRECOMMENDATIONS BY RISK PROFILE:")
print("="*80)

print("\n1. CONSERVATIVE (Prioritize edge):")
print("   - SMA>0.3% + Vol>80: 11.08 bps edge, only 0.04 t/day")
print("   - Expected return: 1.11% annual")
print("   - Very selective, high win rate (72.7%)")

print("\n2. MODERATE (Balance edge and frequency):")
print("   - Strict OR filter: 7.36 bps @ 0.08 t/day")
print("   - Expected return: 1.48% annual")
print("   - Good win rate (60%)")

print("\n3. AGGRESSIVE (Prioritize frequency):")
print("   - Any: Vol>70 OR SMA>0.2% OR Vol>1.1x: 1.66 bps @ 0.13 t/day")
print("   - Expected return: 0.54% annual")
print("   - More trades but lower edge")

print("\n\nKEY INSIGHTS:")
print("="*80)
print("1. Relaxing filters too much kills profitability")
print("2. Sweet spot appears to be around Vol>75-80 for volatility filters")
print("3. SMA distance of 0.25-0.3% maintains good edge")
print("4. Combined OR filters can improve frequency while maintaining edge")
print("5. None achieve the target frequencies (2-3 t/day) from the config")
print("6. This confirms swing pivot bounce is NOT the source of those results")