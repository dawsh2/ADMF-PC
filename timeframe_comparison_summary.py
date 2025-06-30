#!/usr/bin/env python3
"""Create a comprehensive timeframe comparison summary."""

import pandas as pd
import numpy as np

# Results from our analysis
results = {
    '1m (estimated)': {
        'trades': 210,  # ~5x more than 15m
        'win_rate': 0.60,
        'annual_return': -0.05,  # Negative due to high costs
        'trade_frequency': 0.2,  # Hours per trade
        'note': 'Estimated based on 5x trade frequency vs 15m'
    },
    '5m_basic': {
        'trades': 11,
        'win_rate': 0.636,
        'annual_return': 0.0101,
        'trade_frequency': 0.9,
        'note': 'Only 11 trades - may be too selective'
    },
    '5m_tuned': {
        'trades': 80,
        'win_rate': 0.650,
        'annual_return': 0.0056,
        'trade_frequency': 1.0,
        'note': 'More trades but lower returns'
    },
    '15m_basic': {
        'trades': 42,
        'win_rate': 0.619,
        'annual_return': 0.0272,
        'trade_frequency': 3.3,
        'note': 'Best performer - good balance'
    },
    '15m_optimized': {
        'trades': 16,
        'win_rate': 0.562,
        'annual_return': -0.0038,
        'trade_frequency': 2.2,
        'note': 'Too selective - filtered good trades'
    }
}

print('=== COMPREHENSIVE TIMEFRAME ANALYSIS ===')
print('Execution cost: 0.5 bps (0.005%) per trade\n')

# Create summary table
print(f"{'Timeframe':<20} {'Trades/Year':<12} {'Win Rate':<10} {'Annual Return':<14} {'Hours/Trade':<12} {'Cost Impact':<12}")
print('-' * 90)

for name, data in results.items():
    trades_per_year = data['trades'] * (365 / 306)  # Annualize
    cost_impact = data['trades'] * 0.0001  # 1bp round trip
    
    print(f"{name:<20} {trades_per_year:<12.0f} {data['win_rate']*100:<10.1f}% "
          f"{data['annual_return']*100:<14.2f}% {data['trade_frequency']:<12.1f} "
          f"{cost_impact*100:<12.2f}%")

print('\n=== EXECUTION COST SENSITIVITY ===')
print('\nBreak-even cost per trade (bps):')

for name, data in results.items():
    if name == '1m (estimated)':
        continue
    
    # Calculate gross return before costs
    # annual_return = gross_return - (trades * cost)
    # If annual_return = 0, then cost = gross_return / trades
    
    # Estimate gross return
    gross_annual = data['annual_return'] + (data['trades'] * 0.0001)
    breakeven_cost_bps = (gross_annual / data['trades']) * 10000 if data['trades'] > 0 else 0
    
    print(f'{name}: {breakeven_cost_bps:.1f} bps')

print('\n=== OPTIMAL TIMEFRAME SELECTION ===')
print('\nBased on execution costs:')
print('- < 0.5 bps: Any timeframe works, but 15m basic is best')
print('- 0.5-2 bps: Use 15m basic (2.72% annual return)')
print('- 2-5 bps: Use 15m basic (still profitable)')
print('- > 5 bps: Need even longer timeframes or different strategy')

print('\n=== FINAL RECOMMENDATION ===')
print('\nFor your 0.5 bps execution cost:')
print('1. PRIMARY: Use 15m basic configuration')
print('   - 2.72% annual return after costs')
print('   - 42 trades per year')
print('   - 62% win rate')
print('   - Robust and simple')
print('\n2. ALTERNATIVE: Use 5m tuned if you want more action')
print('   - 0.56% annual return after costs')
print('   - 80 trades per year')
print('   - 65% win rate')
print('   - More opportunities but lower per-trade profit')

print('\n⚠️  Key Learning: Simple parameters (basic configs) outperformed')
print('   "optimized" versions - avoid over-fitting!')