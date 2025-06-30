#!/usr/bin/env python3
"""Deep dive into the most promising patterns discovered."""

import pandas as pd
import numpy as np

# Load the data
trades_df = pd.read_csv('swing_zones_all_trades.csv')
TRADING_DAYS = 306
TOTAL_STRATEGIES = 600

print("=== ANALYZING MOST PROMISING PATTERNS ===\n")

# 1. Volume breakout pattern showed 5.52 bps
print("1. VOLUME BREAKOUT PATTERN")
volume_breakout = trades_df[(trades_df['volume_ratio'] > 2.0) & 
                           (trades_df['volatility_rank'] > 0.5)]

print(f"   Total trades: {len(volume_breakout)}")
print(f"   Trades per day per strategy: {len(volume_breakout)/TRADING_DAYS/TOTAL_STRATEGIES:.3f}")
print(f"   Average return: {volume_breakout['net_return_bps'].mean():.2f} bps")
print(f"   Win rate: {(volume_breakout['net_return'] > 0).mean()*100:.1f}%")
print(f"   Annual return (after 1bp cost): {((volume_breakout['net_return_bps'].mean()-1)/10000 * len(volume_breakout)/TRADING_DAYS/TOTAL_STRATEGIES * 252)*100:.2f}%")

# Break down by direction
print("\n   By direction:")
for direction in ['long', 'short']:
    subset = volume_breakout[volume_breakout['direction'] == direction]
    if len(subset) > 0:
        print(f"   {direction}: {len(subset)} trades, {subset['net_return_bps'].mean():.2f} bps")

# 2. Fade volatility spikes
print("\n2. FADE VOLATILITY SPIKES")
fade_vol = trades_df[(trades_df['volatility_rank'] > 0.9) & 
                     (trades_df['direction'] == 'short')]

print(f"   Total trades: {len(fade_vol)}")
print(f"   Trades per day per strategy: {len(fade_vol)/TRADING_DAYS/TOTAL_STRATEGIES:.3f}")
print(f"   Average return: {fade_vol['net_return_bps'].mean():.2f} bps")
print(f"   Win rate: {(fade_vol['net_return'] > 0).mean()*100:.1f}%")

# 3. Look for high-frequency positive patterns
print("\n3. SEARCHING FOR HIGH-FREQUENCY POSITIVE PATTERNS")

# Test various combinations
test_filters = {
    # Volume patterns
    "Volume 1.5-2.5x": (trades_df['volume_ratio'] > 1.5) & (trades_df['volume_ratio'] < 2.5),
    "Volume >2x + any vol": trades_df['volume_ratio'] > 2.0,
    
    # Short holding periods with conditions
    "Quick trades (<2 bars)": trades_df['bars_held'] <= 2,
    "Quick + high volume": (trades_df['bars_held'] <= 2) & (trades_df['volume_ratio'] > 1.5),
    "Quick + medium vol": (trades_df['bars_held'] <= 2) & (trades_df['volatility_rank'].between(0.4, 0.7)),
    
    # Time-based
    "Afternoon (14-16)": trades_df['hour'].between(14, 16),
    "Late day (18-19)": trades_df['hour'].between(18, 19),
    "Afternoon + volume": (trades_df['hour'].between(14, 16)) & (trades_df['volume_ratio'] > 1.2),
    
    # Price extremes
    "Near MA shorts": (trades_df['direction'] == 'short') & 
                      (trades_df['price_to_sma20'] > -0.01) & 
                      (trades_df['price_to_sma20'] < 0.01),
    
    # Moderate conditions (not extreme)
    "Moderate vol (40-70%)": trades_df['volatility_rank'].between(0.4, 0.7),
    "Moderate vol + volume": (trades_df['volatility_rank'].between(0.4, 0.7)) & 
                            (trades_df['volume_ratio'] > 1.2),
    
    # Session-based
    "Regular hours only": trades_df['session'] == 'regular',
    "Regular + volume": (trades_df['session'] == 'regular') & (trades_df['volume_ratio'] > 1.3),
}

results = []
print(f"\n{'Filter':<25} {'Trades':<8} {'Freq/Day':<10} {'Avg(bps)':<10} {'Annual%':<10}")
print("-" * 70)

for name, mask in test_filters.items():
    filtered = trades_df[mask]
    if len(filtered) > 1000:
        freq = len(filtered) / TRADING_DAYS / TOTAL_STRATEGIES
        avg_bps = filtered['net_return_bps'].mean()
        annual = ((avg_bps - 1) / 10000 * freq * 252) * 100
        
        results.append({
            'name': name,
            'trades': len(filtered),
            'freq': freq,
            'avg_bps': avg_bps,
            'annual': annual
        })
        
        print(f"{name:<25} {len(filtered):<8} {freq:<10.3f} {avg_bps:<10.2f} {annual:<10.2f}%")

# Sort by frequency while maintaining positive returns
print("\n4. BEST BALANCE OF FREQUENCY AND RETURN")
positive_results = [r for r in results if r['annual'] > 0]
positive_results.sort(key=lambda x: x['freq'], reverse=True)

if positive_results:
    print(f"\n{'Filter':<25} {'Freq/Day':<10} {'Annual%':<10}")
    print("-" * 45)
    for r in positive_results[:5]:
        print(f"{r['name']:<25} {r['freq']:<10.3f} {r['annual']:<10.2f}%")

# Analyze parameter relationships
print("\n5. PARAMETER OPTIMIZATION FOR FREQUENCY")

# Group by parameters
param_analysis = trades_df.groupby(['sr_period', 'entry_zone', 'exit_zone']).agg({
    'net_return_bps': ['count', 'mean'],
    'bars_held': 'mean'
}).round(2)

param_analysis.columns = ['trades', 'avg_bps', 'avg_bars']
param_analysis = param_analysis.reset_index()

# Find high frequency with positive returns
high_freq_positive = param_analysis[(param_analysis['trades'] > 5000) & 
                                   (param_analysis['avg_bps'] > 0)]

if len(high_freq_positive) > 0:
    print("\nParameter combinations with 5000+ trades AND positive returns:")
    print(high_freq_positive.sort_values('avg_bps', ascending=False).head(10))

# Test specific parameter ranges
print("\n6. OPTIMAL PARAMETER RANGES")

# Tighter zones = more signals?
tight_zones = trades_df[(trades_df['entry_zone'] <= 0.0015) & 
                        (trades_df['exit_zone'] <= 0.0015)]

wider_zones = trades_df[(trades_df['entry_zone'] >= 0.002) & 
                        (trades_df['exit_zone'] >= 0.002)]

print(f"\nTight zones (≤0.0015):")
print(f"  Trades: {len(tight_zones)} ({len(tight_zones)/TRADING_DAYS/TOTAL_STRATEGIES:.2f} per day)")
print(f"  Average: {tight_zones['net_return_bps'].mean():.2f} bps")

print(f"\nWider zones (≥0.002):")
print(f"  Trades: {len(wider_zones)} ({len(wider_zones)/TRADING_DAYS/TOTAL_STRATEGIES:.2f} per day)")
print(f"  Average: {wider_zones['net_return_bps'].mean():.2f} bps")

# Final recommendations
print("\n=== KEY INSIGHTS ===")
print("1. Volume breakouts (>2x) show best edge but low frequency")
print("2. Quick trades (<2 bars) provide higher frequency") 
print("3. Moderate volatility (40-70%) better than extremes")
print("4. Regular hours with volume filters show promise")
print("5. Consider asymmetric entry/exit zones for better frequency")