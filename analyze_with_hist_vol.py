"""Analyze using historical volatility instead of ATR"""
import pandas as pd
import numpy as np

# Load the trades with fresh indicators
trades_df = pd.read_csv('bb_trades_fresh_indicators.csv')

print("=== Using Historical Volatility for Categorization ===")

# Check hist_vol distribution
print(f"\nHistorical volatility stats:")
print(f"Mean: {trades_df['hist_vol'].mean():.1f}%")
print(f"Std: {trades_df['hist_vol'].std():.1f}%")
print(f"Min: {trades_df['hist_vol'].min():.1f}%")
print(f"Max: {trades_df['hist_vol'].max():.1f}%")

# Use historical volatility percentiles
hist_vol_33 = trades_df['hist_vol'].quantile(0.33)
hist_vol_67 = trades_df['hist_vol'].quantile(0.67)

print(f"\nVolatility thresholds (historical vol):")
print(f"Low: < {hist_vol_33:.1f}%")
print(f"Medium: {hist_vol_33:.1f}% - {hist_vol_67:.1f}%")
print(f"High: > {hist_vol_67:.1f}%")

# Categorize by historical volatility
trades_df['vol_hist'] = 'medium'
trades_df.loc[trades_df['hist_vol'] < hist_vol_33, 'vol_hist'] = 'low'
trades_df.loc[trades_df['hist_vol'] > hist_vol_67, 'vol_hist'] = 'high'

# Performance by Historical Volatility
print(f"\n=== Performance by Historical Volatility ===")
for vol in ['low', 'medium', 'high']:
    vol_trades = trades_df[trades_df['vol_hist'] == vol]
    if len(vol_trades) > 0:
        print(f"{vol.capitalize()} volatility: {len(vol_trades)} trades, "
              f"avg: {vol_trades['pnl_pct'].mean():.3f}%, "
              f"win rate: {(vol_trades['pnl_pct'] > 0).mean():.1%}")

# Direction performance by volatility
print(f"\n=== Direction Performance by Historical Volatility ===")
for vol in ['low', 'medium', 'high']:
    vol_trades = trades_df[trades_df['vol_hist'] == vol]
    if len(vol_trades) > 0:
        # Longs
        vol_longs = vol_trades[vol_trades['direction'] == 'long']
        if len(vol_longs) > 0:
            print(f"{vol.capitalize()} vol - Longs: {len(vol_longs)} trades, "
                  f"avg: {vol_longs['pnl_pct'].mean():.3f}%, "
                  f"win rate: {(vol_longs['pnl_pct'] > 0).mean():.1%}")
        # Shorts
        vol_shorts = vol_trades[vol_trades['direction'] == 'short']
        if len(vol_shorts) > 0:
            print(f"{vol.capitalize()} vol - Shorts: {len(vol_shorts)} trades, "
                  f"avg: {vol_shorts['pnl_pct'].mean():.3f}%, "
                  f"win rate: {(vol_shorts['pnl_pct'] > 0).mean():.1%}")

# Try absolute thresholds that might match user's data
print(f"\n=== Testing Fixed Volatility Thresholds ===")

# Common market volatility levels (annualized)
test_thresholds = [
    (10, 15),  # Very low vol market
    (12, 18),  # Low vol market  
    (15, 25),  # Normal market
    (20, 30),  # Elevated vol
]

for low_thresh, high_thresh in test_thresholds:
    trades_df['vol_fixed'] = 'medium'
    trades_df.loc[trades_df['hist_vol'] < low_thresh, 'vol_fixed'] = 'low'
    trades_df.loc[trades_df['hist_vol'] > high_thresh, 'vol_fixed'] = 'high'
    
    print(f"\nUsing thresholds: Low < {low_thresh}%, High > {high_thresh}%")
    for vol in ['low', 'medium', 'high']:
        vol_trades = trades_df[trades_df['vol_fixed'] == vol]
        if len(vol_trades) > 0:
            print(f"  {vol}: {len(vol_trades)} trades, avg: {vol_trades['pnl_pct'].mean():.3f}%, "
                  f"win rate: {(vol_trades['pnl_pct'] > 0).mean():.1%}")

# Market condition combinations with historical vol
print(f"\n=== Best Combinations (Historical Vol) ===")
condition_groups = trades_df.groupby(['vol_hist', 'trend', 'vwap_position'])
results = []
for (vol, trend, vwap), group in condition_groups:
    if len(group) >= 3:
        avg_return = group['pnl_pct'].mean()
        win_rate = (group['pnl_pct'] > 0).mean()
        results.append({
            'conditions': f"{vol} vol + {trend} + {vwap}",
            'trades': len(group),
            'avg_return': avg_return,
            'win_rate': win_rate
        })

results_df = pd.DataFrame(results).sort_values('avg_return', ascending=False)
for _, row in results_df.head(10).iterrows():
    print(f"{row['conditions']}: {row['trades']} trades, "
          f"avg: {row['avg_return']:.3f}%, win rate: {row['win_rate']:.1%}")

# Check if certain hours show the patterns
print(f"\n=== Volatility Performance by Hour ===")
for hour in sorted(trades_df['hour'].unique()):
    hour_trades = trades_df[trades_df['hour'] == hour]
    if len(hour_trades) >= 10:
        # Group by volatility for this hour
        print(f"\nHour {hour}:")
        for vol in ['low', 'medium', 'high']:
            vol_trades = hour_trades[hour_trades['vol_hist'] == vol]
            if len(vol_trades) > 0:
                print(f"  {vol}: {len(vol_trades)} trades, avg: {vol_trades['pnl_pct'].mean():.3f}%")