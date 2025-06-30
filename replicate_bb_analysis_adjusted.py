"""Replicate BB analysis with volatility thresholds based on actual data distribution"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load the trades we already calculated
trades_df = pd.read_csv('bb_trades_with_conditions_test.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

print("=== Analyzing trades with adjusted volatility thresholds ===")
print(f"Total trades: {len(trades_df)}")

# Recategorize volatility based on percentiles to match user's distribution
# User showed: low (majority), medium (poor performance), high (best performance)
# Let's use 33rd and 67th percentiles
atr_33 = trades_df['atr_pct'].quantile(0.33)
atr_67 = trades_df['atr_pct'].quantile(0.67)

print(f"\nVolatility thresholds based on data:")
print(f"Low: < {atr_33:.3f}%")
print(f"Medium: {atr_33:.3f}% - {atr_67:.3f}%")
print(f"High: > {atr_67:.3f}%")

# Recategorize volatility
trades_df['volatility_new'] = 'medium'
trades_df.loc[trades_df['atr_pct'] < atr_33, 'volatility_new'] = 'low'
trades_df.loc[trades_df['atr_pct'] > atr_67, 'volatility_new'] = 'high'

# Overall Performance
print(f"\n=== Overall Performance ===")
print(f"Average return: {trades_df['pnl_pct'].mean():.3f}%")
print(f"Win rate: {(trades_df['pnl_pct'] > 0).mean():.1%}")

# Performance by Volatility
print(f"\n=== Performance by Volatility Regime (Adjusted) ===")
for vol in ['low', 'medium', 'high']:
    vol_trades = trades_df[trades_df['volatility_new'] == vol]
    if len(vol_trades) > 0:
        print(f"{vol.capitalize()} volatility: {len(vol_trades)} trades, "
              f"avg: {vol_trades['pnl_pct'].mean():.3f}%, "
              f"win rate: {(vol_trades['pnl_pct'] > 0).mean():.1%}")

# Direction by Volatility
print(f"\n=== Direction Performance by Volatility ===")
for vol in ['low', 'medium', 'high']:
    vol_trades = trades_df[trades_df['volatility_new'] == vol]
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

# Performance by Trend
print(f"\n=== Performance by Trend Regime ===")
for trend in ['uptrend', 'downtrend', 'sideways']:
    trend_trades = trades_df[trades_df['trend'] == trend]
    if len(trend_trades) > 0:
        print(f"{trend.capitalize()}: {len(trend_trades)} trades, "
              f"avg: {trend_trades['pnl_pct'].mean():.3f}%, "
              f"win rate: {(trend_trades['pnl_pct'] > 0).mean():.1%}")

# Performance by VWAP Position
print(f"\n=== Performance by VWAP Position ===")
vwap_positions = ['far_below', 'below', 'near', 'above', 'far_above', 'unknown']
for vwap_pos in vwap_positions:
    vwap_trades = trades_df[trades_df['vwap_position'] == vwap_pos]
    if len(vwap_trades) > 0:
        print(f"{vwap_pos.replace('_', ' ').capitalize()}: {len(vwap_trades)} trades, "
              f"avg: {vwap_trades['pnl_pct'].mean():.3f}%, "
              f"win rate: {(vwap_trades['pnl_pct'] > 0).mean():.1%}")

# Best/Worst volatility conditions
print(f"\n=== Detailed Volatility Analysis ===")
for vol in ['low', 'medium', 'high']:
    vol_trades = trades_df[trades_df['volatility_new'] == vol]
    if len(vol_trades) > 5:
        print(f"\n{vol.capitalize()} volatility ({len(vol_trades)} trades):")
        print(f"  ATR% range: {vol_trades['atr_pct'].min():.3f}% - {vol_trades['atr_pct'].max():.3f}%")
        print(f"  Winners: {(vol_trades['pnl_pct'] > 0).sum()} trades, avg: {vol_trades[vol_trades['pnl_pct'] > 0]['pnl_pct'].mean():.3f}%")
        print(f"  Losers: {(vol_trades['pnl_pct'] < 0).sum()} trades, avg: {vol_trades[vol_trades['pnl_pct'] < 0]['pnl_pct'].mean():.3f}%")
        print(f"  Avg bars held: {vol_trades['bars_held'].mean():.1f}")

# Check if we can match user's numbers by using different thresholds
print(f"\n=== Testing Different Volatility Thresholds ===")
# Try percentile-based approach with different splits
for low_pct, high_pct in [(25, 75), (30, 70), (40, 60)]:
    atr_low = trades_df['atr_pct'].quantile(low_pct/100)
    atr_high = trades_df['atr_pct'].quantile(high_pct/100)
    
    # Recategorize
    trades_df['vol_test'] = 'medium'
    trades_df.loc[trades_df['atr_pct'] < atr_low, 'vol_test'] = 'low'
    trades_df.loc[trades_df['atr_pct'] > atr_high, 'vol_test'] = 'high'
    
    print(f"\nUsing {low_pct}th/{high_pct}th percentiles:")
    for vol in ['low', 'medium', 'high']:
        vol_trades = trades_df[trades_df['vol_test'] == vol]
        if len(vol_trades) > 0:
            print(f"  {vol}: {len(vol_trades)} trades, avg: {vol_trades['pnl_pct'].mean():.3f}%")