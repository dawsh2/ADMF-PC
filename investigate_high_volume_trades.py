"""Investigate the profitable high-volume trades we're filtering out"""
import pandas as pd
import numpy as np

# Load trades
trades_df = pd.read_csv('bb_trades_fresh_indicators.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

# Get VWAP filtered trades (good trades)
longs = trades_df[trades_df['direction'] == 'long']
shorts = trades_df[trades_df['direction'] == 'short']
shorts_above_vwap = shorts[shorts['vwap_position'].isin(['above', 'far_above'])]
vwap_filtered = pd.concat([longs, shorts_above_vwap])

# High volume trades from VWAP filtered set
high_vol_good_trades = vwap_filtered[vwap_filtered['volume_ratio'] > 1.2]

print("=== Investigation: Profitable High-Volume Trades ===\n")
print(f"Found {len(high_vol_good_trades)} high-volume trades that are profitable")
print(f"Average return: {high_vol_good_trades['pnl_pct'].mean():.3f}%")
print(f"Total PnL contribution: {high_vol_good_trades['pnl_pct'].sum():.2f}%")

# Analyze characteristics
print(f"\n=== Characteristics of These Trades ===")

# By direction
hv_longs = high_vol_good_trades[high_vol_good_trades['direction'] == 'long']
hv_shorts = high_vol_good_trades[high_vol_good_trades['direction'] == 'short']

print(f"\nDirection breakdown:")
print(f"Longs: {len(hv_longs)} trades, avg: {hv_longs['pnl_pct'].mean():.3f}%, win rate: {(hv_longs['pnl_pct'] > 0).mean():.1%}")
print(f"Shorts: {len(hv_shorts)} trades, avg: {hv_shorts['pnl_pct'].mean():.3f}%, win rate: {(hv_shorts['pnl_pct'] > 0).mean():.1%}")

# By trend
print(f"\nTrend breakdown:")
for trend in ['uptrend', 'downtrend', 'sideways']:
    trend_trades = high_vol_good_trades[high_vol_good_trades['trend'] == trend]
    if len(trend_trades) > 0:
        print(f"{trend}: {len(trend_trades)} trades, avg: {trend_trades['pnl_pct'].mean():.3f}%")

# By time of day
print(f"\nTime of day:")
high_vol_good_trades['hour'] = high_vol_good_trades['entry_time'].dt.hour
hour_stats = high_vol_good_trades.groupby('hour').agg({
    'pnl_pct': ['count', 'mean']
}).round(3)
hour_stats.columns = ['count', 'avg_return']
print(hour_stats[hour_stats['count'] >= 2])

# Volume distribution
print(f"\nVolume ratio distribution:")
print(f"Min: {high_vol_good_trades['volume_ratio'].min():.2f}")
print(f"25th: {high_vol_good_trades['volume_ratio'].quantile(0.25):.2f}")
print(f"Median: {high_vol_good_trades['volume_ratio'].median():.2f}")
print(f"75th: {high_vol_good_trades['volume_ratio'].quantile(0.75):.2f}")
print(f"Max: {high_vol_good_trades['volume_ratio'].max():.2f}")

# Look for patterns in winners vs losers
hv_winners = high_vol_good_trades[high_vol_good_trades['pnl_pct'] > 0]
hv_losers = high_vol_good_trades[high_vol_good_trades['pnl_pct'] < 0]

print(f"\n=== Winners vs Losers in High Volume ===")
print(f"Winners: {len(hv_winners)} trades")
print(f"  Avg return: {hv_winners['pnl_pct'].mean():.3f}%")
print(f"  Avg volume ratio: {hv_winners['volume_ratio'].mean():.2f}")
print(f"  Avg bars held: {hv_winners['bars_held'].mean():.1f}")

print(f"\nLosers: {len(hv_losers)} trades")
print(f"  Avg return: {hv_losers['pnl_pct'].mean():.3f}%")
print(f"  Avg volume ratio: {hv_losers['volume_ratio'].mean():.2f}")
print(f"  Avg bars held: {hv_losers['bars_held'].mean():.1f}")

# Test different volume thresholds
print(f"\n=== Testing Different Volume Thresholds ===")
for threshold in [1.5, 2.0, 2.5, 3.0]:
    very_high_vol = vwap_filtered[vwap_filtered['volume_ratio'] > threshold]
    moderate_vol = vwap_filtered[(vwap_filtered['volume_ratio'] > 1.2) & (vwap_filtered['volume_ratio'] <= threshold)]
    
    if len(very_high_vol) > 5:
        print(f"\nVolume > {threshold}x: {len(very_high_vol)} trades, avg: {very_high_vol['pnl_pct'].mean():.3f}%")
        print(f"Volume 1.2-{threshold}x: {len(moderate_vol)} trades, avg: {moderate_vol['pnl_pct'].mean():.3f}%")

# Hypothesis: Maybe high volume mean reversion works differently
print(f"\n=== High Volume Mean Reversion Hypothesis ===")
print("Perhaps high volume creates stronger mean reversion moves?")

# Check if high volume trades have faster exits
all_vwap_filtered = vwap_filtered.copy()
all_vwap_filtered['volume_category'] = 'normal'
all_vwap_filtered.loc[all_vwap_filtered['volume_ratio'] > 1.2, 'volume_category'] = 'high'
all_vwap_filtered.loc[all_vwap_filtered['volume_ratio'] < 0.8, 'volume_category'] = 'low'

print(f"\nAverage bars held by volume category:")
for cat in ['low', 'normal', 'high']:
    cat_trades = all_vwap_filtered[all_vwap_filtered['volume_category'] == cat]
    if len(cat_trades) > 0:
        winners = cat_trades[cat_trades['pnl_pct'] > 0]
        print(f"{cat}: {winners['bars_held'].mean():.1f} bars for winners")

# Final recommendation
print(f"\n=== Recommendation ===")
print("The data shows high-volume trades after VWAP filtering are PROFITABLE!")
print("Options:")
print("1. Don't filter by volume at all - keep these profitable trades")
print("2. Use a higher threshold (e.g., >2.0x instead of >1.2x)")
print("3. Filter only high-volume SHORTS (they perform worse)")

# Test option 3
hv_filtered_refined = vwap_filtered.copy()
# Remove only high volume shorts
hv_shorts_to_remove = (hv_filtered_refined['direction'] == 'short') & (hv_filtered_refined['volume_ratio'] > 1.2)
refined_filtered = hv_filtered_refined[~hv_shorts_to_remove]

print(f"\n=== Refined Filter: Remove High-Volume Shorts Only ===")
print(f"Trades: {len(refined_filtered)} (vs {len(high_vol_good_trades)} removed completely)")
print(f"Average return: {refined_filtered['pnl_pct'].mean():.3f}%")
print(f"This keeps {len(hv_longs)} profitable high-volume longs!")