"""Analyze VWAP, volume patterns and test short filters"""
import pandas as pd
import numpy as np

# Load trades
trades_df = pd.read_csv('bb_trades_sensitive_trend.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

print("=== VWAP Analysis ===")
# Overall VWAP performance
for vwap_pos in ['far_below', 'below', 'near', 'above', 'far_above']:
    vwap_trades = trades_df[trades_df['vwap_position'] == vwap_pos]
    if len(vwap_trades) > 0:
        print(f"\n{vwap_pos.replace('_', ' ').capitalize()}: {len(vwap_trades)} trades")
        print(f"  Average return: {vwap_trades['pnl_pct'].mean():.3f}%")
        print(f"  Win rate: {(vwap_trades['pnl_pct'] > 0).mean():.1%}")
        
        # By direction
        vwap_longs = vwap_trades[vwap_trades['direction'] == 'long']
        vwap_shorts = vwap_trades[vwap_trades['direction'] == 'short']
        if len(vwap_longs) > 0:
            print(f"  Longs: {len(vwap_longs)} trades, avg: {vwap_longs['pnl_pct'].mean():.3f}%")
        if len(vwap_shorts) > 0:
            print(f"  Shorts: {len(vwap_shorts)} trades, avg: {vwap_shorts['pnl_pct'].mean():.3f}%")

print("\n=== Volume Analysis ===")
# Volume ratio distribution
print(f"Volume ratio stats:")
print(f"  Mean: {trades_df['volume_ratio'].mean():.2f}")
print(f"  Std: {trades_df['volume_ratio'].std():.2f}")
print(f"  25th percentile: {trades_df['volume_ratio'].quantile(0.25):.2f}")
print(f"  75th percentile: {trades_df['volume_ratio'].quantile(0.75):.2f}")

# Performance by volume
trades_df['volume_category'] = 'normal'
trades_df.loc[trades_df['volume_ratio'] < 0.8, 'volume_category'] = 'low'
trades_df.loc[trades_df['volume_ratio'] > 1.2, 'volume_category'] = 'high'

print("\nPerformance by volume:")
for vol_cat in ['low', 'normal', 'high']:
    vol_trades = trades_df[trades_df['volume_category'] == vol_cat]
    if len(vol_trades) > 0:
        print(f"\n{vol_cat.capitalize()} volume: {len(vol_trades)} trades")
        print(f"  Average return: {vol_trades['pnl_pct'].mean():.3f}%")
        print(f"  Win rate: {(vol_trades['pnl_pct'] > 0).mean():.1%}")
        
        # By direction
        vol_longs = vol_trades[vol_trades['direction'] == 'long']
        vol_shorts = vol_trades[vol_trades['direction'] == 'short']
        if len(vol_longs) > 0:
            print(f"  Longs: {len(vol_longs)} trades, avg: {vol_longs['pnl_pct'].mean():.3f}%")
        if len(vol_shorts) > 0:
            print(f"  Shorts: {len(vol_shorts)} trades, avg: {vol_shorts['pnl_pct'].mean():.3f}%")

# Check increasing volume pattern
trades_df['volume_increasing'] = trades_df['volume_ratio'] > 1.0

print("\n=== Volume Trend Analysis ===")
for increasing in [True, False]:
    label = "Increasing" if increasing else "Decreasing"
    vol_trend_trades = trades_df[trades_df['volume_increasing'] == increasing]
    print(f"\n{label} volume: {len(vol_trend_trades)} trades")
    print(f"  Average return: {vol_trend_trades['pnl_pct'].mean():.3f}%")
    print(f"  Win rate: {(vol_trend_trades['pnl_pct'] > 0).mean():.1%}")

# Test "Short only above VWAP" filter
print("\n=== Testing 'Short Only Above VWAP' Filter ===")

# Current short performance
shorts = trades_df[trades_df['direction'] == 'short']
print(f"\nCurrent short performance:")
print(f"  Total shorts: {len(shorts)}")
print(f"  Average return: {shorts['pnl_pct'].mean():.3f}%")
print(f"  Win rate: {(shorts['pnl_pct'] > 0).mean():.1%}")

# Filtered shorts (above or far_above VWAP)
filtered_shorts = shorts[shorts['vwap_position'].isin(['above', 'far_above'])]
print(f"\nShorts above VWAP only:")
print(f"  Total shorts: {len(filtered_shorts)} ({len(filtered_shorts)/len(shorts)*100:.1f}% of original)")
print(f"  Average return: {filtered_shorts['pnl_pct'].mean():.3f}%")
print(f"  Win rate: {(filtered_shorts['pnl_pct'] > 0).mean():.1%}")

# Eliminated shorts
eliminated_shorts = shorts[~shorts['vwap_position'].isin(['above', 'far_above'])]
print(f"\nEliminated shorts (below/near VWAP):")
print(f"  Total: {len(eliminated_shorts)} trades")
print(f"  Average return: {eliminated_shorts['pnl_pct'].mean():.3f}% (avoided!)")

# Calculate new strategy performance
longs = trades_df[trades_df['direction'] == 'long']
filtered_strategy = pd.concat([longs, filtered_shorts])

print(f"\n=== Strategy Performance with VWAP Filter ===")
print(f"Total trades: {len(filtered_strategy)} (was {len(trades_df)})")
print(f"Average return: {filtered_strategy['pnl_pct'].mean():.3f}% (was {trades_df['pnl_pct'].mean():.3f}%)")
print(f"Win rate: {(filtered_strategy['pnl_pct'] > 0).mean():.1%}")

# With stop loss
filtered_with_stop = filtered_strategy.copy()
filtered_with_stop.loc[filtered_with_stop['pnl_pct'] < -0.1, 'pnl_pct'] = -0.1
print(f"\nWith -0.1% stop loss:")
print(f"Average return: {filtered_with_stop['pnl_pct'].mean():.3f}%")

# Calculate annualized returns
original_tpy = 772
filtered_tpy = original_tpy * len(filtered_strategy) / len(trades_df)

print(f"\n=== Annualized Return Comparison ===")
scenarios = [
    ("Original L/S", trades_df['pnl_pct'].mean(), original_tpy),
    ("Original + Stop", 0.034, original_tpy),
    ("VWAP Filter", filtered_strategy['pnl_pct'].mean(), filtered_tpy),
    ("VWAP Filter + Stop", filtered_with_stop['pnl_pct'].mean(), filtered_tpy),
]

print(f"{'Strategy':<20} {'Avg/Trade':<10} {'Trades/Yr':<10} {'Annual Return':<15}")
print("-" * 55)
for name, avg_ret, tpy in scenarios:
    annual = (1 + avg_ret/100)**tpy - 1
    print(f"{name:<20} {avg_ret:>6.3f}%   {tpy:>8.0f}   {annual*100:>10.1f}%")

# Best filters combination
print("\n=== Combined Filters Analysis ===")
# No overnight + above VWAP shorts + stop loss
# Add held_overnight to filtered_strategy
filtered_strategy['entry_date'] = filtered_strategy['entry_time'].dt.date
filtered_strategy['exit_date'] = pd.to_datetime(filtered_strategy['exit_time']).dt.date
filtered_strategy['held_overnight'] = filtered_strategy['entry_date'] != filtered_strategy['exit_date']
best_filtered = filtered_strategy[~filtered_strategy['held_overnight']]
best_with_stop = best_filtered.copy()
best_with_stop.loc[best_with_stop['pnl_pct'] < -0.1, 'pnl_pct'] = -0.1

best_tpy = original_tpy * len(best_filtered) / len(trades_df)
print(f"\nIntraday + VWAP Filter + Stop Loss:")
print(f"  Total trades: {len(best_filtered)}")
print(f"  Average return: {best_with_stop['pnl_pct'].mean():.3f}%")
print(f"  Trades per year: {best_tpy:.0f}")
best_annual = (1 + best_with_stop['pnl_pct'].mean()/100)**best_tpy - 1
print(f"  Annualized return: {best_annual*100:.1f}%")

# Additional insights
print("\n=== Key Insights ===")
above_vwap_shorts = shorts[shorts['vwap_position'].isin(['above', 'far_above'])]
below_vwap_shorts = shorts[~shorts['vwap_position'].isin(['above', 'far_above'])]

if len(above_vwap_shorts) > 0 and len(below_vwap_shorts) > 0:
    print(f"\nShort performance by VWAP position:")
    print(f"  Above VWAP: {above_vwap_shorts['pnl_pct'].mean():.3f}% avg, {(above_vwap_shorts['pnl_pct'] > 0).mean():.1%} win rate")
    print(f"  Below/Near VWAP: {below_vwap_shorts['pnl_pct'].mean():.3f}% avg, {(below_vwap_shorts['pnl_pct'] > 0).mean():.1%} win rate")
    print(f"\nImprovement from filter: {above_vwap_shorts['pnl_pct'].mean() - shorts['pnl_pct'].mean():.3f}% per short trade")