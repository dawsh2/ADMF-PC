"""Test combining VWAP and Volume filters"""
import pandas as pd
import numpy as np

# Load trades with all indicators
trades_df = pd.read_csv('bb_trades_fresh_indicators.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

print("=== Combined VWAP + Volume Filter Analysis ===\n")

# Apply individual filters first to see impact
# VWAP filter: shorts only above VWAP
longs = trades_df[trades_df['direction'] == 'long']
shorts = trades_df[trades_df['direction'] == 'short']
shorts_above_vwap = shorts[shorts['vwap_position'].isin(['above', 'far_above'])]

# Volume filter: exclude high volume (>1.2x)
low_volume_trades = trades_df[trades_df['volume_ratio'] <= 1.2]

print("Individual filter impacts:")
print(f"Original: {len(trades_df)} trades, avg: {trades_df['pnl_pct'].mean():.3f}%")
print(f"VWAP filter alone: {len(longs) + len(shorts_above_vwap)} trades, avg: {pd.concat([longs, shorts_above_vwap])['pnl_pct'].mean():.3f}%")
print(f"Volume filter alone: {len(low_volume_trades)} trades, avg: {low_volume_trades['pnl_pct'].mean():.3f}%")

# Combined filters
# Method 1: Apply both filters to all trades
longs_low_vol = longs[longs['volume_ratio'] <= 1.2]
shorts_above_vwap_low_vol = shorts_above_vwap[shorts_above_vwap['volume_ratio'] <= 1.2]
combined_filtered = pd.concat([longs_low_vol, shorts_above_vwap_low_vol])

print(f"\n=== Combined VWAP + Volume Filter ===")
print(f"Total trades: {len(combined_filtered)} ({len(combined_filtered)/len(trades_df)*100:.1f}% of original)")
print(f"Average return: {combined_filtered['pnl_pct'].mean():.3f}%")
print(f"Win rate: {(combined_filtered['pnl_pct'] > 0).mean():.1%}")

# Breakdown by direction
combined_longs = combined_filtered[combined_filtered['direction'] == 'long']
combined_shorts = combined_filtered[combined_filtered['direction'] == 'short']
print(f"\nBy direction:")
print(f"Longs: {len(combined_longs)} trades, avg: {combined_longs['pnl_pct'].mean():.3f}%")
print(f"Shorts: {len(combined_shorts)} trades, avg: {combined_shorts['pnl_pct'].mean():.3f}%")

# What we're filtering out
print(f"\n=== Trades Filtered Out ===")
# VWAP filtered shorts
vwap_filtered_shorts = shorts[~shorts['vwap_position'].isin(['above', 'far_above'])]
print(f"Shorts below/near VWAP: {len(vwap_filtered_shorts)} trades, avg: {vwap_filtered_shorts['pnl_pct'].mean():.3f}%")

# Volume filtered (from combined set)
volume_filtered_from_vwap = pd.concat([longs, shorts_above_vwap])
volume_filtered_from_vwap = volume_filtered_from_vwap[volume_filtered_from_vwap['volume_ratio'] > 1.2]
print(f"High volume from VWAP-filtered: {len(volume_filtered_from_vwap)} trades, avg: {volume_filtered_from_vwap['pnl_pct'].mean():.3f}%")

# With stop loss
combined_with_stop = combined_filtered.copy()
combined_with_stop.loc[combined_with_stop['pnl_pct'] < -0.1, 'pnl_pct'] = -0.1

print(f"\n=== Combined Filters + Stop Loss ===")
print(f"Average return: {combined_with_stop['pnl_pct'].mean():.3f}%")
print(f"Win rate: {(combined_with_stop['pnl_pct'] > 0).mean():.1%}")

# Also exclude overnight
combined_filtered['held_overnight'] = combined_filtered['entry_time'].dt.date != combined_filtered['exit_time'].dt.date
intraday_combined = combined_filtered[~combined_filtered['held_overnight']]
intraday_combined_stop = intraday_combined.copy()
intraday_combined_stop.loc[intraday_combined_stop['pnl_pct'] < -0.1, 'pnl_pct'] = -0.1

print(f"\n=== All Filters (Intraday + VWAP + Volume + Stop) ===")
print(f"Total trades: {len(intraday_combined)} ({len(intraday_combined)/len(trades_df)*100:.1f}% of original)")
print(f"Average return: {intraday_combined_stop['pnl_pct'].mean():.3f}%")

# Calculate annualized returns
print(f"\n=== Annualized Returns Comparison (1 bps cost) ===")
execution_cost = 0.0002  # 2 bps round trip
trades_per_year_original = 772

scenarios = [
    ("Original", trades_df['pnl_pct'].mean(), trades_per_year_original),
    ("Original + Stop", 0.034, trades_per_year_original),
    ("VWAP Filter", pd.concat([longs, shorts_above_vwap])['pnl_pct'].mean(), 
     trades_per_year_original * (len(longs) + len(shorts_above_vwap)) / len(trades_df)),
    ("Volume Filter", low_volume_trades['pnl_pct'].mean(), 
     trades_per_year_original * len(low_volume_trades) / len(trades_df)),
    ("VWAP + Volume", combined_filtered['pnl_pct'].mean(), 
     trades_per_year_original * len(combined_filtered) / len(trades_df)),
    ("VWAP + Volume + Stop", combined_with_stop['pnl_pct'].mean(), 
     trades_per_year_original * len(combined_filtered) / len(trades_df)),
    ("All Filters + Stop", intraday_combined_stop['pnl_pct'].mean(), 
     trades_per_year_original * len(intraday_combined) / len(trades_df)),
]

print(f"{'Strategy':<25} {'Gross/Trade':<12} {'Net/Trade':<12} {'Trades/Yr':<10} {'Net Annual':<12}")
print("-" * 75)

best_net_annual = -1
best_strategy = ""

for name, gross_pct, tpy in scenarios:
    gross_decimal = gross_pct / 100
    net_decimal = gross_decimal - execution_cost
    
    if net_decimal > 0:
        net_annual = (1 + net_decimal) ** tpy - 1
        print(f"{name:<25} {gross_pct:>10.3f}% {net_decimal*100:>10.3f}% {tpy:>9.0f} {net_annual*100:>10.1f}%")
        if net_annual > best_net_annual:
            best_net_annual = net_annual
            best_strategy = name
    else:
        print(f"{name:<25} {gross_pct:>10.3f}% {net_decimal*100:>10.3f}% {tpy:>9.0f} {'LOSS':>11}")

print(f"\n=== Summary ===")
print(f"Best strategy: {best_strategy} with {best_net_annual*100:.1f}% net annual return")

# Filter interaction analysis
print(f"\n=== Filter Interaction Analysis ===")
# How many high volume trades are also bad VWAP shorts?
bad_shorts = shorts[~shorts['vwap_position'].isin(['above', 'far_above'])]
bad_shorts_high_vol = bad_shorts[bad_shorts['volume_ratio'] > 1.2]
print(f"Bad shorts (below/near VWAP) with high volume: {len(bad_shorts_high_vol)} trades")
print(f"Average return: {bad_shorts_high_vol['pnl_pct'].mean():.3f}%")

# Trade quality metrics
print(f"\n=== Trade Quality Improvement ===")
print(f"{'Filter':<20} {'Avg Return':<12} {'Sharpe Proxy':<15}")
print("-" * 45)
for name, trades in [
    ("Original", trades_df),
    ("VWAP", pd.concat([longs, shorts_above_vwap])),
    ("Volume", low_volume_trades),
    ("VWAP + Volume", combined_filtered)
]:
    avg_ret = trades['pnl_pct'].mean()
    std_ret = trades['pnl_pct'].std()
    sharpe_proxy = avg_ret / std_ret if std_ret > 0 else 0
    print(f"{name:<20} {avg_ret:>10.3f}% {sharpe_proxy:>13.3f}")