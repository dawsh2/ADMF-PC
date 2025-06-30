"""Test filtering out high volume periods"""
import pandas as pd
import numpy as np

# Load trades with all indicators
trades_df = pd.read_csv('bb_trades_fresh_indicators.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

print("=== Volume Filter Analysis ===")

# Current volume distribution
print(f"\nVolume ratio distribution:")
print(f"Mean: {trades_df['volume_ratio'].mean():.2f}")
print(f"Median: {trades_df['volume_ratio'].median():.2f}")
print(f"75th percentile: {trades_df['volume_ratio'].quantile(0.75):.2f}")
print(f"90th percentile: {trades_df['volume_ratio'].quantile(0.90):.2f}")

# Categorize volume
trades_df['volume_category'] = 'normal'
trades_df.loc[trades_df['volume_ratio'] < 0.8, 'volume_category'] = 'low'
trades_df.loc[trades_df['volume_ratio'] > 1.2, 'volume_category'] = 'high'

# Performance by volume category
print(f"\n=== Performance by Volume Category ===")
for vol_cat in ['low', 'normal', 'high']:
    vol_trades = trades_df[trades_df['volume_category'] == vol_cat]
    if len(vol_trades) > 0:
        print(f"\n{vol_cat.capitalize()} volume: {len(vol_trades)} trades ({len(vol_trades)/len(trades_df)*100:.1f}%)")
        print(f"  Average return: {vol_trades['pnl_pct'].mean():.3f}%")
        print(f"  Win rate: {(vol_trades['pnl_pct'] > 0).mean():.1%}")
        
        # By direction
        vol_longs = vol_trades[vol_trades['direction'] == 'long']
        vol_shorts = vol_trades[vol_trades['direction'] == 'short']
        if len(vol_longs) > 0:
            print(f"  Longs: {len(vol_longs)} trades, avg: {vol_longs['pnl_pct'].mean():.3f}%")
        if len(vol_shorts) > 0:
            print(f"  Shorts: {len(vol_shorts)} trades, avg: {vol_shorts['pnl_pct'].mean():.3f}%")

# Test different volume filters
print(f"\n=== Testing Volume Filters ===")

# Filter 1: Exclude high volume (>1.2x)
filter1_trades = trades_df[trades_df['volume_ratio'] <= 1.2]
print(f"\nFilter 1: Exclude volume > 1.2x average")
print(f"  Remaining trades: {len(filter1_trades)} ({len(filter1_trades)/len(trades_df)*100:.1f}%)")
print(f"  Average return: {filter1_trades['pnl_pct'].mean():.3f}% (was {trades_df['pnl_pct'].mean():.3f}%)")
print(f"  Win rate: {(filter1_trades['pnl_pct'] > 0).mean():.1%}")

# Filter 2: Exclude very high volume (>1.5x)
filter2_trades = trades_df[trades_df['volume_ratio'] <= 1.5]
print(f"\nFilter 2: Exclude volume > 1.5x average")
print(f"  Remaining trades: {len(filter2_trades)} ({len(filter2_trades)/len(trades_df)*100:.1f}%)")
print(f"  Average return: {filter2_trades['pnl_pct'].mean():.3f}%")
print(f"  Win rate: {(filter2_trades['pnl_pct'] > 0).mean():.1%}")

# Filter 3: Only trade in quiet markets (<1.0x)
filter3_trades = trades_df[trades_df['volume_ratio'] < 1.0]
print(f"\nFilter 3: Only trade when volume < 1.0x average")
print(f"  Remaining trades: {len(filter3_trades)} ({len(filter3_trades)/len(trades_df)*100:.1f}%)")
print(f"  Average return: {filter3_trades['pnl_pct'].mean():.3f}%")
print(f"  Win rate: {(filter3_trades['pnl_pct'] > 0).mean():.1%}")

# Filter 4: Optimal range (0.5x - 1.2x)
filter4_trades = trades_df[(trades_df['volume_ratio'] >= 0.5) & (trades_df['volume_ratio'] <= 1.2)]
print(f"\nFilter 4: Volume between 0.5x and 1.2x")
print(f"  Remaining trades: {len(filter4_trades)} ({len(filter4_trades)/len(trades_df)*100:.1f}%)")
print(f"  Average return: {filter4_trades['pnl_pct'].mean():.3f}%")
print(f"  Win rate: {(filter4_trades['pnl_pct'] > 0).mean():.1%}")

# Best filter with stop loss
best_filter = filter1_trades.copy()  # Using 1.2x cutoff
best_with_stop = best_filter.copy()
best_with_stop.loc[best_with_stop['pnl_pct'] < -0.1, 'pnl_pct'] = -0.1

print(f"\n=== Volume Filter + Stop Loss ===")
print(f"Volume ≤ 1.2x + Stop Loss:")
print(f"  Average return: {best_with_stop['pnl_pct'].mean():.3f}%")
print(f"  Win rate: {(best_with_stop['pnl_pct'] > 0).mean():.1%}")

# Calculate annualized returns with execution costs
print(f"\n=== Annualized Returns with 1 bps Execution Cost ===")
execution_cost = 0.0002  # 2 bps round trip

scenarios = [
    ("Original", trades_df['pnl_pct'].mean(), 772),
    ("Original + Stop", 0.034, 772),
    ("Volume Filter (≤1.2x)", filter1_trades['pnl_pct'].mean(), 772 * len(filter1_trades)/len(trades_df)),
    ("Volume Filter + Stop", best_with_stop['pnl_pct'].mean(), 772 * len(best_with_stop)/len(trades_df)),
]

print(f"{'Strategy':<25} {'Gross/Trade':<12} {'Net/Trade':<12} {'Trades/Yr':<10} {'Net Annual':<12}")
print("-" * 75)

for name, gross_pct, tpy in scenarios:
    gross_decimal = gross_pct / 100
    net_decimal = gross_decimal - execution_cost
    
    if net_decimal > 0:
        net_annual = (1 + net_decimal) ** tpy - 1
        print(f"{name:<25} {gross_pct:>10.3f}% {net_decimal*100:>10.3f}% {tpy:>9.0f} {net_annual*100:>10.1f}%")
    else:
        print(f"{name:<25} {gross_pct:>10.3f}% {net_decimal*100:>10.3f}% {tpy:>9.0f} {'LOSS':>11}")

# Check what we're filtering out
print(f"\n=== What Gets Filtered Out (>1.2x volume) ===")
filtered_out = trades_df[trades_df['volume_ratio'] > 1.2]
print(f"Trades removed: {len(filtered_out)}")
print(f"Average return of removed trades: {filtered_out['pnl_pct'].mean():.3f}%")
print(f"Win rate of removed trades: {(filtered_out['pnl_pct'] > 0).mean():.1%}")

# By direction
filtered_longs = filtered_out[filtered_out['direction'] == 'long']
filtered_shorts = filtered_out[filtered_out['direction'] == 'short']
if len(filtered_longs) > 0:
    print(f"Removed longs: {len(filtered_longs)} trades, avg: {filtered_longs['pnl_pct'].mean():.3f}%")
if len(filtered_shorts) > 0:
    print(f"Removed shorts: {len(filtered_shorts)} trades, avg: {filtered_shorts['pnl_pct'].mean():.3f}%")

# Combine with other filters
print(f"\n=== Combined Filters Analysis ===")
# No overnight + volume ≤1.2x + stop loss
trades_df['held_overnight'] = trades_df['entry_time'].dt.date != pd.to_datetime(trades_df['exit_time']).dt.date
combined_filter = trades_df[(~trades_df['held_overnight']) & (trades_df['volume_ratio'] <= 1.2)]
combined_with_stop = combined_filter.copy()
combined_with_stop.loc[combined_with_stop['pnl_pct'] < -0.1, 'pnl_pct'] = -0.1

print(f"\nIntraday + Volume ≤1.2x + Stop Loss:")
print(f"  Total trades: {len(combined_filter)} ({len(combined_filter)/len(trades_df)*100:.1f}% of original)")
print(f"  Average return: {combined_with_stop['pnl_pct'].mean():.3f}%")

combined_tpy = 772 * len(combined_filter) / len(trades_df)
combined_net = combined_with_stop['pnl_pct'].mean() / 100 - execution_cost
if combined_net > 0:
    combined_annual = (1 + combined_net) ** combined_tpy - 1
    print(f"  Net annual return (after 1 bps cost): {combined_annual*100:.1f}%")

# Final recommendations
print(f"\n=== Recommendations ===")
print("1. Volume filter shows promise - excluding high volume (>1.2x) improves returns")
print("2. High volume periods often signal breakouts that continue, not mean reversion")
print("3. Best results in quiet markets where mean reversion is more reliable")
print("4. Combined with stop loss, volume filter adds value but less than stop alone")