"""Detailed analysis of what trades get filtered out"""
import pandas as pd
import numpy as np

# Load trades
trades_df = pd.read_csv('bb_trades_fresh_indicators.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

print("=== Detailed Analysis of Filtered Trades ===\n")

# Separate filters
longs = trades_df[trades_df['direction'] == 'long']
shorts = trades_df[trades_df['direction'] == 'short']

# VWAP filter
shorts_above_vwap = shorts[shorts['vwap_position'].isin(['above', 'far_above'])]
shorts_below_vwap = shorts[~shorts['vwap_position'].isin(['above', 'far_above'])]
vwap_filtered_trades = pd.concat([longs, shorts_above_vwap])

# Volume filter on original
volume_filtered_trades = trades_df[trades_df['volume_ratio'] <= 1.2]
high_volume_trades = trades_df[trades_df['volume_ratio'] > 1.2]

# Combined filter
combined_filtered = vwap_filtered_trades[vwap_filtered_trades['volume_ratio'] <= 1.2]

print("=== VWAP Filter Impact ===")
print(f"Keeps: {len(vwap_filtered_trades)} trades")
print(f"  - All {len(longs)} longs (avg: {longs['pnl_pct'].mean():.3f}%)")
print(f"  - {len(shorts_above_vwap)} shorts above VWAP (avg: {shorts_above_vwap['pnl_pct'].mean():.3f}%)")
print(f"Removes: {len(shorts_below_vwap)} shorts below/near VWAP (avg: {shorts_below_vwap['pnl_pct'].mean():.3f}%)")

print(f"\n=== Volume Filter Impact (from VWAP-filtered set) ===")
# What does volume filter remove from the VWAP-filtered trades?
high_vol_from_vwap = vwap_filtered_trades[vwap_filtered_trades['volume_ratio'] > 1.2]
low_vol_from_vwap = vwap_filtered_trades[vwap_filtered_trades['volume_ratio'] <= 1.2]

print(f"From VWAP-filtered trades ({len(vwap_filtered_trades)}):")
print(f"  Keeps: {len(low_vol_from_vwap)} low volume trades (avg: {low_vol_from_vwap['pnl_pct'].mean():.3f}%)")
print(f"  Removes: {len(high_vol_from_vwap)} high volume trades (avg: {high_vol_from_vwap['pnl_pct'].mean():.3f}%)")

# Break down what high volume trades we're removing
high_vol_longs = high_vol_from_vwap[high_vol_from_vwap['direction'] == 'long']
high_vol_shorts = high_vol_from_vwap[high_vol_from_vwap['direction'] == 'short']

print(f"\nHigh volume trades being removed:")
print(f"  - {len(high_vol_longs)} longs (avg: {high_vol_longs['pnl_pct'].mean():.3f}%)")
print(f"  - {len(high_vol_shorts)} shorts above VWAP (avg: {high_vol_shorts['pnl_pct'].mean():.3f}%)")

# The key insight
print(f"\n=== THE KEY INSIGHT ===")
print(f"Original average: {trades_df['pnl_pct'].mean():.3f}% over {len(trades_df)} trades")
print(f"VWAP filtered average: {vwap_filtered_trades['pnl_pct'].mean():.3f}% over {len(vwap_filtered_trades)} trades")
print(f"Combined filtered average: {combined_filtered['pnl_pct'].mean():.3f}% over {len(combined_filtered)} trades")

print(f"\nWe're removing:")
print(f"1. Bad shorts below VWAP: {len(shorts_below_vwap)} trades at {shorts_below_vwap['pnl_pct'].mean():.3f}% ✓")
print(f"2. PROFITABLE high volume trades: {len(high_vol_from_vwap)} trades at {high_vol_from_vwap['pnl_pct'].mean():.3f}% ✗")

# Calculate total PnL impact
total_original = trades_df['pnl_pct'].sum()
total_combined = combined_filtered['pnl_pct'].sum()
total_removed_bad_shorts = shorts_below_vwap['pnl_pct'].sum()
total_removed_high_vol = high_vol_from_vwap['pnl_pct'].sum()

print(f"\n=== Total PnL Analysis ===")
print(f"Original total PnL: {total_original:.2f}%")
print(f"Combined filter total PnL: {total_combined:.2f}%")
print(f"PnL from removed bad shorts: {total_removed_bad_shorts:.2f}%")
print(f"PnL from removed high volume: {total_removed_high_vol:.2f}%")
print(f"Net change: {total_combined - total_original:.2f}%")

# Why annual returns are lower
print(f"\n=== Why Annual Returns Are Lower ===")
print("Even though average per trade improves, we lose compounding opportunities:")

# Simulate compounding
original_compound = 1
combined_compound = 1
trades_per_year_original = 772
trades_per_year_combined = 772 * len(combined_filtered) / len(trades_df)

print(f"\nOriginal: {trades_per_year_original:.0f} trades/year at 0.015% each")
print(f"Combined: {trades_per_year_combined:.0f} trades/year at 0.028% each")

# With execution costs
exec_cost = 0.0002
net_original = 0.00015 - exec_cost  # Negative!
net_combined = 0.00028 - exec_cost

print(f"\nAfter 1bp execution costs:")
print(f"Original: {net_original*100:.3f}% per trade = LOSS")
print(f"Combined: {net_combined*100:.3f}% per trade = {((1+net_combined)**trades_per_year_combined-1)*100:.1f}% annual")

# The real problem
print(f"\n=== The Real Problem ===")
print("The volume filter is TOO AGGRESSIVE!")
print(f"It removes {len(high_vol_from_vwap)} profitable trades averaging {high_vol_from_vwap['pnl_pct'].mean():.3f}%")
print("These aren't losing trades - they're actually good trades in high volume!")

# What if we only used VWAP filter?
vwap_only_net = vwap_filtered_trades['pnl_pct'].mean()/100 - exec_cost
vwap_only_tpy = 772 * len(vwap_filtered_trades) / len(trades_df)
vwap_only_annual = (1 + vwap_only_net) ** vwap_only_tpy - 1

print(f"\n=== Better Option: VWAP Filter Only ===")
print(f"VWAP filter alone: {vwap_filtered_trades['pnl_pct'].mean():.3f}% per trade")
print(f"Trades per year: {vwap_only_tpy:.0f}")
print(f"Net annual return: {vwap_only_annual*100:.1f}%")
print("\nThis is better because we keep the profitable high-volume trades!")