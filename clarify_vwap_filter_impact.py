"""Clarify VWAP filter impact - we're removing losing trades, not random trades"""
import pandas as pd
import numpy as np

print("=== Clarifying VWAP Filter Impact ===\n")

# Key numbers from our analysis
original_stats = {
    'total_trades': 196,
    'avg_return': 0.015,
    'longs': 101,
    'shorts': 95,
    'long_avg': 0.030,
    'short_avg': -0.002
}

vwap_filter_stats = {
    'total_trades': 136,  # 101 longs + 35 shorts
    'avg_return': 0.030,
    'longs': 101,  # All longs kept
    'shorts': 35,   # Only above-VWAP shorts
    'shorts_kept_avg': 0.031,
    'shorts_removed': 60,
    'shorts_removed_avg': -0.022
}

print("What the VWAP filter actually does:")
print(f"- Keeps ALL {vwap_filter_stats['longs']} long trades (avg: {original_stats['long_avg']:.3f}%)")
print(f"- Keeps {vwap_filter_stats['shorts']} shorts above VWAP (avg: {vwap_filter_stats['shorts_kept_avg']:.3f}%)")
print(f"- REMOVES {vwap_filter_stats['shorts_removed']} shorts below/near VWAP (avg: {vwap_filter_stats['shorts_removed_avg']:.3f}%)")

# Calculate the actual impact
removed_trade_impact = vwap_filter_stats['shorts_removed'] * vwap_filter_stats['shorts_removed_avg'] / 100
total_return_original = original_stats['total_trades'] * original_stats['avg_return'] / 100
total_return_filtered = vwap_filter_stats['total_trades'] * vwap_filter_stats['avg_return'] / 100

print(f"\n=== Mathematical Breakdown ===")
print(f"Original strategy total return: {original_stats['total_trades']} × {original_stats['avg_return']:.3f}% = {total_return_original:.2f}%")
print(f"Filtered strategy total return: {vwap_filter_stats['total_trades']} × {vwap_filter_stats['avg_return']:.3f}% = {total_return_filtered:.2f}%")
print(f"Return from removed trades: {vwap_filter_stats['shorts_removed']} × {vwap_filter_stats['shorts_removed_avg']:.3f}% = {removed_trade_impact:.2f}%")
print(f"Net improvement: {total_return_filtered - total_return_original:.2f}%")

# Now let's recalculate annualized returns properly
trades_per_year_original = 772
trades_per_year_filtered = trades_per_year_original * vwap_filter_stats['total_trades'] / original_stats['total_trades']

print(f"\n=== Corrected Annual Return Calculation ===")
print(f"We're removing LOSING trades, not reducing opportunities!")

# With execution costs
execution_cost = 0.0002  # 2 bps round trip

# Original
net_per_trade_original = original_stats['avg_return'] / 100 - execution_cost
annual_original = (1 + net_per_trade_original) ** trades_per_year_original - 1 if net_per_trade_original > 0 else -1

# VWAP filtered
net_per_trade_filtered = vwap_filter_stats['avg_return'] / 100 - execution_cost
annual_filtered = (1 + net_per_trade_filtered) ** trades_per_year_filtered - 1

# With stop loss
net_per_trade_original_stop = 0.034 / 100 - execution_cost
annual_original_stop = (1 + net_per_trade_original_stop) ** trades_per_year_original - 1

net_per_trade_filtered_stop = 0.039 / 100 - execution_cost
annual_filtered_stop = (1 + net_per_trade_filtered_stop) ** trades_per_year_filtered - 1

print(f"\nAnnual returns with 1 bps execution cost:")
print(f"{'Strategy':<25} {'Gross/Trade':<12} {'Net/Trade':<12} {'Trades/Yr':<10} {'Net Annual':<12}")
print("-" * 70)
print(f"{'Original':<25} {original_stats['avg_return']:>10.3f}% {net_per_trade_original*100:>10.3f}% {trades_per_year_original:>9.0f} {annual_original*100 if annual_original > 0 else 'LOSS':>11}")
print(f"{'VWAP Filter':<25} {vwap_filter_stats['avg_return']:>10.3f}% {net_per_trade_filtered*100:>10.3f}% {trades_per_year_filtered:>9.0f} {annual_filtered*100:>10.1f}%")
print(f"{'Original + Stop':<25} {0.034:>10.3f}% {net_per_trade_original_stop*100:>10.3f}% {trades_per_year_original:>9.0f} {annual_original_stop*100:>10.1f}%")
print(f"{'VWAP Filter + Stop':<25} {0.039:>10.3f}% {net_per_trade_filtered_stop*100:>10.3f}% {trades_per_year_filtered:>9.0f} {annual_filtered_stop*100:>10.1f}%")

print(f"\n=== The Real Story ===")
print("1. VWAP filter DOES improve returns by removing losing trades")
print("2. Without stop loss: VWAP filter makes strategy profitable (5.5% vs LOSS)")
print("3. With stop loss: Returns are similar (10.7% vs 11.4%)")
print("4. The 0.7% difference is due to:")
print("   - Slightly fewer compounding opportunities")
print("   - Some of the filtered shorts might have been stopped out profitably at -0.1%")

# What if we kept the trade frequency?
print(f"\n=== Alternative Perspective ===")
print("If we could maintain 772 trades/year with the VWAP filter quality:")
hypothetical_annual = (1 + net_per_trade_filtered) ** trades_per_year_original - 1
print(f"Hypothetical annual return: {hypothetical_annual*100:.1f}%")
print("(But this isn't realistic - good shorts above VWAP are less frequent)")

print(f"\n=== Final Verdict ===")
print("The VWAP filter DOES add value by:")
print("- Turning unprofitable shorts (+0.031%) from losing shorts (-0.022%)")
print("- Making the base strategy profitable even without stop loss")
print("- Providing more consistent returns")
print("\nThe slightly lower annual return with stop loss (10.7% vs 11.4%) is because:")
print("- We have fewer trades to compound")
print("- NOT because we're filtering good trades - we're filtering bad ones!")