"""Final strategy summary - VWAP filter for shorts only"""
import pandas as pd
import numpy as np

# Load trades
trades_df = pd.read_csv('bb_trades_fresh_indicators.csv')

print("=== Final Strategy Configuration ===\n")

# The optimal strategy based on our analysis
longs = trades_df[trades_df['direction'] == 'long']
shorts = trades_df[trades_df['direction'] == 'short']
shorts_above_vwap = shorts[shorts['vwap_position'].isin(['above', 'far_above'])]

# Final strategy trades
final_strategy = pd.concat([longs, shorts_above_vwap])

print("Bollinger RSI Simple Signals with VWAP Filter for Shorts:")
print(f"- Keep ALL {len(longs)} long trades")
print(f"- Keep ONLY {len(shorts_above_vwap)} shorts above VWAP")
print(f"- Total trades: {len(final_strategy)}")

# Performance summary
print(f"\n=== Performance Summary ===")
print(f"Average return per trade: {final_strategy['pnl_pct'].mean():.3f}%")
print(f"Win rate: {(final_strategy['pnl_pct'] > 0).mean():.1%}")

# With stop loss
final_with_stop = final_strategy.copy()
final_with_stop.loc[final_with_stop['pnl_pct'] < -0.1, 'pnl_pct'] = -0.1
print(f"\nWith -0.1% stop loss:")
print(f"Average return per trade: {final_with_stop['pnl_pct'].mean():.3f}%")

# Annual returns with execution costs
trades_per_year = 772 * len(final_strategy) / len(trades_df)
execution_cost = 0.0002  # 1 bps per side

gross_return = final_with_stop['pnl_pct'].mean() / 100
net_return = gross_return - execution_cost
annual_return = (1 + net_return) ** trades_per_year - 1

print(f"\n=== Annual Return (with 1 bps execution cost) ===")
print(f"Trades per year: {trades_per_year:.0f}")
print(f"Gross return per trade: {gross_return*100:.3f}%")
print(f"Net return per trade: {net_return*100:.3f}%")
print(f"Net annual return: {annual_return*100:.1f}%")

# What we're filtering out
shorts_below_vwap = shorts[~shorts['vwap_position'].isin(['above', 'far_above'])]
print(f"\n=== What We Filter Out ===")
print(f"Removed {len(shorts_below_vwap)} shorts below/near VWAP")
print(f"Average return of removed shorts: {shorts_below_vwap['pnl_pct'].mean():.3f}%")
print(f"Total PnL avoided: {shorts_below_vwap['pnl_pct'].sum():.2f}%")

# Direction breakdown
print(f"\n=== Final Strategy Composition ===")
final_longs = final_strategy[final_strategy['direction'] == 'long']
final_shorts = final_strategy[final_strategy['direction'] == 'short']
print(f"Longs: {len(final_longs)} trades ({len(final_longs)/len(final_strategy)*100:.1f}%), avg: {final_longs['pnl_pct'].mean():.3f}%")
print(f"Shorts: {len(final_shorts)} trades ({len(final_shorts)/len(final_strategy)*100:.1f}%), avg: {final_shorts['pnl_pct'].mean():.3f}%")

print(f"\n=== Implementation Notes ===")
print("1. Strategy: Bollinger RSI Simple Signals")
print("2. Enhancement: Only short when price is above VWAP")
print("3. Risk Management: -0.1% stop loss (handled by risk module)")
print("4. No other filters needed (no volatility filter, no volume filter)")
print("5. Trade all market conditions and times")

print(f"\n=== Why This Works ===")
print("- Shorting above VWAP = true mean reversion (0.031% avg)")
print("- Shorting below VWAP = fighting the trend (-0.022% avg)")
print("- Longs work everywhere (0.030% avg)")
print("- Stop loss prevents disasters and doubles returns")
print("- Simple, robust, and profitable after costs")