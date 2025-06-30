"""Analyze long-only performance and investigate short underperformance"""
import pandas as pd
import numpy as np

# Load trades with sensitive trend data
trades_df = pd.read_csv('bb_trades_sensitive_trend.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

print("=== Long-Only Analysis ===")

# Current performance recap
print(f"\nCurrent Performance (Long/Short):")
print(f"Total trades: {len(trades_df)}")
print(f"Average return: {trades_df['pnl_pct'].mean():.3f}%")
print(f"Annualized (772 trades/year): {((1 + trades_df['pnl_pct'].mean()/100)**772 - 1)*100:.1f}%")

# Long-only performance
longs_only = trades_df[trades_df['direction'] == 'long']
print(f"\nLong-Only Performance:")
print(f"Total trades: {len(longs_only)} (51.5% of all trades)")
print(f"Average return: {longs_only['pnl_pct'].mean():.3f}%")
print(f"Win rate: {(longs_only['pnl_pct'] > 0).mean():.1%}")

# Adjusted annual trades for long-only
trades_per_year_long_only = 772 * len(longs_only) / len(trades_df)
print(f"Trades per year (long-only): {trades_per_year_long_only:.0f}")
annual_return_long_only = (1 + longs_only['pnl_pct'].mean()/100)**trades_per_year_long_only - 1
print(f"Annualized return (long-only): {annual_return_long_only*100:.1f}%")

# Long-only with stop loss
longs_with_stop = longs_only.copy()
longs_with_stop.loc[longs_with_stop['pnl_pct'] < -0.1, 'pnl_pct'] = -0.1
avg_return_long_stop = longs_with_stop['pnl_pct'].mean()
print(f"\nLong-Only with -0.1% stop loss:")
print(f"Average return: {avg_return_long_stop:.3f}%")
annual_return_long_stop = (1 + avg_return_long_stop/100)**trades_per_year_long_only - 1
print(f"Annualized return: {annual_return_long_stop*100:.1f}%")

# Investigate short underperformance
print("\n=== Investigating Short Underperformance ===")

shorts = trades_df[trades_df['direction'] == 'short']
print(f"\nShort trades analysis:")
print(f"Total: {len(shorts)} trades")
print(f"Average return: {shorts['pnl_pct'].mean():.3f}%")
print(f"Win rate: {(shorts['pnl_pct'] > 0).mean():.1%}")

# By trend
for trend in ['uptrend', 'downtrend', 'sideways']:
    trend_shorts = shorts[shorts['trend_new'] == trend]
    if len(trend_shorts) > 0:
        print(f"\nShorts in {trend}: {len(trend_shorts)} trades")
        print(f"  Average: {trend_shorts['pnl_pct'].mean():.3f}%")
        print(f"  Win rate: {(trend_shorts['pnl_pct'] > 0).mean():.1%}")
        print(f"  Avg bars held: {trend_shorts['bars_held'].mean():.1f}")

# Compare entry/exit timing for shorts
print("\n=== Short Trade Timing Analysis ===")
shorts['entry_hour'] = pd.to_datetime(shorts['entry_time']).dt.hour
shorts['exit_hour'] = pd.to_datetime(shorts['exit_time']).dt.hour

# Look at holding periods
print(f"\nHolding period analysis:")
print(f"Short winners avg bars: {shorts[shorts['pnl_pct'] > 0]['bars_held'].mean():.1f}")
print(f"Short losers avg bars: {shorts[shorts['pnl_pct'] < 0]['bars_held'].mean():.1f}")

# Market structure bias
print("\n=== Market Structure Analysis ===")
print("SPY has natural upward drift over time.")
print("Mean reversion strategies short rallies expecting pullbacks,")
print("but in trending bull markets, 'overbought' can stay overbought.")

# What if we filtered shorts more strictly?
print("\n=== Selective Shorting Analysis ===")

# Only short in downtrends or with extreme RSI divergence
selective_shorts = shorts[
    (shorts['trend_new'] == 'downtrend') | 
    (shorts['vwap_position'] == 'far_above')
]
print(f"\nSelective shorts (downtrend or far above VWAP): {len(selective_shorts)} trades")
if len(selective_shorts) > 0:
    print(f"Average return: {selective_shorts['pnl_pct'].mean():.3f}%")
    print(f"Win rate: {(selective_shorts['pnl_pct'] > 0).mean():.1%}")

# Compare different strategy modes
print("\n=== Strategy Comparison Summary ===")
modes = {
    'Current (L/S)': {
        'trades': len(trades_df),
        'avg_return': trades_df['pnl_pct'].mean(),
        'trades_year': 772
    },
    'Long-Only': {
        'trades': len(longs_only),
        'avg_return': longs_only['pnl_pct'].mean(),
        'trades_year': trades_per_year_long_only
    },
    'L/S + Stop': {
        'trades': len(trades_df),
        'avg_return': 0.034,  # From previous analysis
        'trades_year': 772
    },
    'Long + Stop': {
        'trades': len(longs_only),
        'avg_return': avg_return_long_stop,
        'trades_year': trades_per_year_long_only
    }
}

print(f"\n{'Strategy':<15} {'Avg/Trade':<10} {'Trades/Yr':<10} {'Annual Return':<15}")
print("-" * 50)
for name, data in modes.items():
    annual = (1 + data['avg_return']/100)**data['trades_year'] - 1
    print(f"{name:<15} {data['avg_return']:.3f}%    {data['trades_year']:<10.0f} {annual*100:.1f}%")

# Risk-adjusted view
print("\n=== Risk Considerations ===")
print("Long-only benefits:")
print("- Eliminates wrong-way risk in uptrending markets")
print("- Reduces trade frequency (lower costs)")
print("- Aligns with market's natural upward bias")
print("- Simpler execution and risk management")
print("\nLong-only drawbacks:")
print("- Misses profit opportunities in downtrends")
print("- Less diversification of market conditions")
print("- May have longer flat periods")