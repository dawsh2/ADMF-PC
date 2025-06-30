"""Final analysis attempting to match user's results"""
import pandas as pd
import numpy as np

# Load trades with all indicators
trades_df = pd.read_csv('bb_trades_fresh_indicators.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

print("=== Bollinger RSI Simple Signals Analysis ===")
print(f"Total trades: {len(trades_df)}")
print(f"Date range: {trades_df['entry_time'].min()} to {trades_df['entry_time'].max()}")

# Overall performance (matching user's numbers)
print(f"\n=== Overall Performance ===")
print(f"Average return: {trades_df['pnl_pct'].mean():.3f}%")
print(f"Win rate: {(trades_df['pnl_pct'] > 0).mean():.1%}")

# Winners vs Losers
winners = trades_df[trades_df['pnl_pct'] > 0]
losers = trades_df[trades_df['pnl_pct'] < 0]

print(f"\n=== Winners vs Losers ===")
print(f"Winners: {len(winners)} trades, avg return: {winners['pnl_pct'].mean():.3f}%, avg bars: {winners['bars_held'].mean():.1f}")
print(f"Losers: {len(losers)} trades, avg return: {losers['pnl_pct'].mean():.3f}%, avg bars: {losers['bars_held'].mean():.1f}")
print(f"Losers hold {losers['bars_held'].mean() / winners['bars_held'].mean():.1f}x longer than winners")

# Quick exits
quick_winners = winners[winners['bars_held'] < 5]
quick_losers = losers[losers['bars_held'] < 5]
print(f"\nQuick exits (<5 bars):")
print(f"Winners: {len(quick_winners)} ({len(quick_winners)/len(winners)*100:.1f}%)")
print(f"Losers: {len(quick_losers)} ({len(quick_losers)/len(losers)*100:.1f}%)")

# Long vs Short
longs = trades_df[trades_df['direction'] == 'long']
shorts = trades_df[trades_df['direction'] == 'short']

print(f"\n=== Long vs Short Performance ===")
print(f"Longs: {len(longs)} trades, avg: {longs['pnl_pct'].mean():.3f}%, win rate: {(longs['pnl_pct'] > 0).mean():.1%}")
print(f"Shorts: {len(shorts)} trades, avg: {shorts['pnl_pct'].mean():.3f}%, win rate: {(shorts['pnl_pct'] > 0).mean():.1%}")

# What-if with stop loss
print(f"\n=== What-If Scenarios ===")
trades_with_stop = trades_df.copy()
trades_with_stop.loc[trades_with_stop['pnl_pct'] < -0.1, 'pnl_pct'] = -0.1
print(f"With -0.1% stop loss: avg return would be {trades_with_stop['pnl_pct'].mean():.3f}% (from {trades_df['pnl_pct'].mean():.3f}%)")
improvement = (trades_with_stop['pnl_pct'].mean() - trades_df['pnl_pct'].mean()) / trades_df['pnl_pct'].mean() * 100
print(f"That's a {improvement:.0f}% improvement!")

# Now let's try to find the volatility pattern user mentioned
# Maybe volatility was calculated differently - let's try range-based volatility
trades_df['entry_hour'] = trades_df['entry_time'].dt.hour

# Let's see if we can find any pattern that gives us high vol = 0.044% return
print(f"\n=== Searching for Volatility Pattern ===")

# Try using ATR with different thresholds
for low_pct in [20, 25, 30, 35, 40]:
    for high_pct in [60, 65, 70, 75, 80]:
        if high_pct - low_pct < 20:
            continue
            
        atr_low = trades_df['atr_pct'].quantile(low_pct/100)
        atr_high = trades_df['atr_pct'].quantile(high_pct/100)
        
        # Categorize
        trades_df['vol_test'] = 'medium'
        trades_df.loc[trades_df['atr_pct'] < atr_low, 'vol_test'] = 'low'
        trades_df.loc[trades_df['atr_pct'] > atr_high, 'vol_test'] = 'high'
        
        # Check if high vol gives ~0.044%
        high_vol_trades = trades_df[trades_df['vol_test'] == 'high']
        if len(high_vol_trades) > 20:
            high_vol_return = high_vol_trades['pnl_pct'].mean()
            if 0.040 < high_vol_return < 0.050:
                print(f"\nFound matching pattern!")
                print(f"Using {low_pct}th/{high_pct}th percentiles:")
                print(f"Thresholds: Low < {atr_low:.3f}%, High > {atr_high:.3f}%")
                for vol in ['low', 'medium', 'high']:
                    vol_trades = trades_df[trades_df['vol_test'] == vol]
                    if len(vol_trades) > 0:
                        print(f"{vol.capitalize()}: {len(vol_trades)} trades, "
                              f"avg: {vol_trades['pnl_pct'].mean():.3f}%, "
                              f"win rate: {(vol_trades['pnl_pct'] > 0).mean():.1%}")

# Market conditions summary
print(f"\n=== Market Conditions Summary ===")
print(f"Trend distribution: {trades_df['trend'].value_counts().to_dict()}")
print(f"VWAP position distribution: {trades_df['vwap_position'].value_counts().to_dict()}")

# Best performing conditions
print(f"\n=== Top Performing Conditions ===")
# Group by market conditions
for col in ['trend', 'vwap_position', 'hour']:
    print(f"\nBy {col}:")
    grouped = trades_df.groupby(col).agg({
        'pnl_pct': ['mean', 'count'],
    }).round(3)
    grouped.columns = ['avg_return', 'count']
    grouped = grouped[grouped['count'] >= 5].sort_values('avg_return', ascending=False)
    print(grouped.head())

# Direction performance in different conditions
print(f"\n=== Long vs Short by Market Conditions ===")
for trend in ['uptrend', 'downtrend', 'sideways']:
    trend_trades = trades_df[trades_df['trend'] == trend]
    if len(trend_trades) >= 5:
        trend_longs = trend_trades[trend_trades['direction'] == 'long']
        trend_shorts = trend_trades[trend_trades['direction'] == 'short']
        print(f"\n{trend.capitalize()}:")
        if len(trend_longs) > 0:
            print(f"  Longs: {len(trend_longs)} trades, avg: {trend_longs['pnl_pct'].mean():.3f}%")
        if len(trend_shorts) > 0:
            print(f"  Shorts: {len(trend_shorts)} trades, avg: {trend_shorts['pnl_pct'].mean():.3f}%")