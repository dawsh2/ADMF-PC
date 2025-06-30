"""Analyze overnight holding and short entry patterns"""
import pandas as pd
import numpy as np

# Load trades
trades_df = pd.read_csv('bb_trades_sensitive_trend.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

# Add time components
trades_df['entry_date'] = trades_df['entry_time'].dt.date
trades_df['exit_date'] = trades_df['exit_time'].dt.date
trades_df['entry_hour'] = trades_df['entry_time'].dt.hour
trades_df['exit_hour'] = trades_df['exit_time'].dt.hour

# Check overnight holding
trades_df['held_overnight'] = trades_df['entry_date'] != trades_df['exit_date']

print("=== Overnight Holding Analysis ===")
overnight_trades = trades_df[trades_df['held_overnight']]
print(f"Trades held overnight: {len(overnight_trades)} ({len(overnight_trades)/len(trades_df)*100:.1f}%)")

if len(overnight_trades) > 0:
    print(f"\nOvernight trades performance:")
    print(f"Average return: {overnight_trades['pnl_pct'].mean():.3f}%")
    print(f"Win rate: {(overnight_trades['pnl_pct'] > 0).mean():.1%}")
    print(f"Average bars held: {overnight_trades['bars_held'].mean():.1f}")
    
    # By direction
    overnight_longs = overnight_trades[overnight_trades['direction'] == 'long']
    overnight_shorts = overnight_trades[overnight_trades['direction'] == 'short']
    if len(overnight_longs) > 0:
        print(f"\nOvernight longs: {len(overnight_longs)} trades, avg: {overnight_longs['pnl_pct'].mean():.3f}%")
    if len(overnight_shorts) > 0:
        print(f"Overnight shorts: {len(overnight_shorts)} trades, avg: {overnight_shorts['pnl_pct'].mean():.3f}%")

# Intraday-only performance
intraday_trades = trades_df[~trades_df['held_overnight']]
print(f"\n=== Intraday-Only Performance ===")
print(f"Intraday trades: {len(intraday_trades)} ({len(intraday_trades)/len(trades_df)*100:.1f}%)")
print(f"Average return: {intraday_trades['pnl_pct'].mean():.3f}%")
print(f"Win rate: {(intraday_trades['pnl_pct'] > 0).mean():.1%}")

# Calculate annualized returns for intraday-only
trades_per_year_intraday = 772 * len(intraday_trades) / len(trades_df)
annual_return_intraday = (1 + intraday_trades['pnl_pct'].mean()/100)**trades_per_year_intraday - 1
print(f"Trades per year (intraday): {trades_per_year_intraday:.0f}")
print(f"Annualized return (intraday): {annual_return_intraday*100:.1f}%")

# With stop loss
intraday_with_stop = intraday_trades.copy()
intraday_with_stop.loc[intraday_with_stop['pnl_pct'] < -0.1, 'pnl_pct'] = -0.1
print(f"\nIntraday with -0.1% stop loss:")
print(f"Average return: {intraday_with_stop['pnl_pct'].mean():.3f}%")
annual_return_intraday_stop = (1 + intraday_with_stop['pnl_pct'].mean()/100)**trades_per_year_intraday - 1
print(f"Annualized return: {annual_return_intraday_stop*100:.1f}%")

# Impact of stop loss on holding duration
print("\n=== Stop Loss Impact on Holding Duration ===")
losers = trades_df[trades_df['pnl_pct'] < 0]
big_losers = trades_df[trades_df['pnl_pct'] < -0.1]

print(f"Current losing trades: {len(losers)}")
print(f"  Average bars held: {losers['bars_held'].mean():.1f}")
print(f"  Average loss: {losers['pnl_pct'].mean():.3f}%")

print(f"\nTrades that would hit -0.1% stop: {len(big_losers)}")
if len(big_losers) > 0:
    print(f"  Average bars held: {big_losers['bars_held'].mean():.1f}")
    print(f"  Average loss: {big_losers['pnl_pct'].mean():.3f}%")
    print(f"  These trades would be cut early, saving time and capital")

# Analyze short entries more carefully
print("\n=== Short Entry Analysis ===")
shorts = trades_df[trades_df['direction'] == 'short']

# Look at short winners vs losers
short_winners = shorts[shorts['pnl_pct'] > 0]
short_losers = shorts[shorts['pnl_pct'] < 0]

print(f"Short winners: {len(short_winners)} trades")
print(f"  Average gain: {short_winners['pnl_pct'].mean():.3f}%")
print(f"  Average bars: {short_winners['bars_held'].mean():.1f}")

print(f"\nShort losers: {len(short_losers)} trades")
print(f"  Average loss: {short_losers['pnl_pct'].mean():.3f}%")
print(f"  Average bars: {short_losers['bars_held'].mean():.1f}")

# The key insight about shorts
print("\n=== Key Insight: Short Entry Timing ===")
print("You're right - the issue isn't 'catching falling knives' in downtrends.")
print("The strategy actually does WELL going long in downtrends (0.057% avg)!")
print("\nThe problem with shorts in downtrends (-0.053% avg) suggests:")
print("1. Short entries might be catching BREAKOUTS instead of mean reversion")
print("2. When price breaks above upper BB in a downtrend, it might signal")
print("   a trend reversal rather than a mean reversion opportunity")
print("3. The RSI divergence filter isn't enough to avoid these false signals")

# Check if shorts in downtrends are at potential reversal points
downtrend_shorts = shorts[shorts['trend_new'] == 'downtrend']
if len(downtrend_shorts) > 0:
    print(f"\n=== Downtrend Short Analysis ===")
    print(f"Total downtrend shorts: {len(downtrend_shorts)}")
    print(f"Average return: {downtrend_shorts['pnl_pct'].mean():.3f}%")
    
    # Look at VWAP position for these trades
    print("\nVWAP position for downtrend shorts:")
    for vwap_pos, count in downtrend_shorts['vwap_position'].value_counts().items():
        sub_trades = downtrend_shorts[downtrend_shorts['vwap_position'] == vwap_pos]
        print(f"  {vwap_pos}: {count} trades, avg: {sub_trades['pnl_pct'].mean():.3f}%")

# Final comparison table
print("\n=== Strategy Comparison with Overnight Filter ===")
print(f"{'Strategy':<25} {'Trades':<8} {'Avg/Trade':<10} {'Annual':<10}")
print("-" * 55)

strategies = [
    ("Current (all)", len(trades_df), trades_df['pnl_pct'].mean(), 772),
    ("Intraday only", len(intraday_trades), intraday_trades['pnl_pct'].mean(), trades_per_year_intraday),
    ("Intraday + Stop", len(intraday_trades), intraday_with_stop['pnl_pct'].mean(), trades_per_year_intraday),
]

for name, trades, avg_ret, tpy in strategies:
    annual = (1 + avg_ret/100)**tpy - 1
    print(f"{name:<25} {trades:<8} {avg_ret:>6.3f}%   {annual*100:>6.1f}%")