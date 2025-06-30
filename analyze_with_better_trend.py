"""Analyze with more sensitive trend detection and calculate annualized returns"""
import pandas as pd
import numpy as np

# Load market data first
print("Loading market data for better trend detection...")
market_df = pd.read_csv('data/SPY_1m.csv')
market_df['timestamp'] = pd.to_datetime(market_df['timestamp'], utc=True)

# Get test portion
test_start_idx = int(len(market_df) * 0.8)
test_df = market_df.iloc[test_start_idx:].copy()
test_df.set_index('timestamp', inplace=True)

# More sensitive trend detection using multiple methods
print("Calculating more sensitive trend indicators...")

# Method 1: Shorter moving averages
test_df['SMA10'] = test_df['Close'].rolling(10).mean()
test_df['SMA20'] = test_df['Close'].rolling(20).mean()
test_df['SMA50'] = test_df['Close'].rolling(50).mean()

# Method 2: Rate of change
test_df['ROC20'] = test_df['Close'].pct_change(20) * 100  # 20-bar rate of change
test_df['ROC50'] = test_df['Close'].pct_change(50) * 100  # 50-bar rate of change

# Method 3: Linear regression slope
def calculate_slope(series, window):
    """Calculate rolling linear regression slope"""
    slopes = pd.Series(index=series.index, dtype=float)
    for i in range(window, len(series)):
        y = series.iloc[i-window:i].values
        x = np.arange(window)
        if len(y) == window:
            slope = np.polyfit(x, y, 1)[0]
            slopes.iloc[i] = slope / series.iloc[i] * 100  # Normalize as percentage
    return slopes

test_df['slope20'] = calculate_slope(test_df['Close'], 20)
test_df['slope50'] = calculate_slope(test_df['Close'], 50)

# Combine methods for trend classification
test_df['trend_sensitive'] = 'sideways'

# Uptrend conditions (any of these)
uptrend_conditions = (
    (test_df['SMA10'] > test_df['SMA20']) & (test_df['SMA20'] > test_df['SMA50']) |  # Classic MA alignment
    (test_df['ROC20'] > 0.5) |  # 20-bar momentum > 0.5%
    (test_df['slope20'] > 0.1)   # 20-bar slope > 0.1% per bar
)

# Downtrend conditions
downtrend_conditions = (
    (test_df['SMA10'] < test_df['SMA20']) & (test_df['SMA20'] < test_df['SMA50']) |  # Classic MA alignment
    (test_df['ROC20'] < -0.5) |  # 20-bar momentum < -0.5%
    (test_df['slope20'] < -0.1)   # 20-bar slope < -0.1% per bar
)

test_df.loc[uptrend_conditions, 'trend_sensitive'] = 'uptrend'
test_df.loc[downtrend_conditions, 'trend_sensitive'] = 'downtrend'

# Load trades and merge with new trend data
trades_df = pd.read_csv('bb_trades_fresh_indicators.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

# Update trend for each trade
print("\nUpdating trades with sensitive trend detection...")
for idx, trade in trades_df.iterrows():
    entry_time = trade['entry_time']
    try:
        market_data = test_df.loc[entry_time]
        trades_df.loc[idx, 'trend_new'] = market_data['trend_sensitive']
    except:
        # Find nearest
        nearest_idx = test_df.index.get_indexer([entry_time], method='nearest')[0]
        if abs(test_df.index[nearest_idx] - entry_time) < pd.Timedelta(minutes=5):
            trades_df.loc[idx, 'trend_new'] = test_df.iloc[nearest_idx]['trend_sensitive']
        else:
            trades_df.loc[idx, 'trend_new'] = 'unknown'

# Analyze with new trend
print("\n=== Performance by Sensitive Trend Detection ===")
trend_counts = trades_df['trend_new'].value_counts()
print(f"Trend distribution: {trend_counts.to_dict()}")
print(f"Percentage in trends: {(trend_counts['uptrend'] + trend_counts['downtrend']) / len(trades_df) * 100:.1f}%")

for trend in ['uptrend', 'downtrend', 'sideways']:
    trend_trades = trades_df[trades_df['trend_new'] == trend]
    if len(trend_trades) > 0:
        print(f"\n{trend.capitalize()}: {len(trend_trades)} trades")
        print(f"  Average return: {trend_trades['pnl_pct'].mean():.3f}%")
        print(f"  Win rate: {(trend_trades['pnl_pct'] > 0).mean():.1%}")
        
        # By direction
        trend_longs = trend_trades[trend_trades['direction'] == 'long']
        trend_shorts = trend_trades[trend_trades['direction'] == 'short']
        if len(trend_longs) > 0:
            print(f"  Longs: {len(trend_longs)} trades, avg: {trend_longs['pnl_pct'].mean():.3f}%")
        if len(trend_shorts) > 0:
            print(f"  Shorts: {len(trend_shorts)} trades, avg: {trend_shorts['pnl_pct'].mean():.3f}%")

# Calculate annualized returns
print("\n=== Annualized Return Calculations ===")

# Current performance
avg_return_current = trades_df['pnl_pct'].mean() / 100  # Convert to decimal
print(f"Current average return per trade: {avg_return_current*100:.3f}%")

# With stop loss
avg_return_stop = 0.034 / 100  # 0.034% as decimal
print(f"With -0.1% stop loss: {avg_return_stop*100:.3f}%")

# Calculate trades per year
total_days = (trades_df['entry_time'].max() - trades_df['entry_time'].min()).days
trades_per_day = len(trades_df) / total_days
trades_per_year = trades_per_day * 252  # Trading days

print(f"\nTrade frequency:")
print(f"Total trading days: {total_days}")
print(f"Trades per day: {trades_per_day:.2f}")
print(f"Trades per year (252 days): {trades_per_year:.0f}")

# Calculate annualized returns using compound formula
# (1 + avg_return)^trades_per_year - 1
annual_return_current = (1 + avg_return_current) ** trades_per_year - 1
annual_return_stop = (1 + avg_return_stop) ** trades_per_year - 1

print(f"\nAnnualized returns (compounded):")
print(f"Current strategy: {annual_return_current*100:.1f}%")
print(f"With -0.1% stop loss: {annual_return_stop*100:.1f}%")

# Also show simple multiplication (non-compounded)
annual_simple_current = avg_return_current * trades_per_year
annual_simple_stop = avg_return_stop * trades_per_year

print(f"\nAnnualized returns (simple):")
print(f"Current strategy: {annual_simple_current*100:.1f}%")
print(f"With -0.1% stop loss: {annual_simple_stop*100:.1f}%")

# What about transaction costs?
print(f"\n=== Impact of Transaction Costs ===")
for cost_bps in [0, 5, 10, 20]:  # basis points per trade (both sides)
    cost_pct = cost_bps / 10000  # Convert bps to percentage
    net_return_current = avg_return_current - cost_pct
    net_return_stop = avg_return_stop - cost_pct
    
    if net_return_current > 0:
        annual_net_current = (1 + net_return_current) ** trades_per_year - 1
        annual_net_stop = (1 + net_return_stop) ** trades_per_year - 1
        
        print(f"\nWith {cost_bps} bps round-trip cost:")
        print(f"  Current: {annual_net_current*100:.1f}% annual")
        print(f"  With stop: {annual_net_stop*100:.1f}% annual")

# Save updated data
trades_df.to_csv('bb_trades_sensitive_trend.csv', index=False)
print(f"\nSaved {len(trades_df)} trades with sensitive trend detection")