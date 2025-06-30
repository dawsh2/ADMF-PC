"""Analyze workspace signal_generation_1c64d62f to reproduce the 0.93 bps findings"""
import pandas as pd
import numpy as np
from pathlib import Path

# This is the workspace that showed 0.93 bps for counter-trend shorts
workspace = Path("workspaces/signal_generation_1c64d62f")
signal_file = workspace / "traces/SPY_1m/signals/swing_pivot_bounce/SPY_compiled_strategy_0.parquet"

print("=== ANALYZING WORKSPACE signal_generation_1c64d62f ===")
print("This workspace previously showed 0.93 bps for counter-trend shorts in uptrends\n")

# Load sparse signal data
signals = pd.read_parquet(signal_file)
print(f"Total signal changes: {len(signals)}")

# Load SPY 1m data - we need to find the right time period
# This workspace has 81,787 bars total
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})

# Since workspace has 81,787 bars, let's find the matching subset
print(f"\nFull SPY dataset: {len(spy_1m)} bars")
print(f"Workspace data: 81,787 bars")

# The workspace likely used a subset - let's check the signal indices
max_idx = signals['idx'].max()
print(f"Maximum signal index: {max_idx}")

# Use the appropriate subset
if max_idx < len(spy_1m):
    spy_subset = spy_1m.iloc[:max_idx+5].copy()  # Add buffer
else:
    spy_subset = spy_1m.copy()

print(f"Using SPY subset: {len(spy_subset)} bars")

# Calculate all indicators as in the original analysis
spy_subset['returns'] = spy_subset['close'].pct_change()

# Volatility
spy_subset['volatility_20'] = spy_subset['returns'].rolling(20).std() * np.sqrt(390) * 100
spy_subset['vol_percentile'] = spy_subset['volatility_20'].rolling(window=390*5).rank(pct=True) * 100

# Trend
spy_subset['sma_50'] = spy_subset['close'].rolling(50).mean()
spy_subset['sma_200'] = spy_subset['close'].rolling(200).mean()
spy_subset['trend_up'] = (spy_subset['close'] > spy_subset['sma_50']) & (spy_subset['sma_50'] > spy_subset['sma_200'])
spy_subset['trend_down'] = (spy_subset['close'] < spy_subset['sma_50']) & (spy_subset['sma_50'] < spy_subset['sma_200'])
spy_subset['trend_neutral'] = ~(spy_subset['trend_up'] | spy_subset['trend_down'])

# VWAP
spy_subset['date'] = spy_subset['timestamp'].dt.date
spy_subset['typical_price'] = (spy_subset['high'] + spy_subset['low'] + spy_subset['close']) / 3
spy_subset['pv'] = spy_subset['typical_price'] * spy_subset['volume']
spy_subset['cum_pv'] = spy_subset.groupby('date')['pv'].cumsum()
spy_subset['cum_volume'] = spy_subset.groupby('date')['volume'].cumsum()
spy_subset['vwap'] = spy_subset['cum_pv'] / spy_subset['cum_volume']
spy_subset['above_vwap'] = spy_subset['close'] > spy_subset['vwap']
spy_subset['vwap_distance'] = (spy_subset['close'] - spy_subset['vwap']) / spy_subset['vwap'] * 100

# VWAP momentum
spy_subset['vwap_momentum'] = spy_subset['vwap'].pct_change(5) * 100
spy_subset['price_momentum'] = spy_subset['close'].pct_change(5) * 100
spy_subset['with_vwap_momentum'] = np.sign(spy_subset['price_momentum']) == np.sign(spy_subset['vwap_momentum'])

# Daily range
spy_subset['daily_high'] = spy_subset.groupby('date')['high'].transform('max')
spy_subset['daily_low'] = spy_subset.groupby('date')['low'].transform('min')
spy_subset['daily_range'] = (spy_subset['daily_high'] - spy_subset['daily_low']) / spy_subset['daily_low'] * 100
spy_subset['ranging_market'] = (spy_subset['daily_range'] >= 1.0) & (spy_subset['daily_range'] <= 2.0)

# Now collect trades from sparse signals
trades = []
entry_idx = None
entry_price = None
entry_signal = None

for i in range(len(signals)):
    curr = signals.iloc[i]
    
    # Entry
    if entry_idx is None and curr['val'] != 0:
        entry_idx = curr['idx']
        entry_price = curr['px']
        entry_signal = curr['val']
    
    # Exit
    elif entry_idx is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_signal)):
        if entry_idx < len(spy_subset) and curr['idx'] < len(spy_subset):
            entry_conditions = spy_subset.iloc[entry_idx]
            
            pct_return = (curr['px'] / entry_price - 1) * entry_signal * 100
            duration = curr['idx'] - entry_idx
            
            trade = {
                'entry_idx': entry_idx,
                'exit_idx': curr['idx'],
                'pct_return': pct_return,
                'direction': 'short' if entry_signal < 0 else 'long',
                'duration': duration,
                'trend_up': entry_conditions.get('trend_up', False),
                'trend_down': entry_conditions.get('trend_down', False),
                'trend_neutral': entry_conditions.get('trend_neutral', False),
                'vol_percentile': entry_conditions.get('vol_percentile', 50),
                'above_vwap': entry_conditions.get('above_vwap', False),
                'vwap_distance': entry_conditions.get('vwap_distance', 0),
                'with_vwap_momentum': entry_conditions.get('with_vwap_momentum', False),
                'ranging_market': entry_conditions.get('ranging_market', False),
                'daily_range': entry_conditions.get('daily_range', 1)
            }
            trades.append(trade)
        
        # Set up for next entry if current signal is not flat
        if curr['val'] != 0:
            entry_idx = curr['idx']
            entry_price = curr['px']
            entry_signal = curr['val']
        else:
            entry_idx = None
            entry_price = None
            entry_signal = None

trades_df = pd.DataFrame(trades)
total_days = 81787 / 390  # From metadata

print(f"\n\nTotal trades collected: {len(trades_df)}")
print(f"Trading days: {total_days:.1f}")
print(f"Average trades per day: {len(trades_df)/total_days:.1f}")

# Overall performance
print(f"\n=== OVERALL PERFORMANCE ===")
print(f"Total return: {trades_df['pct_return'].sum():.2f}%")
print(f"Average return per trade: {trades_df['pct_return'].mean():.2f} bps")
print(f"Win rate: {(trades_df['pct_return'] > 0).mean():.1%}")
print(f"Average duration: {trades_df['duration'].mean():.1f} bars")

# Performance by direction
print(f"\n=== PERFORMANCE BY DIRECTION ===")
for direction in ['long', 'short']:
    dir_trades = trades_df[trades_df['direction'] == direction]
    if len(dir_trades) > 0:
        print(f"\n{direction.upper()} trades: {len(dir_trades)}")
        print(f"  Total return: {dir_trades['pct_return'].sum():.2f}%")
        print(f"  Avg return: {dir_trades['pct_return'].mean():.2f} bps")
        print(f"  Win rate: {(dir_trades['pct_return'] > 0).mean():.1%}")

# Performance by trend
print(f"\n=== PERFORMANCE BY TREND ===")
for trend in ['trend_up', 'trend_down', 'trend_neutral']:
    trend_trades = trades_df[trades_df[trend]]
    if len(trend_trades) > 0:
        print(f"\n{trend.replace('_', ' ').title()}: {len(trend_trades)} trades")
        print(f"  Total return: {trend_trades['pct_return'].sum():.2f}%")
        print(f"  Avg per trade: {trend_trades['pct_return'].mean():.2f} bps")

# The key analysis: Counter-trend shorts in uptrends
print(f"\n=== COUNTER-TREND ANALYSIS ===")
ct_shorts_up = trades_df[(trades_df['trend_up']) & (trades_df['direction'] == 'short')]
print(f"\nCounter-trend shorts in uptrends: {len(ct_shorts_up)} trades")
if len(ct_shorts_up) > 0:
    print(f"  Average return: {ct_shorts_up['pct_return'].mean():.2f} bps")
    print(f"  Total return: {ct_shorts_up['pct_return'].sum():.2f}%")
    print(f"  Win rate: {(ct_shorts_up['pct_return'] > 0).mean():.1%}")
    print(f"  Trades per day: {len(ct_shorts_up)/total_days:.1f}")
    
    if ct_shorts_up['pct_return'].mean() >= 0.90:
        print(f"\n  ✓ MATCHES THE 0.93 BPS CLAIM!")

# Other combinations that were mentioned
print(f"\n=== OTHER KEY FILTERS ===")

# High volatility (80th+ percentile)
high_vol = trades_df[trades_df['vol_percentile'] >= 80]
if len(high_vol) > 0:
    print(f"\nHigh volatility (80th+ percentile): {len(high_vol)} trades")
    print(f"  Average return: {high_vol['pct_return'].mean():.2f} bps")
    print(f"  Trades per day: {len(high_vol)/total_days:.1f}")

# Ranging markets (1-2%)
ranging = trades_df[trades_df['ranging_market']]
if len(ranging) > 0:
    print(f"\nRanging markets (1-2%): {len(ranging)} trades")
    print(f"  Average return: {ranging['pct_return'].mean():.2f} bps")
    print(f"  Trades per day: {len(ranging)/total_days:.1f}")

# WITH VWAP momentum
with_vwap = trades_df[trades_df['with_vwap_momentum']]
if len(with_vwap) > 0:
    print(f"\nWITH VWAP momentum: {len(with_vwap)} trades")
    print(f"  Average return: {with_vwap['pct_return'].mean():.2f} bps")

# Trend-aligned vs Counter-trend
print(f"\n=== TREND ALIGNMENT ANALYSIS ===")
trend_aligned = trades_df[
    ((trades_df['trend_up']) & (trades_df['direction'] == 'long')) |
    ((trades_df['trend_down']) & (trades_df['direction'] == 'short'))
]
counter_trend = trades_df[
    ((trades_df['trend_up']) & (trades_df['direction'] == 'short')) |
    ((trades_df['trend_down']) & (trades_df['direction'] == 'long'))
]

print(f"\nTrend-aligned trades: {len(trend_aligned)}")
if len(trend_aligned) > 0:
    print(f"  Average return: {trend_aligned['pct_return'].mean():.2f} bps")
    print(f"  Total return: {trend_aligned['pct_return'].sum():.2f}%")

print(f"\nCounter-trend trades: {len(counter_trend)}")
if len(counter_trend) > 0:
    print(f"  Average return: {counter_trend['pct_return'].mean():.2f} bps")
    print(f"  Total return: {counter_trend['pct_return'].sum():.2f}%")

# Volatility analysis
print(f"\n=== VOLATILITY PERCENTILE ANALYSIS ===")
vol_bins = [0, 20, 40, 60, 80, 100]
for i in range(len(vol_bins)-1):
    vol_trades = trades_df[(trades_df['vol_percentile'] >= vol_bins[i]) & 
                          (trades_df['vol_percentile'] < vol_bins[i+1])]
    if len(vol_trades) > 0:
        print(f"{vol_bins[i]}-{vol_bins[i+1]}th percentile: "
              f"{vol_trades['pct_return'].mean():.2f} bps on {len(vol_trades)} trades")

# Best combinations
print(f"\n=== TESTING COMBINED FILTERS ===")
filters_to_test = [
    ("CT shorts + Vol>70", 
     (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
     (trades_df['vol_percentile'] > 70)),
    
    ("CT shorts + Vol>80", 
     (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
     (trades_df['vol_percentile'] > 80)),
    
    ("CT shorts + Ranging", 
     (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
     (trades_df['ranging_market'])),
    
    ("High Vol + Ranging", 
     (trades_df['vol_percentile'] > 80) & (trades_df['ranging_market'])),
    
    ("CT shorts + WITH VWAP", 
     (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
     (trades_df['with_vwap_momentum']))
]

for filter_name, filter_mask in filters_to_test:
    filtered = trades_df[filter_mask]
    if len(filtered) >= 5:
        print(f"\n{filter_name}: {len(filtered)} trades")
        print(f"  Average return: {filtered['pct_return'].mean():.2f} bps")
        print(f"  Trades per day: {len(filtered)/total_days:.1f}")
        print(f"  Win rate: {(filtered['pct_return'] > 0).mean():.1%}")

print("\n\nCONCLUSIONS:")
print("Comparing workspace signal_generation_1c64d62f results with previous claims...")