"""Analyze Keltner Bands strategy performance"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_d5807cc2")
signal_file = workspace / "traces/SPY_1m/signals/keltner_bands/SPY_compiled_strategy_0.parquet"

print("=== KELTNER BANDS STRATEGY ANALYSIS ===\n")

# Load signals
signals = pd.read_parquet(signal_file)
print(f"Total signal changes: {len(signals)}")

# Load SPY 1m data
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})

# Use subset matching the workspace
spy_subset = spy_1m.iloc[:81787].copy()
print(f"Analyzing {len(spy_subset)} bars of 1-minute SPY data")
print(f"Date range: {spy_subset['timestamp'].min()} to {spy_subset['timestamp'].max()}")

# Calculate indicators for analysis
spy_subset['returns'] = spy_subset['close'].pct_change()
spy_subset['volatility_20'] = spy_subset['returns'].rolling(20).std() * np.sqrt(390) * 100
spy_subset['vol_percentile'] = spy_subset['volatility_20'].rolling(window=390*5).rank(pct=True) * 100

# Trend
spy_subset['sma_50'] = spy_subset['close'].rolling(50).mean()
spy_subset['sma_200'] = spy_subset['close'].rolling(200).mean()
spy_subset['trend_up'] = (spy_subset['close'] > spy_subset['sma_50']) & (spy_subset['sma_50'] > spy_subset['sma_200'])
spy_subset['trend_down'] = (spy_subset['close'] < spy_subset['sma_50']) & (spy_subset['sma_50'] < spy_subset['sma_200'])

# ATR for volatility context
spy_subset['tr'] = np.maximum(
    spy_subset['high'] - spy_subset['low'],
    np.maximum(
        abs(spy_subset['high'] - spy_subset['close'].shift(1)),
        abs(spy_subset['low'] - spy_subset['close'].shift(1))
    )
)
spy_subset['atr_20'] = spy_subset['tr'].rolling(20).mean()

# Collect trades from sparse signals
trades = []
entry_data = None

for i in range(len(signals)):
    curr = signals.iloc[i]
    
    if entry_data is None and curr['val'] != 0:
        entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
    
    elif entry_data is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])):
        if entry_data['idx'] < len(spy_subset) and curr['idx'] < len(spy_subset):
            entry_conditions = spy_subset.iloc[entry_data['idx']]
            
            pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
            duration = curr['idx'] - entry_data['idx']
            
            trade = {
                'pct_return': pct_return,
                'direction': 'short' if entry_data['signal'] < 0 else 'long',
                'duration': duration,
                'trend_up': entry_conditions.get('trend_up', False),
                'trend_down': entry_conditions.get('trend_down', False),
                'vol_percentile': entry_conditions.get('vol_percentile', 50),
                'atr': entry_conditions.get('atr_20', 0),
                'entry_idx': entry_data['idx'],
                'exit_idx': curr['idx']
            }
            trades.append(trade)
        
        if curr['val'] != 0:
            entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
        else:
            entry_data = None

trades_df = pd.DataFrame(trades)
total_days = 81787 / 390

print(f"\n\nTotal trades: {len(trades_df)}")
print(f"Trading days: {total_days:.1f}")
print(f"Average trades per day: {len(trades_df)/total_days:.1f}")

# Overall performance
print(f"\n=== OVERALL PERFORMANCE ===")
print(f"Total return: {trades_df['pct_return'].sum():.2f}%")
print(f"Average return per trade: {trades_df['pct_return'].mean():.2f} bps")
print(f"Median return: {trades_df['pct_return'].median():.2f} bps")
print(f"Win rate: {(trades_df['pct_return'] > 0).mean():.1%}")
print(f"Average duration: {trades_df['duration'].mean():.1f} bars ({trades_df['duration'].mean():.0f} minutes)")

# Risk metrics
print(f"\n=== RISK METRICS ===")
print(f"Std dev of returns: {trades_df['pct_return'].std():.2f} bps")
print(f"Sharpe ratio: {trades_df['pct_return'].mean() / trades_df['pct_return'].std():.2f}")
print(f"Max winner: {trades_df['pct_return'].max():.2f} bps")
print(f"Max loser: {trades_df['pct_return'].min():.2f} bps")
print(f"Win/Loss ratio: {abs(trades_df[trades_df['pct_return'] > 0]['pct_return'].mean() / trades_df[trades_df['pct_return'] < 0]['pct_return'].mean()):.2f}")

# Performance by direction
print(f"\n=== PERFORMANCE BY DIRECTION ===")
for direction in ['long', 'short']:
    dir_trades = trades_df[trades_df['direction'] == direction]
    if len(dir_trades) > 0:
        print(f"\n{direction.upper()} trades: {len(dir_trades)} ({len(dir_trades)/len(trades_df)*100:.1f}%)")
        print(f"  Average return: {dir_trades['pct_return'].mean():.2f} bps")
        print(f"  Total return: {dir_trades['pct_return'].sum():.2f}%")
        print(f"  Win rate: {(dir_trades['pct_return'] > 0).mean():.1%}")
        print(f"  Avg duration: {dir_trades['duration'].mean():.0f} bars")

# Performance by market regime
print(f"\n=== PERFORMANCE BY MARKET REGIME ===")
for trend, trend_name in [(True, "Uptrend"), (False, "Not Uptrend")]:
    trend_trades = trades_df[trades_df['trend_up'] == trend]
    if len(trend_trades) > 0:
        print(f"\n{trend_name}: {len(trend_trades)} trades")
        print(f"  Average return: {trend_trades['pct_return'].mean():.2f} bps")
        print(f"  Win rate: {(trend_trades['pct_return'] > 0).mean():.1%}")

# Volatility analysis
print(f"\n=== VOLATILITY ANALYSIS ===")
vol_bins = [0, 20, 40, 60, 80, 100]
for i in range(len(vol_bins)-1):
    vol_trades = trades_df[(trades_df['vol_percentile'] >= vol_bins[i]) & 
                          (trades_df['vol_percentile'] < vol_bins[i+1])]
    if len(vol_trades) > 0:
        print(f"{vol_bins[i]}-{vol_bins[i+1]}th percentile: "
              f"{vol_trades['pct_return'].mean():.2f} bps on {len(vol_trades)} trades "
              f"({len(vol_trades)/total_days:.1f} tpd)")

# Duration analysis
print(f"\n=== DURATION ANALYSIS ===")
duration_bins = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 1000)]
for min_dur, max_dur in duration_bins:
    dur_trades = trades_df[(trades_df['duration'] >= min_dur) & (trades_df['duration'] < max_dur)]
    if len(dur_trades) > 0:
        print(f"{min_dur}-{max_dur} bars: {dur_trades['pct_return'].mean():.2f} bps on "
              f"{len(dur_trades)} trades ({len(dur_trades)/len(trades_df)*100:.1f}%)")

# Best and worst trades
print(f"\n=== BEST AND WORST TRADES ===")
best_trades = trades_df.nlargest(5, 'pct_return')
worst_trades = trades_df.nsmallest(5, 'pct_return')

print("\nTop 5 winners:")
for _, trade in best_trades.iterrows():
    print(f"  {trade['pct_return']:.2f} bps, {trade['direction']}, {trade['duration']} bars")

print("\nTop 5 losers:")
for _, trade in worst_trades.iterrows():
    print(f"  {trade['pct_return']:.2f} bps, {trade['direction']}, {trade['duration']} bars")

# Filter recommendations
print(f"\n=== FILTER RECOMMENDATIONS ===")

# Test various filters
filters_to_test = [
    ("Longs only", trades_df['direction'] == 'long'),
    ("Shorts only", trades_df['direction'] == 'short'),
    ("Vol > 70", trades_df['vol_percentile'] > 70),
    ("Vol > 80", trades_df['vol_percentile'] > 80),
    ("Quick exits (< 10 bars)", trades_df['duration'] < 10),
    ("Uptrend only", trades_df['trend_up']),
    ("Downtrend only", trades_df['trend_down']),
    ("Longs in uptrend", (trades_df['direction'] == 'long') & (trades_df['trend_up'])),
    ("Shorts in downtrend", (trades_df['direction'] == 'short') & (trades_df['trend_down'])),
    ("High vol + Quick", (trades_df['vol_percentile'] > 70) & (trades_df['duration'] < 10))
]

best_filters = []
for filter_name, filter_mask in filters_to_test:
    filtered = trades_df[filter_mask]
    if len(filtered) >= 50:  # Minimum trades for reliability
        edge = filtered['pct_return'].mean()
        tpd = len(filtered) / total_days
        win_rate = (filtered['pct_return'] > 0).mean()
        
        if edge > 0.5:  # Look for meaningful edge
            best_filters.append((filter_name, edge, tpd, win_rate))
            print(f"\n{filter_name}:")
            print(f"  Edge: {edge:.2f} bps")
            print(f"  Trades/day: {tpd:.1f}")
            print(f"  Win rate: {win_rate:.1%}")

print(f"\n\n=== CONCLUSIONS ===")
print(f"Keltner Bands on 1-minute SPY:")
print(f"- Baseline: {trades_df['pct_return'].mean():.2f} bps with {len(trades_df)/total_days:.1f} trades/day")
print(f"- Win rate: {(trades_df['pct_return'] > 0).mean():.1%}")
print(f"- Best direction: {'Long' if trades_df[trades_df['direction'] == 'long']['pct_return'].mean() > trades_df[trades_df['direction'] == 'short']['pct_return'].mean() else 'Short'}")

if best_filters:
    print(f"\nPromising filters found:")
    for name, edge, tpd, wr in sorted(best_filters, key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {name}: {edge:.2f} bps on {tpd:.1f} tpd")
else:
    print(f"\nNo filters found with meaningful edge (>0.5 bps)")
    
print(f"\nFor your goal of >=1 bps with 2-3+ trades/day:")
if any(edge >= 1.0 and tpd >= 2.0 for _, edge, tpd, _ in best_filters):
    print("✓ Keltner Bands can potentially meet your requirements with filters")
else:
    print("✗ Keltner Bands does not meet your requirements")
    print("  Consider different parameters or strategies")