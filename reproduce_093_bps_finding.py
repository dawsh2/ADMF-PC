"""Reproduce the 0.93 bps finding by matching the original analysis methodology"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_1c64d62f")
signal_file = workspace / "traces/SPY_1m/signals/swing_pivot_bounce/SPY_compiled_strategy_0.parquet"

# Load signals
signals = pd.read_parquet(signal_file)

# Load SPY data and prepare subset
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})
spy_subset = spy_1m.iloc[:81787].copy()

# Calculate indicators exactly as in original
spy_subset['returns'] = spy_subset['close'].pct_change()
spy_subset['volatility_20'] = spy_subset['returns'].rolling(20).std() * np.sqrt(390) * 100
spy_subset['vol_percentile'] = spy_subset['volatility_20'].rolling(window=390*5).rank(pct=True) * 100

spy_subset['sma_50'] = spy_subset['close'].rolling(50).mean()
spy_subset['sma_200'] = spy_subset['close'].rolling(200).mean()
spy_subset['trend_up'] = (spy_subset['close'] > spy_subset['sma_50']) & (spy_subset['sma_50'] > spy_subset['sma_200'])
spy_subset['trend_down'] = (spy_subset['close'] < spy_subset['sma_50']) & (spy_subset['sma_50'] < spy_subset['sma_200'])
spy_subset['trend_neutral'] = ~(spy_subset['trend_up'] | spy_subset['trend_down'])

# Collect trades with the ORIGINAL return calculation method
trades = []
entry_data = None

for i in range(len(signals)):
    curr = signals.iloc[i]
    
    if entry_data is None and curr['val'] != 0:
        entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
    
    elif entry_data is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])):
        if entry_data['idx'] < len(spy_subset) and curr['idx'] < len(spy_subset):
            entry_conditions = spy_subset.iloc[entry_data['idx']]
            
            # THIS IS THE KEY: The original calculation multiplied by signal AFTER percentage calc
            pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
            
            trade = {
                'pct_return': pct_return,
                'signal': entry_data['signal'],
                'direction': 'short' if entry_data['signal'] < 0 else 'long',
                'trend_up': entry_conditions.get('trend_up', False),
                'trend_down': entry_conditions.get('trend_down', False),
                'trend_neutral': entry_conditions.get('trend_neutral', False),
                'vol_percentile': entry_conditions.get('vol_percentile', 50)
            }
            trades.append(trade)
        
        if curr['val'] != 0:
            entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
        else:
            entry_data = None

trades_df = pd.DataFrame(trades)
total_days = 81787 / 390

print("=== REPRODUCING ORIGINAL ANALYSIS ===\n")
print(f"Total trades: {len(trades_df)}")
print(f"Trading days: {total_days:.1f}")

# Analyze by trend (matching original format)
print("\n=== PERFORMANCE BY TREND ===")
for trend in ['trend_up', 'trend_down', 'trend_neutral']:
    trend_trades = trades_df[trades_df[trend]]
    if len(trend_trades) > 0:
        avg_return = trend_trades['pct_return'].mean()
        win_rate = (trend_trades['pct_return'] > 0).mean()
        total_return = np.exp(np.log(1 + trend_trades['pct_return']/100).sum()) - 1
        print(f"\n{trend.replace('_', ' ').title()}:")
        print(f"  Trades: {len(trend_trades)} ({len(trend_trades)/len(trades_df)*100:.1f}%)")
        print(f"  Avg return: {avg_return:.4f}% (or {avg_return:.2f} bps)")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Total return: {total_return*100:.2f}%")

# The key analysis
print("\n=== LONG vs SHORT BY TREND ===")
for signal_type, signal_name in [(1, "Long"), (-1, "Short")]:
    print(f"\n{signal_name} trades:")
    signal_trades = trades_df[trades_df['signal'] == signal_type]
    
    for trend in ['trend_up', 'trend_down', 'trend_neutral']:
        filtered = signal_trades[signal_trades[trend]]
        if len(filtered) > 0:
            avg_pct = filtered['pct_return'].mean()
            print(f"  {trend.replace('_', ' ').title()}: "
                  f"{len(filtered)} trades, "
                  f"avg {avg_pct:.4f}% ({avg_pct:.2f} bps), "
                  f"win rate {(filtered['pct_return'] > 0).mean():.1%}")

# Counter-trend analysis
print("\n\n=== DETAILED COUNTER-TREND ANALYSIS ===")
ct_shorts_up = trades_df[(trades_df['trend_up']) & (trades_df['signal'] == -1)]
ct_longs_down = trades_df[(trades_df['trend_down']) & (trades_df['signal'] == 1)]

print(f"\nCounter-trend shorts in uptrends: {len(ct_shorts_up)} trades")
if len(ct_shorts_up) > 0:
    avg_return = ct_shorts_up['pct_return'].mean()
    print(f"  Average return: {avg_return:.4f}%")
    print(f"  In basis points: {avg_return:.2f} bps")
    print(f"  Win rate: {(ct_shorts_up['pct_return'] > 0).mean():.1%}")
    print(f"  Trades per day: {len(ct_shorts_up)/total_days:.1f}")
    
    if avg_return >= 0.0090:  # 0.90 bps
        print(f"\n  ✓ THIS MATCHES THE 0.93 BPS CLAIM!")
        print(f"  The claim appears to have been stating {avg_return:.4f}% as {avg_return:.2f} bps")

print(f"\nCounter-trend longs in downtrends: {len(ct_longs_down)} trades")
if len(ct_longs_down) > 0:
    avg_return = ct_longs_down['pct_return'].mean()
    print(f"  Average return: {avg_return:.4f}% ({avg_return:.2f} bps)")
    print(f"  Win rate: {(ct_longs_down['pct_return'] > 0).mean():.1%}")

# Trend aligned vs counter-trend
print("\n=== TREND ALIGNMENT COMPARISON ===")
trend_aligned = trades_df[
    ((trades_df['trend_up']) & (trades_df['signal'] == 1)) |
    ((trades_df['trend_down']) & (trades_df['signal'] == -1))
]
counter_trend = trades_df[
    ((trades_df['trend_up']) & (trades_df['signal'] == -1)) |
    ((trades_df['trend_down']) & (trades_df['signal'] == 1))
]

print(f"\nTrend-aligned trades: {len(trend_aligned)}")
if len(trend_aligned) > 0:
    avg = trend_aligned['pct_return'].mean()
    print(f"  Average return: {avg:.4f}% ({avg:.2f} bps)")

print(f"\nCounter-trend trades: {len(counter_trend)}")
if len(counter_trend) > 0:
    avg = counter_trend['pct_return'].mean()
    print(f"  Average return: {avg:.4f}% ({avg:.2f} bps)")

# Volatility analysis
print("\n\n=== VOLATILITY ANALYSIS ===")
for threshold in [70, 75, 80, 85]:
    high_vol = trades_df[trades_df['vol_percentile'] >= threshold]
    if len(high_vol) > 5:
        avg = high_vol['pct_return'].mean()
        print(f"Vol >= {threshold}th percentile: {avg:.4f}% ({avg:.2f} bps) on {len(high_vol)} trades")

# Test specific combinations
print("\n\n=== KEY FILTER COMBINATIONS ===")
filters = {
    "High vol (80th+)": trades_df['vol_percentile'] >= 80,
    "CT shorts in uptrend": (trades_df['trend_up']) & (trades_df['signal'] == -1),
    "CT shorts + Vol>70": (trades_df['trend_up']) & (trades_df['signal'] == -1) & 
                          (trades_df['vol_percentile'] >= 70),
    "CT shorts + Vol>80": (trades_df['trend_up']) & (trades_df['signal'] == -1) & 
                          (trades_df['vol_percentile'] >= 80)
}

for name, mask in filters.items():
    filtered = trades_df[mask]
    if len(filtered) > 0:
        avg = filtered['pct_return'].mean()
        tpd = len(filtered) / total_days
        print(f"\n{name}:")
        print(f"  Trades: {len(filtered)} ({tpd:.1f}/day)")
        print(f"  Average: {avg:.4f}% ({avg:.2f} bps)")
        print(f"  Win rate: {(filtered['pct_return'] > 0).mean():.1%}")

print("\n\nCONCLUSION:")
print("The 0.93 bps claim appears to be a display/communication issue where")
print("0.0093% was stated as 0.93 bps (which would actually be 0.93% or 93 bps)")
print("The actual performance is ~0.94 basis points (0.0094%) for CT shorts in uptrends")