"""Analyze Keltner Bands with 0.3% stop loss AND filters"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_d5807cc2")
signal_file = workspace / "traces/SPY_1m/signals/keltner_bands/SPY_compiled_strategy_0.parquet"

print("=== KELTNER BANDS WITH 0.3% STOP + FILTERS ===\n")

# Load signals
signals = pd.read_parquet(signal_file)

# Load SPY 1m data
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})
spy_subset = spy_1m.iloc[:81787].copy()

# Calculate comprehensive indicators
spy_subset['returns'] = spy_subset['close'].pct_change()

# Volatility
spy_subset['volatility_20'] = spy_subset['returns'].rolling(20).std() * np.sqrt(390) * 100
spy_subset['vol_percentile_20'] = spy_subset['volatility_20'].rolling(window=390*5).rank(pct=True) * 100

# Trend
spy_subset['sma_50'] = spy_subset['close'].rolling(50).mean()
spy_subset['sma_200'] = spy_subset['close'].rolling(200).mean()
spy_subset['trend_up'] = (spy_subset['close'] > spy_subset['sma_50']) & (spy_subset['sma_50'] > spy_subset['sma_200'])
spy_subset['trend_down'] = (spy_subset['close'] < spy_subset['sma_50']) & (spy_subset['sma_50'] < spy_subset['sma_200'])

# VWAP
spy_subset['date'] = spy_subset['timestamp'].dt.date
spy_subset['typical_price'] = (spy_subset['high'] + spy_subset['low'] + spy_subset['close']) / 3
spy_subset['pv'] = spy_subset['typical_price'] * spy_subset['volume']
spy_subset['cum_pv'] = spy_subset.groupby('date')['pv'].cumsum()
spy_subset['cum_volume'] = spy_subset.groupby('date')['volume'].cumsum()
spy_subset['vwap'] = spy_subset['cum_pv'] / spy_subset['cum_volume']
spy_subset['above_vwap'] = spy_subset['close'] > spy_subset['vwap']

# Volume
spy_subset['volume_sma_20'] = spy_subset['volume'].rolling(20).mean()
spy_subset['volume_ratio'] = spy_subset['volume'] / spy_subset['volume_sma_20']

# Time of day
spy_subset['hour'] = spy_subset['timestamp'].dt.hour

def analyze_with_stop_and_filter(filter_mask, filter_name, stop_pct=0.003):
    """Analyze performance with 0.3% stop and a specific filter"""
    trades = []
    entry_data = None
    
    for i in range(len(signals)):
        curr = signals.iloc[i]
        
        if entry_data is None and curr['val'] != 0:
            entry_idx = curr['idx']
            if entry_idx < len(spy_subset):
                # Check if this entry passes the filter
                if not filter_mask.iloc[entry_idx]:
                    continue  # Skip this entry
                    
                entry_data = {
                    'idx': entry_idx,
                    'price': curr['px'],
                    'signal': curr['val']
                }
        
        elif entry_data is not None:
            # Check each bar for stop loss
            start_idx = entry_data['idx'] + 1
            end_idx = min(curr['idx'] + 1, len(spy_subset))
            
            stopped_out = False
            exit_idx = curr['idx']
            exit_price = curr['px']
            
            for check_idx in range(start_idx, end_idx):
                if check_idx >= len(spy_subset):
                    break
                    
                check_bar = spy_subset.iloc[check_idx]
                
                # Check 0.3% stop
                if entry_data['signal'] > 0:  # Long
                    stop_level = entry_data['price'] * (1 - stop_pct)
                    if check_bar['low'] <= stop_level:
                        stopped_out = True
                        exit_price = stop_level
                        exit_idx = check_idx
                        break
                else:  # Short
                    stop_level = entry_data['price'] * (1 + stop_pct)
                    if check_bar['high'] >= stop_level:
                        stopped_out = True
                        exit_price = stop_level
                        exit_idx = check_idx
                        break
            
            # Process exit
            should_exit = stopped_out or curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])
            
            if should_exit and entry_data['idx'] < len(spy_subset):
                entry_conditions = spy_subset.iloc[entry_data['idx']]
                pct_return = (exit_price / entry_data['price'] - 1) * entry_data['signal'] * 100
                duration = exit_idx - entry_data['idx']
                
                trade = {
                    'pct_return': pct_return,
                    'direction': 'short' if entry_data['signal'] < 0 else 'long',
                    'duration': duration,
                    'stopped_out': stopped_out,
                    'vol_percentile': entry_conditions.get('vol_percentile_20', 50),
                    'trend_up': entry_conditions.get('trend_up', False),
                    'above_vwap': entry_conditions.get('above_vwap', False),
                    'volume_ratio': entry_conditions.get('volume_ratio', 1)
                }
                trades.append(trade)
                
                # Check next signal
                if curr['val'] != 0 and not stopped_out:
                    entry_idx = curr['idx']
                    if entry_idx < len(spy_subset) and filter_mask.iloc[entry_idx]:
                        entry_data = {
                            'idx': entry_idx,
                            'price': curr['px'],
                            'signal': curr['val']
                        }
                    else:
                        entry_data = None
                else:
                    entry_data = None
    
    return pd.DataFrame(trades)

# Test filters that showed promise
filters_to_test = [
    ("No filter (0.3% stop only)", pd.Series([True] * len(spy_subset))),
    ("High Vol (>70%)", spy_subset['vol_percentile_20'] > 70),
    ("High Vol (>80%)", spy_subset['vol_percentile_20'] > 80),
    ("Longs only", pd.Series([True] * len(spy_subset))),  # Applied in processing
    ("Shorts only", pd.Series([True] * len(spy_subset))),  # Applied in processing
    ("Quick exits (<5 bars)", pd.Series([True] * len(spy_subset))),  # Check in results
    ("High Vol + Longs", spy_subset['vol_percentile_20'] > 70),
    ("High Vol + Shorts", spy_subset['vol_percentile_20'] > 70),
    ("Longs Below VWAP", ~spy_subset['above_vwap']),
    ("Shorts Above VWAP", spy_subset['above_vwap']),
    ("High Volume (>1.5x)", spy_subset['volume_ratio'] > 1.5),
    ("Morning (9:30-11am)", (spy_subset['hour'] >= 9) & (spy_subset['hour'] < 11)),
    ("Afternoon (2-4pm)", (spy_subset['hour'] >= 14) & (spy_subset['hour'] < 16)),
]

total_days = 81787 / 390
results = []

for filter_name, filter_mask in filters_to_test:
    trades_df = analyze_with_stop_and_filter(filter_mask, filter_name)
    
    # Apply additional filters in post-processing
    if "Longs only" in filter_name:
        trades_df = trades_df[trades_df['direction'] == 'long']
    elif "Shorts only" in filter_name:
        trades_df = trades_df[trades_df['direction'] == 'short']
    elif "Quick exits" in filter_name:
        trades_df = trades_df[trades_df['duration'] < 5]
    elif "High Vol + Longs" in filter_name:
        trades_df = trades_df[trades_df['direction'] == 'long']
    elif "High Vol + Shorts" in filter_name:
        trades_df = trades_df[trades_df['direction'] == 'short']
    elif "Longs Below VWAP" in filter_name:
        trades_df = trades_df[trades_df['direction'] == 'long']
    elif "Shorts Above VWAP" in filter_name:
        trades_df = trades_df[trades_df['direction'] == 'short']
    
    if len(trades_df) >= 30:  # Minimum trades for reliability
        avg_return = trades_df['pct_return'].mean()
        win_rate = (trades_df['pct_return'] > 0).mean()
        tpd = len(trades_df) / total_days
        stop_rate = trades_df['stopped_out'].mean() if len(trades_df) > 0 else 0
        
        results.append({
            'filter': filter_name,
            'avg_return_pct': avg_return,
            'avg_return_bps': avg_return * 100,  # Convert to actual basis points
            'win_rate': win_rate,
            'trades_per_day': tpd,
            'total_trades': len(trades_df),
            'stop_rate': stop_rate
        })

# Display results
print("Results for each filter with 0.3% stop loss:\n")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('avg_return_bps', ascending=False)

print(f"{'Filter':<30} {'Edge (bps)':<12} {'Win Rate':<10} {'Trades/Day':<12} {'Stop Rate':<10}")
print("-" * 84)

for _, row in results_df.iterrows():
    print(f"{row['filter']:<30} {row['avg_return_bps']:>10.2f} {row['win_rate']:>9.1%} "
          f"{row['trades_per_day']:>11.1f} {row['stop_rate']:>9.1%}")

# Combined filters - test the most promising combinations
print("\n\n=== TESTING COMBINED FILTERS ===\n")

combined_filters = [
    ("Quick Longs + High Vol>70", 
     (spy_subset['vol_percentile_20'] > 70), "long", 5),
    ("Quick Shorts + High Vol>70", 
     (spy_subset['vol_percentile_20'] > 70), "short", 5),
    ("Quick Longs + Vol>80", 
     (spy_subset['vol_percentile_20'] > 80), "long", 5),
    ("Morning Longs + High Vol", 
     (spy_subset['hour'] >= 9) & (spy_subset['hour'] < 11) & (spy_subset['vol_percentile_20'] > 70), "long", None),
    ("Afternoon Shorts + Low Vol", 
     (spy_subset['hour'] >= 14) & (spy_subset['hour'] < 16) & (spy_subset['vol_percentile_20'] < 40), "short", None),
    ("Longs Below VWAP + High Vol", 
     (~spy_subset['above_vwap']) & (spy_subset['vol_percentile_20'] > 70), "long", None),
    ("Shorts Above VWAP + High Vol", 
     (spy_subset['above_vwap']) & (spy_subset['vol_percentile_20'] > 70), "short", None),
]

combined_results = []

for filter_name, filter_mask, direction_filter, duration_filter in combined_filters:
    trades_df = analyze_with_stop_and_filter(filter_mask, filter_name)
    
    # Apply direction filter
    if direction_filter and len(trades_df) > 0 and 'direction' in trades_df.columns:
        trades_df = trades_df[trades_df['direction'] == direction_filter]
    
    # Apply duration filter
    if duration_filter and len(trades_df) > 0 and 'duration' in trades_df.columns:
        trades_df = trades_df[trades_df['duration'] < duration_filter]
    
    if len(trades_df) >= 20:  # Lower threshold for combined filters
        avg_return = trades_df['pct_return'].mean()
        win_rate = (trades_df['pct_return'] > 0).mean()
        tpd = len(trades_df) / total_days
        stop_rate = trades_df['stopped_out'].mean() if len(trades_df) > 0 else 0
        
        combined_results.append({
            'filter': filter_name,
            'avg_return_bps': avg_return * 100,
            'win_rate': win_rate,
            'trades_per_day': tpd,
            'total_trades': len(trades_df),
            'stop_rate': stop_rate
        })

if combined_results:
    combined_df = pd.DataFrame(combined_results)
    combined_df = combined_df.sort_values('avg_return_bps', ascending=False)
    
    print(f"{'Combined Filter':<35} {'Edge (bps)':<12} {'Win Rate':<10} {'Trades/Day':<12}")
    print("-" * 79)
    
    for _, row in combined_df.iterrows():
        print(f"{row['filter']:<35} {row['avg_return_bps']:>10.2f} {row['win_rate']:>9.1%} "
              f"{row['trades_per_day']:>11.1f}")

# Final summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

all_results = results_df.to_dict('records') + combined_results
all_results_df = pd.DataFrame(all_results)
all_results_df = all_results_df.sort_values('avg_return_bps', ascending=False)

# Check if any meet requirements
meets_requirements = all_results_df[(all_results_df['avg_return_bps'] >= 1.0) & 
                                   (all_results_df['trades_per_day'] >= 2.0)]

if len(meets_requirements) > 0:
    print("\n✓ FILTERS MEETING YOUR REQUIREMENTS (>=1 bps, 2+ tpd):")
    for _, row in meets_requirements.iterrows():
        print(f"  {row['filter']}: {row['avg_return_bps']:.2f} bps on {row['trades_per_day']:.1f} tpd, "
              f"{row['win_rate']:.1%} win rate")
else:
    print("\n✗ No filter + 0.3% stop combination achieves >=1 bps with 2+ trades/day")
    
    # Show best performers
    print("\nTop 5 by edge:")
    for i, (_, row) in enumerate(all_results_df.head(5).iterrows()):
        print(f"  {i+1}. {row['filter']}: {row['avg_return_bps']:.2f} bps on {row['trades_per_day']:.1f} tpd")
    
    # Best with good frequency
    good_freq = all_results_df[all_results_df['trades_per_day'] >= 2.0]
    if len(good_freq) > 0:
        print(f"\nBest with 2+ tpd:")
        for _, row in good_freq.head(3).iterrows():
            print(f"  {row['filter']}: {row['avg_return_bps']:.2f} bps on {row['trades_per_day']:.1f} tpd")