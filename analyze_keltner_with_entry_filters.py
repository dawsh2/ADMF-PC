"""Analyze Keltner Bands with entry filters applied post-hoc"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_d5807cc2")
signal_file = workspace / "traces/SPY_1m/signals/keltner_bands/SPY_compiled_strategy_0.parquet"

print("=== KELTNER BANDS WITH ENTRY FILTERS (POST-HOC) ===\n")

# Load signals
signals = pd.read_parquet(signal_file)

# Load SPY 1m data and calculate indicators
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})
spy_subset = spy_1m.iloc[:81787].copy()

# Calculate all indicators we need for filtering
spy_subset['returns'] = spy_subset['close'].pct_change()

# RSI
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

spy_subset['rsi'] = calculate_rsi(spy_subset['close'])

# Volatility percentile
spy_subset['volatility_20'] = spy_subset['returns'].rolling(20).std() * np.sqrt(390) * 100
spy_subset['vol_percentile'] = spy_subset['volatility_20'].rolling(window=390*5).rank(pct=True) * 100

# Volume ratio
spy_subset['volume_sma_20'] = spy_subset['volume'].rolling(20).mean()
spy_subset['volume_ratio'] = spy_subset['volume'] / spy_subset['volume_sma_20']

# VWAP
spy_subset['date'] = spy_subset['timestamp'].dt.date
spy_subset['typical_price'] = (spy_subset['high'] + spy_subset['low'] + spy_subset['close']) / 3
spy_subset['pv'] = spy_subset['typical_price'] * spy_subset['volume']
spy_subset['cum_pv'] = spy_subset.groupby('date')['pv'].cumsum()
spy_subset['cum_volume'] = spy_subset.groupby('date')['volume'].cumsum()
spy_subset['vwap'] = spy_subset['cum_pv'] / spy_subset['cum_volume']

# Test different filter combinations
filter_configs = [
    ("No filter (baseline)", lambda row, signal: True),
    ("RSI < 50", lambda row, signal: row['rsi'] < 50),
    ("High Vol (>80%)", lambda row, signal: row['vol_percentile'] > 80),
    ("High Volume (>1.5x)", lambda row, signal: row['volume_ratio'] > 1.5),
    ("RSI<50 + HighVol>80", lambda row, signal: (row['rsi'] < 50) and (row['vol_percentile'] > 80)),
    ("RSI<50 + HighVol>90", lambda row, signal: (row['rsi'] < 50) and (row['vol_percentile'] > 90)),
    ("RSI<50 + HighVol>80 + HighVolume>1.5x", 
     lambda row, signal: (row['rsi'] < 50) and (row['vol_percentile'] > 80) and (row['volume_ratio'] > 1.5)),
    ("Best: HighVol>90 + HighVolume>2x + RSI<50",
     lambda row, signal: (row['vol_percentile'] > 90) and (row['volume_ratio'] > 2.0) and (row['rsi'] < 50)),
    ("Directional: Longs below VWAP, Shorts above",
     lambda row, signal: ((signal > 0) and (row['close'] < row['vwap'])) or ((signal < 0) and (row['close'] > row['vwap'])))
]

def analyze_with_filter(filter_func, filter_name, stop_pct=0.003):
    """Analyze performance with entry filter and 0.3% stop"""
    trades = []
    entry_data = None
    filtered_entries = 0
    total_potential_entries = 0
    
    for i in range(len(signals)):
        curr = signals.iloc[i]
        
        if entry_data is None and curr['val'] != 0:
            total_potential_entries += 1
            entry_idx = curr['idx']
            
            # Apply entry filter
            if entry_idx < len(spy_subset):
                row = spy_subset.iloc[entry_idx]
                if filter_func(row, curr['val']):
                    filtered_entries += 1
                    entry_data = {
                        'idx': entry_idx,
                        'price': curr['px'],
                        'signal': curr['val']
                    }
        
        elif entry_data is not None:
            # Check for stop loss
            stopped_out = False
            exit_price = curr['px']
            exit_idx = curr['idx']
            
            for check_idx in range(entry_data['idx'] + 1, min(curr['idx'] + 1, len(spy_subset))):
                check_bar = spy_subset.iloc[check_idx]
                
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
            
            # Natural exit or stop
            if stopped_out or curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal']):
                pct_return = (exit_price / entry_data['price'] - 1) * entry_data['signal'] * 100
                duration = exit_idx - entry_data['idx']
                is_quick = duration < 5
                
                trades.append({
                    'pct_return': pct_return,
                    'direction': 'short' if entry_data['signal'] < 0 else 'long',
                    'duration': duration,
                    'stopped_out': stopped_out,
                    'is_quick': is_quick
                })
                
                # Check next signal with filter
                if curr['val'] != 0 and not stopped_out and curr['idx'] < len(spy_subset):
                    row = spy_subset.iloc[curr['idx']]
                    if filter_func(row, curr['val']):
                        entry_data = {
                            'idx': curr['idx'],
                            'price': curr['px'],
                            'signal': curr['val']
                        }
                    else:
                        entry_data = None
                else:
                    entry_data = None
    
    return pd.DataFrame(trades), filtered_entries, total_potential_entries

# Run analysis for each filter
total_days = 81787 / 390
results = []

print(f"{'Filter':<45} {'Trades':<8} {'Filtered%':<10} {'Avg(bps)':<10} {'TPD':<8} {'WinRate':<8} {'QuickTPD':<10}")
print("-" * 110)

for filter_name, filter_func in filter_configs:
    trades_df, filtered_entries, total_entries = analyze_with_filter(filter_func, filter_name)
    
    if len(trades_df) > 0:
        avg_return = trades_df['pct_return'].mean()
        win_rate = (trades_df['pct_return'] > 0).mean()
        tpd = len(trades_df) / total_days
        
        # Quick exits
        quick_trades = trades_df[trades_df['is_quick']]
        quick_tpd = len(quick_trades) / total_days
        quick_return = quick_trades['pct_return'].mean() if len(quick_trades) > 0 else 0
        
        filter_pct = (filtered_entries / total_entries * 100) if total_entries > 0 else 0
        
        print(f"{filter_name:<45} {len(trades_df):<8} {filter_pct:<10.1f} "
              f"{avg_return * 100:<10.2f} {tpd:<8.1f} {win_rate:<8.1%} {quick_tpd:<10.1f}")
        
        results.append({
            'filter': filter_name,
            'trades': len(trades_df),
            'avg_bps': avg_return * 100,
            'tpd': tpd,
            'win_rate': win_rate,
            'quick_trades': len(quick_trades),
            'quick_bps': quick_return * 100,
            'quick_tpd': quick_tpd
        })

# Summary of best performers
print("\n=== BEST PERFORMERS (>1 bps with 2+ TPD) ===")
results_df = pd.DataFrame(results)
good_results = results_df[(results_df['avg_bps'] >= 1.0) & (results_df['tpd'] >= 2.0)]

if len(good_results) > 0:
    for _, row in good_results.iterrows():
        annual_return = (row['avg_bps'] / 100 - 0.01) * row['tpd'] * 252  # After 1bp cost
        print(f"\n{row['filter']}:")
        print(f"  Edge: {row['avg_bps']:.2f} bps on {row['tpd']:.1f} trades/day")
        print(f"  Annual return after costs: {annual_return:.1%}")
else:
    print("\nNo filters achieve >=1 bps with 2+ trades/day")
    
    # Show best edge
    best_edge = results_df.nlargest(1, 'avg_bps').iloc[0]
    print(f"\nBest edge: {best_edge['filter']} with {best_edge['avg_bps']:.2f} bps on {best_edge['tpd']:.1f} tpd")
    
    # Show best frequency
    high_freq = results_df[results_df['tpd'] >= 2.0]
    if len(high_freq) > 0:
        best_freq = high_freq.nlargest(1, 'avg_bps').iloc[0]
        print(f"Best with 2+ tpd: {best_freq['filter']} with {best_freq['avg_bps']:.2f} bps")