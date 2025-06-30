#!/usr/bin/env python3
"""
Apply the performance analysis filters to swing pivot bounce signals.

Filters to test:
1. Vol>70: Edge 2.18 bps, 2.8 trades/day
2. Vol>60: Edge 1.61 bps, 3.7 trades/day
3. Vol>50 + VWAP Distance >0.1%: Edge 1.70 bps, 2.6 trades/day
4. Volume >1.2x Average: Edge 1.40 bps, 3.5 trades/day
5. High Vol + Far VWAP (>0.2%): Edge 4.49 bps, 0.81 trades/day
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_source_data(data_file: str = "./data/SPY_5m.csv"):
    """Load and prepare source data with all necessary features."""
    
    print("Loading source data...")
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date
    
    print("Computing features...")
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # VWAP (daily reset)
    def calculate_vwap(group):
        group = group.copy()
        group['cum_vol'] = group['volume'].cumsum()
        group['cum_vol_price'] = (group['volume'] * group['close']).cumsum()
        group['vwap'] = group['cum_vol_price'] / group['cum_vol']
        return group
    
    df = df.groupby('date').apply(calculate_vwap).reset_index(drop=True)
    
    # ATR for volatility
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    df['tr'] = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr_20'] = df['tr'].rolling(20).mean()
    
    # Volatility percentile (rolling 100 bars)
    df['vol_percentile'] = df['atr_20'].rolling(100).rank(pct=True) * 100
    
    # Volume ratio
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Price extensions
    df['vwap_distance'] = np.abs((df['close'] - df['vwap']) / df['vwap'])
    df['sma20_distance'] = np.abs((df['close'] - df['sma_20']) / df['sma_20'])
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    
    # Market trend
    df['uptrend'] = df['close'] > df['sma_50']
    
    print(f"Features computed for {len(df)} bars")
    return df

def apply_filter_to_signals(signals_df, source_df, filter_name, filter_func):
    """Apply a filter to signals and calculate performance."""
    
    # Convert timestamps
    signals_df = signals_df.copy()
    signals_df['timestamp'] = pd.to_datetime(signals_df['ts'])
    
    # Join signals with source features
    enhanced = pd.merge_asof(
        signals_df.sort_values('timestamp'),
        source_df.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    
    # Apply filter
    filter_mask = filter_func(enhanced)
    
    # Calculate performance only for trades that start with filtered signals
    trades = []
    entry_price = None
    entry_signal = None
    entry_matched_filter = False
    filtered_entries = 0
    
    for idx in range(len(enhanced)):
        row = enhanced.iloc[idx]
        signal = row['val']
        price = row['px']
        matched = filter_mask.iloc[idx] if idx < len(filter_mask) else False
        
        if signal != 0 and entry_price is None:
            # Only enter if filter matches
            if matched:
                entry_price = price
                entry_signal = signal
                entry_matched_filter = True
                filtered_entries += 1
                
        elif entry_price is not None and (signal == 0 or signal == -entry_signal):
            # Exit current position
            log_return = np.log(price / entry_price) * entry_signal
            trades.append({
                'log_return': log_return,
                'direction': 'long' if entry_signal > 0 else 'short',
                'filtered': entry_matched_filter
            })
            
            # Check for reversal entry
            if signal != 0 and matched:
                entry_price = price
                entry_signal = signal
                entry_matched_filter = True
                filtered_entries += 1
            else:
                entry_price = None
    
    if not trades:
        return None
    
    df_trades = pd.DataFrame(trades)
    log_returns = df_trades['log_return'].values
    edge_bps = np.mean(log_returns) * 10000 - 2  # 2bp costs
    
    # Time span for frequency
    first_ts = signals_df['timestamp'].min()
    last_ts = signals_df['timestamp'].max()
    trading_days = (last_ts - first_ts).days or 1
    
    return {
        'filter': filter_name,
        'total_signals': len(signals_df),
        'filtered_entries': filtered_entries,
        'filter_rate': filtered_entries / len(signals_df[signals_df['val'] != 0]) * 100,
        'trades': len(df_trades),
        'edge_bps': edge_bps,
        'trades_per_day': len(df_trades) / trading_days,
        'win_rate': (df_trades['log_return'] > 0).mean() * 100,
        'long_trades': len(df_trades[df_trades['direction'] == 'long']),
        'short_trades': len(df_trades[df_trades['direction'] == 'short'])
    }

# Define filters
def vol_70_filter(df):
    return (df['vol_percentile'] > 70) & (df['val'] != 0)

def vol_60_filter(df):
    return (df['vol_percentile'] > 60) & (df['val'] != 0)

def vol_50_vwap_filter(df):
    return (df['vol_percentile'] > 50) & (df['vwap_distance'] > 0.001) & (df['val'] != 0)

def volume_spike_filter(df):
    return (df['volume_ratio'] > 1.2) & (df['val'] != 0)

def high_vol_far_vwap_filter(df):
    return (df['vol_percentile'] > 85) & (df['vwap_distance'] > 0.002) & (df['val'] != 0)

def extended_sma_high_vol_filter(df):
    return (df['vol_percentile'] > 85) & (df['sma20_distance'] > 0.003) & (df['val'] != 0)

def best_hour_filter(df):
    return (df['hour'] >= 20) & (df['hour'] <= 21) & (df['val'] != 0)

def afternoon_filter(df):
    return (df['hour'] >= 14) & (df['val'] != 0)

def shorts_only_high_vol_filter(df):
    return (df['vol_percentile'] > 70) & (df['val'] < 0)

# Load source data once
source_df = load_source_data()

# Analyze strategies
workspace = "workspaces/signal_generation_a2d31737"
signal_pattern = str(Path(workspace) / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones/*.parquet")
signal_files = sorted(glob(signal_pattern))

# Define filters to test
filters = [
    ("Baseline (No Filter)", lambda df: df['val'] != 0),
    ("Vol>70", vol_70_filter),
    ("Vol>60", vol_60_filter),
    ("Vol>50 + VWAP>0.1%", vol_50_vwap_filter),
    ("Volume>1.2x", volume_spike_filter),
    ("High Vol + Far VWAP", high_vol_far_vwap_filter),
    ("Extended SMA + High Vol", extended_sma_high_vol_filter),
    ("Best Hour (20-21)", best_hour_filter),
    ("Afternoon (14+)", afternoon_filter),
    ("Shorts Only + Vol>70", shorts_only_high_vol_filter)
]

# Test on a subset of strategies
print(f"\nAnalyzing {len(signal_files)} strategies with filters...")
filter_results = {filter_name: [] for filter_name, _ in filters}

# Sample strategies - test on best performers and random selection
strategy_ids = [1012, 1013, 1014, 88, 48]  # Include mentioned IDs
strategy_ids.extend(np.random.choice(range(1500), 45, replace=False))  # Random sample

analyzed = 0
for i, signal_file in enumerate(signal_files):
    # Extract strategy ID
    strategy_name = Path(signal_file).stem
    try:
        sid = int(strategy_name.split('_')[-1])
        if sid not in strategy_ids:
            continue
    except:
        continue
    
    print(f"Analyzing strategy {sid}...", end='\r')
    
    try:
        signals_df = pd.read_parquet(signal_file)
        if len(signals_df) < 10:
            continue
        
        for filter_name, filter_func in filters:
            result = apply_filter_to_signals(signals_df, source_df, filter_name, filter_func)
            if result:
                result['strategy_id'] = sid
                filter_results[filter_name].append(result)
        
        analyzed += 1
        
    except Exception as e:
        print(f"\nError on strategy {sid}: {e}")
        continue

print(f"\n\nAnalyzed {analyzed} strategies")

# Summarize results
print("\n" + "="*100)
print("FILTER PERFORMANCE COMPARISON")
print("="*100)
print("\nFilter                    | Strategies | Avg Trades | T/Day | Edge  | Win%  | Filter% | Target")
print("--------------------------|------------|------------|-------|-------|-------|---------|-------")

for filter_name, _ in filters:
    results = filter_results[filter_name]
    if results:
        df = pd.DataFrame(results)
        avg_trades = df['trades'].mean()
        avg_tpd = df['trades_per_day'].mean()
        avg_edge = df['edge_bps'].mean()
        avg_win = df['win_rate'].mean()
        avg_filter = df['filter_rate'].mean()
        
        # Target values
        target = ""
        if "Vol>70" in filter_name and "Shorts" not in filter_name:
            target = "2.18 bps @ 2.8"
        elif "Vol>60" in filter_name:
            target = "1.61 bps @ 3.7"
        elif "VWAP>0.1%" in filter_name:
            target = "1.70 bps @ 2.6"
        elif "Volume>1.2x" in filter_name:
            target = "1.40 bps @ 3.5"
        elif "Far VWAP" in filter_name:
            target = "4.49 bps @ 0.81"
        elif "Extended SMA" in filter_name:
            target = "3.36 bps @ 0.32"
        
        print(f"{filter_name:25s} | {len(results):10d} | {avg_trades:10.1f} | {avg_tpd:5.2f} | "
              f"{avg_edge:5.2f} | {avg_win:5.1f} | {avg_filter:7.1f} | {target}")

# Find best filter combinations
print("\n\nBEST PERFORMING FILTER COMBINATIONS:")
all_results = []
for filter_name in filter_results:
    for r in filter_results[filter_name]:
        r['filter_name'] = filter_name
        all_results.append(r)

if all_results:
    df_all = pd.DataFrame(all_results)
    
    # Best by edge
    print("\nTop 5 by Edge:")
    for _, row in df_all.nlargest(5, 'edge_bps').iterrows():
        print(f"  Strategy {row['strategy_id']} + {row['filter_name']}: "
              f"{row['edge_bps']:.2f} bps @ {row['trades_per_day']:.2f} t/day")
    
    # Best by frequency
    print("\nTop 5 by Frequency:")
    for _, row in df_all.nlargest(5, 'trades_per_day').iterrows():
        print(f"  Strategy {row['strategy_id']} + {row['filter_name']}: "
              f"{row['edge_bps']:.2f} bps @ {row['trades_per_day']:.2f} t/day")

print("\n\nCONCLUSIONS:")
print("Compare actual results to targets from the config file comments.")