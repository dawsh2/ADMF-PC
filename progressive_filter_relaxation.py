#!/usr/bin/env python3
"""
Progressively relax filters to find optimal trade-offs between edge and frequency.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings('ignore')

def load_source_data(data_file: str = "./data/SPY_5m.csv"):
    """Load and prepare source data with all necessary features."""
    
    print("Loading source data...")
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date
    
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
    
    # Volatility percentile
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
    
    return df

def test_filter_performance(signals_df, source_df, filter_func, filter_name):
    """Test a filter and return performance metrics."""
    
    signals_df = signals_df.copy()
    signals_df['timestamp'] = pd.to_datetime(signals_df['ts'])
    
    # Join with source features
    enhanced = pd.merge_asof(
        signals_df.sort_values('timestamp'),
        source_df.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    
    # Apply filter
    filter_mask = filter_func(enhanced)
    
    # Calculate performance
    trades = []
    entry_price = None
    entry_signal = None
    filtered_entries = 0
    
    for idx in range(len(enhanced)):
        row = enhanced.iloc[idx]
        signal = row['val']
        price = row['px']
        matched = filter_mask.iloc[idx] if idx < len(filter_mask) else False
        
        if signal != 0 and entry_price is None and matched:
            entry_price = price
            entry_signal = signal
            filtered_entries += 1
                
        elif entry_price is not None and (signal == 0 or signal == -entry_signal):
            log_return = np.log(price / entry_price) * entry_signal
            trades.append(log_return)
            
            if signal != 0 and matched:
                entry_price = price
                entry_signal = signal
                filtered_entries += 1
            else:
                entry_price = None
    
    if not trades:
        return None
    
    # Calculate metrics
    edge_bps = (np.mean(trades) * 10000) - 2  # 2bp costs
    
    # Time span
    first_ts = signals_df['timestamp'].min()
    last_ts = signals_df['timestamp'].max()
    trading_days = (last_ts - first_ts).days or 1
    
    return {
        'filter': filter_name,
        'filtered_entries': filtered_entries,
        'trades': len(trades),
        'edge_bps': edge_bps,
        'trades_per_day': len(trades) / trading_days,
        'win_rate': (np.array(trades) > 0).mean() * 100
    }

# Load source data
source_df = load_source_data()

# Progressive filter relaxations for each successful pattern
print("\nTesting progressive filter relaxations...\n")

# 1. Extended SMA + High Vol (Best performer: 5.84 bps @ 0.08 t/day)
print("="*80)
print("1. EXTENDED SMA + HIGH VOL PROGRESSIVE RELAXATION")
print("="*80)

sma_vol_filters = [
    # Original
    ("SMA>0.3% + Vol>85", lambda df: (df['vol_percentile'] > 85) & (df['sma20_distance'] > 0.003) & (df['val'] != 0)),
    # Relax SMA distance
    ("SMA>0.25% + Vol>85", lambda df: (df['vol_percentile'] > 85) & (df['sma20_distance'] > 0.0025) & (df['val'] != 0)),
    ("SMA>0.2% + Vol>85", lambda df: (df['vol_percentile'] > 85) & (df['sma20_distance'] > 0.002) & (df['val'] != 0)),
    # Relax volatility
    ("SMA>0.3% + Vol>80", lambda df: (df['vol_percentile'] > 80) & (df['sma20_distance'] > 0.003) & (df['val'] != 0)),
    ("SMA>0.3% + Vol>75", lambda df: (df['vol_percentile'] > 75) & (df['sma20_distance'] > 0.003) & (df['val'] != 0)),
    ("SMA>0.3% + Vol>70", lambda df: (df['vol_percentile'] > 70) & (df['sma20_distance'] > 0.003) & (df['val'] != 0)),
    # Relax both
    ("SMA>0.25% + Vol>80", lambda df: (df['vol_percentile'] > 80) & (df['sma20_distance'] > 0.0025) & (df['val'] != 0)),
    ("SMA>0.2% + Vol>75", lambda df: (df['vol_percentile'] > 75) & (df['sma20_distance'] > 0.002) & (df['val'] != 0)),
    ("SMA>0.15% + Vol>70", lambda df: (df['vol_percentile'] > 70) & (df['sma20_distance'] > 0.0015) & (df['val'] != 0)),
]

# 2. Shorts Only + High Vol (Second best: 4.48 bps @ 0.08 t/day)
print("\n2. SHORTS ONLY + HIGH VOL PROGRESSIVE RELAXATION")
print("="*80)

shorts_vol_filters = [
    # Original
    ("Shorts + Vol>70", lambda df: (df['vol_percentile'] > 70) & (df['val'] < 0)),
    # Relax volatility
    ("Shorts + Vol>65", lambda df: (df['vol_percentile'] > 65) & (df['val'] < 0)),
    ("Shorts + Vol>60", lambda df: (df['vol_percentile'] > 60) & (df['val'] < 0)),
    ("Shorts + Vol>55", lambda df: (df['vol_percentile'] > 55) & (df['val'] < 0)),
    ("Shorts + Vol>50", lambda df: (df['vol_percentile'] > 50) & (df['val'] < 0)),
    # Add time filters
    ("Shorts + Vol>60 + PM", lambda df: (df['vol_percentile'] > 60) & (df['val'] < 0) & (df['hour'] >= 14)),
    ("Shorts + Vol>50 + PM", lambda df: (df['vol_percentile'] > 50) & (df['val'] < 0) & (df['hour'] >= 14)),
]

# 3. Volume Spike (2.77 bps @ 0.09 t/day)
print("\n3. VOLUME SPIKE PROGRESSIVE RELAXATION")
print("="*80)

volume_filters = [
    # Original
    ("Volume>1.2x", lambda df: (df['volume_ratio'] > 1.2) & (df['val'] != 0)),
    # Relax volume
    ("Volume>1.15x", lambda df: (df['volume_ratio'] > 1.15) & (df['val'] != 0)),
    ("Volume>1.1x", lambda df: (df['volume_ratio'] > 1.1) & (df['val'] != 0)),
    ("Volume>1.05x", lambda df: (df['volume_ratio'] > 1.05) & (df['val'] != 0)),
    # Add volatility
    ("Volume>1.1x + Vol>50", lambda df: (df['volume_ratio'] > 1.1) & (df['vol_percentile'] > 50) & (df['val'] != 0)),
    ("Volume>1.05x + Vol>60", lambda df: (df['volume_ratio'] > 1.05) & (df['vol_percentile'] > 60) & (df['val'] != 0)),
]

# 4. Combined filters
print("\n4. COMBINED FILTERS (OR conditions)")
print("="*80)

combined_filters = [
    # Strict OR
    ("ExtSMA OR ShortVol OR VolSpike", 
     lambda df: ((df['vol_percentile'] > 85) & (df['sma20_distance'] > 0.003) | 
                 (df['vol_percentile'] > 70) & (df['val'] < 0) |
                 (df['volume_ratio'] > 1.2)) & (df['val'] != 0)),
    # Relaxed OR
    ("ExtSMA(relaxed) OR ShortVol(relaxed) OR Vol>1.1x", 
     lambda df: ((df['vol_percentile'] > 75) & (df['sma20_distance'] > 0.0025) | 
                 (df['vol_percentile'] > 60) & (df['val'] < 0) |
                 (df['volume_ratio'] > 1.1)) & (df['val'] != 0)),
    # Very relaxed
    ("Any: Vol>70 OR SMA>0.2% OR Vol>1.1x", 
     lambda df: ((df['vol_percentile'] > 70) | 
                 (df['sma20_distance'] > 0.002) |
                 (df['volume_ratio'] > 1.1)) & (df['val'] != 0)),
]

# Test all filters on sample strategies
workspace = "workspaces/signal_generation_a2d31737"
signal_pattern = str(Path(workspace) / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones/*.parquet")
signal_files = sorted(glob(signal_pattern))

# Use best performing strategies
strategy_ids = [1012, 1013, 1014, 1015]  # Top performers from earlier
test_results = []

for sid in strategy_ids:
    signal_file = f"{workspace}/traces/SPY_5m_1m/signals/swing_pivot_bounce_zones/SPY_5m_compiled_strategy_{sid}.parquet"
    if not Path(signal_file).exists():
        continue
        
    try:
        signals_df = pd.read_parquet(signal_file)
        
        # Test each filter set
        for filter_set in [sma_vol_filters, shorts_vol_filters, volume_filters, combined_filters]:
            for filter_name, filter_func in filter_set:
                result = test_filter_performance(signals_df, source_df, filter_func, filter_name)
                if result:
                    result['strategy_id'] = sid
                    test_results.append(result)
                    
    except Exception as e:
        print(f"Error with strategy {sid}: {e}")

# Analyze results
df_results = pd.DataFrame(test_results)

# Group by filter and average across strategies
filter_summary = df_results.groupby('filter').agg({
    'trades': 'mean',
    'trades_per_day': 'mean',
    'edge_bps': 'mean',
    'win_rate': 'mean',
    'filtered_entries': 'mean'
}).round(2)

print("\n\nFILTER RELAXATION SUMMARY")
print("="*80)
print("Filter                           | Trades | T/Day | Edge  | Win%  | Entries")
print("---------------------------------|--------|-------|-------|-------|--------")

for idx, row in filter_summary.iterrows():
    print(f"{idx:32s} | {row['trades']:6.1f} | {row['trades_per_day']:5.2f} | "
          f"{row['edge_bps']:5.2f} | {row['win_rate']:5.1f} | {row['filtered_entries']:7.1f}")

# Find sweet spots
print("\n\nSWEET SPOTS (Edge > 2 bps, Frequency > 0.5 t/day):")
print("="*80)

sweet_spots = filter_summary[(filter_summary['edge_bps'] > 2) & (filter_summary['trades_per_day'] > 0.5)]
if len(sweet_spots) > 0:
    for idx, row in sweet_spots.iterrows():
        annual_return = row['edge_bps'] * row['trades_per_day'] * 252 / 10000
        print(f"{idx}: {row['edge_bps']:.2f} bps @ {row['trades_per_day']:.2f} t/day = {annual_return:.1f}% annual")
else:
    # Find best trade-offs
    print("No filters meet both criteria. Best trade-offs:")
    
    # Sort by expected daily return (edge * frequency)
    filter_summary['daily_bps'] = filter_summary['edge_bps'] * filter_summary['trades_per_day']
    best_tradeoffs = filter_summary.nlargest(5, 'daily_bps')
    
    for idx, row in best_tradeoffs.iterrows():
        annual_return = row['daily_bps'] * 252 / 10000
        print(f"{idx}: {row['edge_bps']:.2f} bps @ {row['trades_per_day']:.2f} t/day = {annual_return:.1f}% annual")

print("\n\nRECOMMENDATIONS:")
print("1. Look for filters with positive expected daily return (edge * frequency)")
print("2. Consider risk-adjusted returns - higher frequency = smoother equity curve")
print("3. Test robustness across different market conditions")
print("4. Monitor for filter decay over time")