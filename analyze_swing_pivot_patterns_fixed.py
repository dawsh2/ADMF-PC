#!/usr/bin/env python3
"""
Analyze swing pivot bounce traces for high-value patterns - Fixed version.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_enhance_signals(signal_file: str, data_file: str = "./data/SPY_5m.csv"):
    """Load signals and join with source data to compute context features."""
    
    # Load sparse signals
    signals_df = pd.read_parquet(signal_file)
    if signals_df.empty or len(signals_df) < 10:
        return None
    
    # Check which data file to use based on timeframe
    tf = signals_df['tf'].iloc[0] if 'tf' in signals_df.columns else '5m'
    if tf == '1m' and Path("./data/SPY_1m.csv").exists():
        data_file = "./data/SPY_1m.csv"
    
    # Load source data
    source_df = pd.read_csv(data_file)
    source_df['timestamp'] = pd.to_datetime(source_df['timestamp'])
    
    # Convert signal timestamps
    signals_df['timestamp'] = pd.to_datetime(signals_df['ts'])
    
    # Get date range from signals
    start_date = signals_df['timestamp'].min()
    end_date = signals_df['timestamp'].max()
    
    # Filter source data to match signal date range
    source_df = source_df[(source_df['timestamp'] >= start_date) & 
                          (source_df['timestamp'] <= end_date)]
    
    if len(source_df) == 0:
        return None
    
    # Compute context features
    source_df['hour'] = source_df['timestamp'].dt.hour
    source_df['date'] = source_df['timestamp'].dt.date
    
    # 1. Moving averages
    source_df['sma_20'] = source_df['close'].rolling(20).mean()
    source_df['sma_50'] = source_df['close'].rolling(50).mean()
    
    # 2. VWAP (daily reset)
    def calculate_vwap(group):
        group = group.copy()
        group['cum_vol'] = group['volume'].cumsum()
        group['cum_vol_price'] = (group['volume'] * group['close']).cumsum()
        group['vwap'] = group['cum_vol_price'] / group['cum_vol']
        return group
    
    source_df = source_df.groupby('date').apply(calculate_vwap).reset_index(drop=True)
    
    # 3. ATR for volatility
    high_low = source_df['high'] - source_df['low']
    high_close = np.abs(source_df['high'] - source_df['close'].shift())
    low_close = np.abs(source_df['low'] - source_df['close'].shift())
    source_df['tr'] = np.maximum(high_low, np.maximum(high_close, low_close))
    source_df['atr_20'] = source_df['tr'].rolling(20).mean()
    
    # 4. Volatility percentile (rolling 100 bars)
    source_df['atr_percentile'] = source_df['atr_20'].rolling(100).rank(pct=True) * 100
    
    # 5. Volume ratio
    source_df['volume_sma_20'] = source_df['volume'].rolling(20).mean()
    source_df['volume_ratio'] = source_df['volume'] / source_df['volume_sma_20']
    
    # 6. Price extensions
    source_df['vwap_distance'] = (source_df['close'] - source_df['vwap']) / source_df['vwap']
    source_df['sma20_distance'] = (source_df['close'] - source_df['sma_20']) / source_df['sma_20']
    source_df['range_pct'] = (source_df['high'] - source_df['low']) / source_df['close']
    
    # 7. Market trend
    source_df['uptrend'] = source_df['close'] > source_df['sma_50']
    
    # Join signals with context
    enhanced_signals = pd.merge_asof(
        signals_df.sort_values('timestamp'),
        source_df[['timestamp', 'hour', 'atr_percentile', 'volume_ratio', 
                   'vwap_distance', 'sma20_distance', 'range_pct', 'uptrend']].sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    
    # Drop rows with NaN values in key columns
    enhanced_signals = enhanced_signals.dropna(subset=['atr_percentile', 'vwap_distance'])
    
    return enhanced_signals

def analyze_pattern_performance(enhanced_signals: pd.DataFrame, pattern_name: str, pattern_filter):
    """Analyze performance of signals matching a specific pattern."""
    
    # Apply pattern filter
    pattern_mask = pattern_filter(enhanced_signals)
    
    # Count how many signals match
    matching_count = pattern_mask.sum()
    
    if matching_count == 0:
        return None
    
    # For sparse data, we need to analyze all signals, not just filtered ones
    # But we'll track which trades started with pattern-matching signals
    trades = []
    entry_price = None
    entry_signal = None
    entry_matched_pattern = False
    pattern_entries = 0
    
    for idx, row in enhanced_signals.iterrows():
        signal = row['val']
        price = row['px']
        matched = pattern_mask.loc[idx] if idx in pattern_mask.index else False
        
        if signal != 0 and entry_price is None:
            # New entry
            entry_price = price
            entry_signal = signal
            entry_matched_pattern = matched
            if matched:
                pattern_entries += 1
                
        elif entry_price is not None and (signal == 0 or signal == -entry_signal):
            # Exit or reversal
            if entry_matched_pattern:  # Only count trades that started with pattern
                log_return = np.log(price / entry_price) * entry_signal
                trades.append(log_return)
            
            if signal != 0:  # Reversal
                entry_price = price
                entry_signal = signal
                entry_matched_pattern = matched
                if matched:
                    pattern_entries += 1
            else:  # Exit
                entry_price = None
    
    if not trades:
        return None
    
    # Calculate metrics
    trades_bps = [t * 10000 for t in trades]
    edge_bps = np.mean(trades_bps) - 2  # 2bp costs
    
    return {
        'pattern': pattern_name,
        'signals_matching': matching_count,
        'pattern_entries': pattern_entries,
        'trades': len(trades),
        'edge_bps': edge_bps,
        'total_return_bps': sum(trades_bps) - 2 * len(trades)
    }

# Define pattern filters
def high_vol_far_vwap(df):
    """High Vol + Far from VWAP (>0.2%)"""
    return (df['atr_percentile'] > 85) & (np.abs(df['vwap_distance']) > 0.002)

def extended_sma_high_vol(df):
    """Extended from SMA20 + High Vol"""
    return (df['atr_percentile'] > 85) & (np.abs(df['sma20_distance']) > 0.003)

def best_hour_pattern(df):
    """Best Hour: 20:00-21:00"""
    return (df['hour'] >= 20) & (df['hour'] <= 21)

def vol_70_pattern(df):
    """Vol>70 filter (best overall)"""
    return df['atr_percentile'] > 70

def vol_60_pattern(df):
    """Vol>60 filter (higher frequency)"""
    return df['atr_percentile'] > 60

def high_vol_vwap_combo(df):
    """High Vol + VWAP distance >0.1%"""
    return (df['atr_percentile'] > 50) & (np.abs(df['vwap_distance']) > 0.001)

def volume_spike_pattern(df):
    """Volume >1.2x average"""
    return df['volume_ratio'] > 1.2

def shorts_in_uptrend(df):
    """Counter-trend shorts in uptrends"""
    return (df['val'] < 0) & (df['uptrend'] == True) & (df['atr_percentile'] > 70)

def high_range_bars(df):
    """High range bars (>0.1%)"""
    return df['range_pct'] > 0.001

# Analyze workspace
workspace = "workspaces/signal_generation_a2d31737"
print(f"ANALYZING: {workspace}")
print(f"{'='*60}\n")

signal_pattern = str(Path(workspace) / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones/*.parquet")
signal_files = sorted(glob(signal_pattern))[:50]  # Sample first 50 for debugging

patterns = [
    ("High Vol + Far VWAP", high_vol_far_vwap),
    ("Extended SMA + High Vol", extended_sma_high_vol),
    ("Best Hour (20-21)", best_hour_pattern),
    ("Vol > 70", vol_70_pattern),
    ("Vol > 60", vol_60_pattern),
    ("Vol + VWAP Combo", high_vol_vwap_combo),
    ("Volume Spike", volume_spike_pattern),
    ("Shorts in Uptrend", shorts_in_uptrend),
    ("High Range Bars", high_range_bars)
]

pattern_results = {pattern[0]: [] for pattern in patterns}
successful_loads = 0

for i, signal_file in enumerate(signal_files):
    if i % 10 == 0:
        print(f"Processing file {i+1}/{len(signal_files)}...", end='\r')
    
    try:
        enhanced_signals = load_and_enhance_signals(signal_file)
        if enhanced_signals is None or len(enhanced_signals) < 10:
            continue
        
        successful_loads += 1
        
        for pattern_name, pattern_filter in patterns:
            result = analyze_pattern_performance(enhanced_signals, pattern_name, pattern_filter)
            if result:
                pattern_results[pattern_name].append(result)
    except Exception as e:
        continue

print(f"\n\nSuccessfully analyzed {successful_loads} strategies")

# Summarize results
print("\nPATTERN ANALYSIS RESULTS:")
print("Pattern                    | Strategies | Avg Signals | Avg Trades | Avg Edge | Best Edge")
print("---------------------------|------------|-------------|------------|----------|----------")

for pattern_name in pattern_results:
    results = pattern_results[pattern_name]
    if results:
        avg_signals = np.mean([r['signals_matching'] for r in results])
        avg_trades = np.mean([r['trades'] for r in results])
        avg_edge = np.mean([r['edge_bps'] for r in results])
        best_edge = max([r['edge_bps'] for r in results])
        
        print(f"{pattern_name:26s} | {len(results):10d} | {avg_signals:11.1f} | {avg_trades:10.1f} | "
              f"{avg_edge:8.2f} | {best_edge:8.2f}")
    else:
        print(f"{pattern_name:26s} | {0:10d} |           - |          - |        - |        -")

# Find best performing patterns
all_results = []
for pattern_name, results in pattern_results.items():
    for r in results:
        r['pattern_name'] = pattern_name
        all_results.append(r)

if all_results:
    df_results = pd.DataFrame(all_results)
    best_by_edge = df_results.nlargest(10, 'edge_bps')
    
    print("\n\nTOP PATTERNS BY EDGE:")
    print("Pattern                    | Signals | Trades | Edge (bps)")
    print("---------------------------|---------|--------|----------")
    for _, row in best_by_edge.iterrows():
        print(f"{row['pattern_name']:26s} | {row['signals_matching']:7d} | {row['trades']:6d} | {row['edge_bps']:8.2f}")
    
    # Also show most frequent patterns
    pattern_freq = df_results.groupby('pattern_name').agg({
        'signals_matching': 'sum',
        'trades': 'sum',
        'edge_bps': 'mean'
    }).sort_values('signals_matching', ascending=False)
    
    print("\n\nMOST FREQUENT PATTERNS:")
    print("Pattern                    | Total Signals | Total Trades | Avg Edge")
    print("---------------------------|---------------|--------------|----------")
    for pattern_name, row in pattern_freq.head(5).iterrows():
        print(f"{pattern_name:26s} | {row['signals_matching']:13.0f} | {row['trades']:12.0f} | {row['edge_bps']:8.2f}")

print("\n\nCONCLUSIONS:")
print("Compare these results to the target patterns:")
print("- High Vol + Far VWAP: Target 4.49 bps edge")
print("- Extended SMA + High Vol: Target 3.36 bps edge")
print("- Vol>70: Target 2.18 bps edge")
print("- Vol>60: Target 1.61 bps edge")