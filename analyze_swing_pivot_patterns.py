#!/usr/bin/env python3
"""
Analyze swing pivot bounce traces for high-value patterns.

Looking for patterns like:
- High Vol + Far from VWAP (>0.2%): 4.49 bps edge
- Extended from SMA20 + High Vol: 3.36 bps edge
- Best Hour: 20:00-21:00
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_enhance_signals(signal_file: str, data_file: str = "./data/SPY_5m_1m.csv"):
    """Load signals and join with source data to compute context features."""
    
    # Load sparse signals
    signals_df = pd.read_parquet(signal_file)
    if signals_df.empty or len(signals_df) < 10:
        return None
    
    # Load source data
    source_df = pd.read_csv(data_file)
    source_df['timestamp'] = pd.to_datetime(source_df['timestamp'])
    source_df['hour'] = source_df['timestamp'].dt.hour
    source_df['date'] = source_df['timestamp'].dt.date
    
    # Compute context features
    # 1. Moving averages
    source_df['sma_20'] = source_df['close'].rolling(20).mean()
    source_df['sma_50'] = source_df['close'].rolling(50).mean()
    
    # 2. VWAP (daily reset)
    def calculate_vwap(group):
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
    
    # Convert signal timestamps and join
    signals_df['timestamp'] = pd.to_datetime(signals_df['ts'])
    
    # Join signals with context (using merge_asof for nearest timestamp)
    enhanced_signals = pd.merge_asof(
        signals_df.sort_values('timestamp'),
        source_df[['timestamp', 'hour', 'atr_percentile', 'volume_ratio', 
                   'vwap_distance', 'sma20_distance', 'range_pct', 'uptrend']].sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    
    return enhanced_signals

def analyze_pattern_performance(enhanced_signals: pd.DataFrame, pattern_name: str, pattern_filter):
    """Analyze performance of signals matching a specific pattern."""
    
    # Apply pattern filter
    pattern_signals = enhanced_signals[pattern_filter(enhanced_signals)].copy()
    
    if len(pattern_signals) < 2:
        return None
    
    # Calculate returns
    trades = []
    entry_price = None
    entry_signal = None
    
    for _, row in pattern_signals.iterrows():
        signal = row['val']
        price = row['px']
        
        if signal != 0 and entry_price is None:
            entry_price = price
            entry_signal = signal
        elif entry_price is not None and (signal == 0 or signal == -entry_signal):
            log_return = np.log(price / entry_price) * entry_signal
            trades.append(log_return)
            
            if signal != 0:  # Reversal
                entry_price = price
                entry_signal = signal
            else:  # Exit
                entry_price = None
    
    if not trades:
        return None
    
    # Calculate metrics
    trades_bps = [t * 10000 for t in trades]
    edge_bps = np.mean(trades_bps) - 2  # 2bp costs
    
    return {
        'pattern': pattern_name,
        'signals': len(pattern_signals),
        'trades': len(trades),
        'edge_bps': edge_bps,
        'total_return_bps': sum(trades_bps) - 2 * len(trades)
    }

# Define pattern filters based on the high-value patterns
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

# Analyze workspaces
workspaces = [
    "workspaces/signal_generation_ae5ce1b4",  # Previous (low frequency)
    "workspaces/signal_generation_a2d31737"   # Current (higher frequency)
]

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

for workspace in workspaces:
    print(f"\n{'='*60}")
    print(f"ANALYZING: {workspace}")
    print(f"{'='*60}")
    
    signal_pattern = str(Path(workspace) / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones/*.parquet")
    signal_files = sorted(glob(signal_pattern))[:100]  # Sample first 100
    
    pattern_results = {pattern[0]: [] for pattern in patterns}
    
    for signal_file in signal_files:
        try:
            enhanced_signals = load_and_enhance_signals(signal_file)
            if enhanced_signals is None:
                continue
            
            for pattern_name, pattern_filter in patterns:
                result = analyze_pattern_performance(enhanced_signals, pattern_name, pattern_filter)
                if result:
                    pattern_results[pattern_name].append(result)
        except Exception as e:
            continue
    
    # Summarize results
    print("\nPATTERN ANALYSIS RESULTS:")
    print("Pattern                    | Strategies | Avg Trades | Avg Edge | Best Edge")
    print("---------------------------|------------|------------|----------|----------")
    
    for pattern_name in pattern_results:
        results = pattern_results[pattern_name]
        if results:
            avg_trades = np.mean([r['trades'] for r in results])
            avg_edge = np.mean([r['edge_bps'] for r in results])
            best_edge = max([r['edge_bps'] for r in results])
            
            print(f"{pattern_name:26s} | {len(results):10d} | {avg_trades:10.1f} | "
                  f"{avg_edge:8.2f} | {best_edge:8.2f}")
    
    # Find best performing strategies overall
    all_results = []
    for results in pattern_results.values():
        all_results.extend(results)
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        best_strategies = df_results.nlargest(10, 'edge_bps')
        
        print("\n\nTOP 10 STRATEGY-PATTERN COMBINATIONS:")
        print("Pattern                    | Trades | Edge (bps)")
        print("---------------------------|--------|----------")
        for _, row in best_strategies.iterrows():
            print(f"{row['pattern']:26s} | {row['trades']:6d} | {row['edge_bps']:8.2f}")

print("\n\nKEY INSIGHTS TO REPLICATE:")
print("1. High volatility (>70th percentile) is crucial for edge")
print("2. Distance from VWAP >0.2% provides best opportunities")
print("3. Evening hours (20:00-21:00) show enhanced performance")
print("4. Counter-trend shorts in uptrends with high vol perform best")
print("5. Volume spikes and high range bars indicate better setups")