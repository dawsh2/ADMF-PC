#!/usr/bin/env python3
"""
Analyze ALL 1500 strategies with filters to get the complete picture.
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
    
    # Market trend
    df['uptrend'] = df['close'] > df['sma_50']
    
    print(f"Source data prepared: {len(df)} bars")
    return df

def analyze_strategy_with_filter(signals_df, source_df, filter_func):
    """Analyze a single strategy with a given filter."""
    
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
    
    # Track trades
    trades = []
    entry_price = None
    entry_signal = None
    
    for idx in range(len(enhanced)):
        row = enhanced.iloc[idx]
        signal = row['val']
        price = row['px']
        matched = filter_mask.iloc[idx] if idx < len(filter_mask) else False
        
        if signal != 0 and entry_price is None and matched:
            # New entry with filter match
            entry_price = price
            entry_signal = signal
                
        elif entry_price is not None and (signal == 0 or signal == -entry_signal):
            # Exit position
            log_return = np.log(price / entry_price) * entry_signal
            trades.append({
                'direction': 'long' if entry_signal > 0 else 'short',
                'log_return': log_return
            })
            
            # Check for reversal
            if signal != 0 and matched:
                entry_price = price
                entry_signal = signal
            else:
                entry_price = None
    
    if not trades:
        return None
    
    # Calculate metrics
    df_trades = pd.DataFrame(trades)
    
    # Overall metrics
    total_trades = len(df_trades)
    log_returns = df_trades['log_return'].values
    edge_bps = (np.mean(log_returns) * 10000) - 2  # 2bp costs
    
    # Direction breakdown
    long_trades = df_trades[df_trades['direction'] == 'long']
    short_trades = df_trades[df_trades['direction'] == 'short']
    
    # Time span
    first_ts = signals_df['timestamp'].min()
    last_ts = signals_df['timestamp'].max()
    trading_days = (last_ts - first_ts).days
    if trading_days == 0:
        trading_days = 1
    
    return {
        'total_trades': total_trades,
        'trades_per_day': total_trades / trading_days,
        'edge_bps': edge_bps,
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_edge_bps': (np.mean(long_trades['log_return']) * 10000 - 2) if len(long_trades) > 0 else 0,
        'short_edge_bps': (np.mean(short_trades['log_return']) * 10000 - 2) if len(short_trades) > 0 else 0,
        'trading_days': trading_days
    }

# Load source data once
source_df = load_source_data()

# Define key filters to test
filters = [
    ("No Filter", lambda df: df['val'] != 0),
    ("Shorts Only", lambda df: df['val'] < 0),
    ("Vol>70", lambda df: (df['vol_percentile'] > 70) & (df['val'] != 0)),
    ("Vol>60", lambda df: (df['vol_percentile'] > 60) & (df['val'] != 0)),
    ("Vol>50", lambda df: (df['vol_percentile'] > 50) & (df['val'] != 0)),
    ("Shorts + Vol>70", lambda df: (df['vol_percentile'] > 70) & (df['val'] < 0)),
    ("Shorts + Vol>60", lambda df: (df['vol_percentile'] > 60) & (df['val'] < 0)),
    ("Shorts + Vol>50", lambda df: (df['vol_percentile'] > 50) & (df['val'] < 0)),
    ("VWAP>0.1%", lambda df: (df['vwap_distance'] > 0.001) & (df['val'] != 0)),
    ("Volume>1.2x", lambda df: (df['volume_ratio'] > 1.2) & (df['val'] != 0)),
]

# Analyze ALL 1500 strategies
workspace = "workspaces/signal_generation_a2d31737"
signal_pattern = str(Path(workspace) / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones/*.parquet")
signal_files = sorted(glob(signal_pattern))

print(f"\nAnalyzing ALL {len(signal_files)} strategies...")
print("This will take a few minutes...\n")

# Store results for each filter
filter_results = {filter_name: [] for filter_name, _ in filters}

# Process all strategies
for i, signal_file in enumerate(signal_files):
    if i % 100 == 0:
        print(f"Processing strategy {i}/{len(signal_files)}...")
    
    try:
        signals_df = pd.read_parquet(signal_file)
        
        # Skip strategies with very few signals
        if len(signals_df) < 10:
            continue
        
        # Extract strategy ID
        strategy_name = Path(signal_file).stem
        strategy_id = int(strategy_name.split('_')[-1])
        
        # Test each filter
        for filter_name, filter_func in filters:
            result = analyze_strategy_with_filter(signals_df, source_df, filter_func)
            if result and result['total_trades'] >= 5:  # Min 5 trades for meaningful stats
                result['strategy_id'] = strategy_id
                filter_results[filter_name].append(result)
                
    except Exception as e:
        continue

print("\n\nANALYSIS COMPLETE!")
print("="*120)

# Create summary statistics for each filter
summary_data = []

for filter_name, results in filter_results.items():
    if not results:
        continue
    
    df = pd.DataFrame(results)
    
    # Calculate aggregate statistics
    total_strategies = len(df)
    avg_trades = df['total_trades'].mean()
    avg_tpd = df['trades_per_day'].mean()
    avg_edge = df['edge_bps'].mean()
    
    # Distribution of edges
    positive_edge = len(df[df['edge_bps'] > 0])
    edge_percentiles = df['edge_bps'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    
    # Best performers
    top_10_avg_edge = df.nlargest(10, 'edge_bps')['edge_bps'].mean()
    top_100_avg_edge = df.nlargest(100, 'edge_bps')['edge_bps'].mean() if len(df) >= 100 else df['edge_bps'].mean()
    
    # Long/Short breakdown
    avg_long_edge = df['long_edge_bps'].mean()
    avg_short_edge = df['short_edge_bps'].mean()
    
    # Frequency distribution
    high_freq = len(df[df['trades_per_day'] >= 1])
    very_high_freq = len(df[df['trades_per_day'] >= 2])
    
    summary_data.append({
        'Filter': filter_name,
        'Strategies': total_strategies,
        'Avg Trades': avg_trades,
        'Avg T/Day': avg_tpd,
        'Avg Edge': avg_edge,
        'Positive Edge %': positive_edge / total_strategies * 100,
        'Top 10 Edge': top_10_avg_edge,
        'Top 100 Edge': top_100_avg_edge,
        'Long Edge': avg_long_edge,
        'Short Edge': avg_short_edge,
        '>=1 T/Day': high_freq,
        '>=2 T/Day': very_high_freq,
        '10th %ile': edge_percentiles[0.1],
        '90th %ile': edge_percentiles[0.9]
    })

# Display summary
summary_df = pd.DataFrame(summary_data)

print("\nFILTER PERFORMANCE ACROSS ALL 1500 STRATEGIES")
print("="*120)
print("Filter           | Strats | Avg Trades | Avg T/Day | Avg Edge | Pos% | Top10 | Top100 | Long  | Short | >=1tpd | >=2tpd")
print("-----------------|--------|------------|-----------|----------|------|-------|--------|-------|-------|--------|-------")

for _, row in summary_df.iterrows():
    print(f"{row['Filter']:16s} | {row['Strategies']:6.0f} | {row['Avg Trades']:10.1f} | "
          f"{row['Avg T/Day']:9.2f} | {row['Avg Edge']:8.2f} | {row['Positive Edge %']:4.0f}% | "
          f"{row['Top 10 Edge']:5.2f} | {row['Top 100 Edge']:6.2f} | "
          f"{row['Long Edge']:5.0f} | {row['Short Edge']:5.0f} | "
          f"{row['>=1 T/Day']:6.0f} | {row['>=2 T/Day']:6.0f}")

# Calculate annual returns
print("\n\nANNUAL RETURN CALCULATIONS")
print("="*80)
print("Filter           | Avg Edge | Avg T/Day | Daily bps | Annual % | Top 100 Annual %")
print("-----------------|----------|-----------|-----------|----------|----------------")

for _, row in summary_df.iterrows():
    daily_bps = row['Avg Edge'] * row['Avg T/Day']
    annual_pct = daily_bps * 252 / 10000
    
    top100_daily = row['Top 100 Edge'] * row['Avg T/Day']
    top100_annual = top100_daily * 252 / 10000
    
    print(f"{row['Filter']:16s} | {row['Avg Edge']:8.2f} | {row['Avg T/Day']:9.2f} | "
          f"{daily_bps:9.2f} | {annual_pct:8.2f}% | {top100_annual:14.2f}%")

# Find strategies matching target frequencies
print("\n\nSTRATEGIES MATCHING TARGET FREQUENCIES")
print("="*80)

no_filter_results = pd.DataFrame(filter_results['No Filter'])
shorts_only_results = pd.DataFrame(filter_results['Shorts Only'])

# Find strategies with 2-3 trades per day
target_freq = no_filter_results[(no_filter_results['trades_per_day'] >= 2) & (no_filter_results['trades_per_day'] <= 3)]
print(f"\nStrategies with 2-3 trades/day (No Filter): {len(target_freq)}")
if len(target_freq) > 0:
    print(f"Average edge: {target_freq['edge_bps'].mean():.2f} bps")
    print(f"Best edge: {target_freq['edge_bps'].max():.2f} bps")
    print("Top 5 strategies:")
    for _, s in target_freq.nlargest(5, 'edge_bps').iterrows():
        print(f"  Strategy {s['strategy_id']}: {s['edge_bps']:.2f} bps @ {s['trades_per_day']:.2f} t/day")

# Shorts only at high frequency
shorts_high_freq = shorts_only_results[shorts_only_results['trades_per_day'] >= 1]
print(f"\n\nShorts Only with >=1 trade/day: {len(shorts_high_freq)}")
if len(shorts_high_freq) > 0:
    print(f"Average edge: {shorts_high_freq['edge_bps'].mean():.2f} bps")
    print(f"Best edge: {shorts_high_freq['edge_bps'].max():.2f} bps")
    print("Top 5 strategies:")
    for _, s in shorts_high_freq.nlargest(5, 'edge_bps').iterrows():
        print(f"  Strategy {s['strategy_id']}: {s['edge_bps']:.2f} bps @ {s['trades_per_day']:.2f} t/day")