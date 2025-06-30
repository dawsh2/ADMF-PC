#!/usr/bin/env python3
"""
Detailed filter analysis with careful return calculations and long/short breakdown.
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
    
    # Market trend
    df['uptrend'] = df['close'] > df['sma_50']
    
    return df

def analyze_filter_detailed(signals_df, source_df, filter_func, filter_name):
    """Detailed analysis with proper return calculations and long/short breakdown."""
    
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
    
    # Track all trades with detailed info
    trades = []
    entry_price = None
    entry_signal = None
    entry_idx = None
    entry_matched_filter = False
    
    for idx in range(len(enhanced)):
        row = enhanced.iloc[idx]
        signal = row['val']
        price = row['px']
        matched = filter_mask.iloc[idx] if idx < len(filter_mask) else False
        
        if signal != 0 and entry_price is None and matched:
            # New entry with filter match
            entry_price = price
            entry_signal = signal
            entry_idx = idx
            entry_matched_filter = True
                
        elif entry_price is not None and (signal == 0 or signal == -entry_signal):
            # Exit position
            exit_price = price
            
            # Calculate returns
            gross_return = (exit_price / entry_price - 1) * entry_signal
            log_return = np.log(exit_price / entry_price) * entry_signal
            
            # With 2bp round-trip costs
            cost_adjusted_return = gross_return - 0.0002  # 2bp
            log_cost_adjusted = np.log(1 + cost_adjusted_return)
            
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': 'long' if entry_signal > 0 else 'short',
                'gross_return': gross_return,
                'log_return': log_return,
                'cost_adjusted_return': cost_adjusted_return,
                'log_cost_adjusted': log_cost_adjusted,
                'filtered_entry': entry_matched_filter
            })
            
            # Check for reversal
            if signal != 0 and matched:
                entry_price = price
                entry_signal = signal
                entry_idx = idx
                entry_matched_filter = True
            else:
                entry_price = None
    
    if not trades:
        return None
    
    # Convert to DataFrame for analysis
    df_trades = pd.DataFrame(trades)
    
    # Overall metrics
    total_trades = len(df_trades)
    
    # Calculate different return metrics
    gross_edge_pct = df_trades['gross_return'].mean() * 100
    gross_edge_bps = gross_edge_pct * 100
    
    net_edge_pct = df_trades['cost_adjusted_return'].mean() * 100
    net_edge_bps = net_edge_pct * 100
    
    # Log returns (more accurate for compounding)
    log_edge_bps = df_trades['log_cost_adjusted'].mean() * 10000
    
    # Win rate
    win_rate = (df_trades['cost_adjusted_return'] > 0).mean() * 100
    
    # Time span
    first_ts = signals_df['timestamp'].min()
    last_ts = signals_df['timestamp'].max()
    trading_days = (last_ts - first_ts).days or 1
    
    # Long/Short breakdown
    long_trades = df_trades[df_trades['direction'] == 'long']
    short_trades = df_trades[df_trades['direction'] == 'short']
    
    long_metrics = {
        'count': len(long_trades),
        'gross_edge_bps': long_trades['gross_return'].mean() * 10000 if len(long_trades) > 0 else 0,
        'net_edge_bps': long_trades['cost_adjusted_return'].mean() * 10000 if len(long_trades) > 0 else 0,
        'win_rate': (long_trades['cost_adjusted_return'] > 0).mean() * 100 if len(long_trades) > 0 else 0
    }
    
    short_metrics = {
        'count': len(short_trades),
        'gross_edge_bps': short_trades['gross_return'].mean() * 10000 if len(short_trades) > 0 else 0,
        'net_edge_bps': short_trades['cost_adjusted_return'].mean() * 10000 if len(short_trades) > 0 else 0,
        'win_rate': (short_trades['cost_adjusted_return'] > 0).mean() * 100 if len(short_trades) > 0 else 0
    }
    
    return {
        'filter': filter_name,
        'total_trades': total_trades,
        'trades_per_day': total_trades / trading_days,
        'gross_edge_bps': gross_edge_bps,
        'net_edge_bps': net_edge_bps,
        'log_edge_bps': log_edge_bps,
        'win_rate': win_rate,
        'long_trades': long_metrics['count'],
        'long_edge_bps': long_metrics['net_edge_bps'],
        'long_win_rate': long_metrics['win_rate'],
        'short_trades': short_metrics['count'],
        'short_edge_bps': short_metrics['net_edge_bps'],
        'short_win_rate': short_metrics['win_rate'],
        'trading_days': trading_days
    }

# Load source data
source_df = load_source_data()

# Define MORE RELAXED filters
print("\nDefining more relaxed filter progressions...\n")

filters = [
    # Baseline
    ("No Filter", lambda df: df['val'] != 0),
    
    # Very relaxed volatility
    ("Vol>50", lambda df: (df['vol_percentile'] > 50) & (df['val'] != 0)),
    ("Vol>40", lambda df: (df['vol_percentile'] > 40) & (df['val'] != 0)),
    ("Vol>30", lambda df: (df['vol_percentile'] > 30) & (df['val'] != 0)),
    
    # Very relaxed SMA distance
    ("SMA>0.1%", lambda df: (df['sma20_distance'] > 0.001) & (df['val'] != 0)),
    ("SMA>0.05%", lambda df: (df['sma20_distance'] > 0.0005) & (df['val'] != 0)),
    
    # Very relaxed VWAP
    ("VWAP>0.05%", lambda df: (df['vwap_distance'] > 0.0005) & (df['val'] != 0)),
    ("VWAP>0.03%", lambda df: (df['vwap_distance'] > 0.0003) & (df['val'] != 0)),
    
    # Volume relaxed
    ("Volume>1.0x", lambda df: (df['volume_ratio'] > 1.0) & (df['val'] != 0)),
    ("Volume>0.9x", lambda df: (df['volume_ratio'] > 0.9) & (df['val'] != 0)),
    
    # Combined relaxed
    ("Vol>40 OR SMA>0.1%", lambda df: ((df['vol_percentile'] > 40) | (df['sma20_distance'] > 0.001)) & (df['val'] != 0)),
    ("Vol>30 OR VWAP>0.05%", lambda df: ((df['vol_percentile'] > 30) | (df['vwap_distance'] > 0.0005)) & (df['val'] != 0)),
    
    # Direction specific
    ("Longs Only", lambda df: df['val'] > 0),
    ("Shorts Only", lambda df: df['val'] < 0),
    ("Shorts + Vol>40", lambda df: (df['vol_percentile'] > 40) & (df['val'] < 0)),
    ("Shorts + Vol>30", lambda df: (df['vol_percentile'] > 30) & (df['val'] < 0)),
    
    # Time filters
    ("Morning (9:30-12)", lambda df: (df['hour'] >= 9) & (df['hour'] < 12) & (df['val'] != 0)),
    ("Afternoon (12-16)", lambda df: (df['hour'] >= 12) & (df['hour'] < 16) & (df['val'] != 0)),
    
    # Previous good filters for comparison
    ("Vol>70 (Target)", lambda df: (df['vol_percentile'] > 70) & (df['val'] != 0)),
    ("Vol>60 (Target)", lambda df: (df['vol_percentile'] > 60) & (df['val'] != 0)),
]

# Test on multiple strategies
workspace = "workspaces/signal_generation_a2d31737"
signal_pattern = str(Path(workspace) / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones/*.parquet")
signal_files = sorted(glob(signal_pattern))

# Test on top performers plus random sample
test_strategies = [1012, 1013, 1014, 1015, 1032, 1033]  # Top performers
test_strategies.extend(np.random.choice(range(100, 1400), 10, replace=False))  # Random sample

results = []

print("Analyzing strategies with detailed metrics...")
for sid in test_strategies:
    signal_file = f"{workspace}/traces/SPY_5m_1m/signals/swing_pivot_bounce_zones/SPY_5m_compiled_strategy_{sid}.parquet"
    if not Path(signal_file).exists():
        continue
    
    try:
        signals_df = pd.read_parquet(signal_file)
        
        for filter_name, filter_func in filters:
            result = analyze_filter_detailed(signals_df, source_df, filter_func, filter_name)
            if result:
                result['strategy_id'] = sid
                results.append(result)
                
        print(f"Analyzed strategy {sid}")
        
    except Exception as e:
        print(f"Error with strategy {sid}: {e}")

# Aggregate results
df_results = pd.DataFrame(results)

# Group by filter and calculate averages
filter_summary = df_results.groupby('filter').agg({
    'total_trades': 'mean',
    'trades_per_day': 'mean',
    'gross_edge_bps': 'mean',
    'net_edge_bps': 'mean',
    'log_edge_bps': 'mean',
    'win_rate': 'mean',
    'long_trades': 'mean',
    'long_edge_bps': 'mean',
    'long_win_rate': 'mean',
    'short_trades': 'mean',
    'short_edge_bps': 'mean',
    'short_win_rate': 'mean'
}).round(2)

# Calculate expected annual returns
filter_summary['daily_return_bps'] = filter_summary['net_edge_bps'] * filter_summary['trades_per_day']
filter_summary['annual_return_pct'] = filter_summary['daily_return_bps'] * 252 / 10000

# Sort by annual return
filter_summary = filter_summary.sort_values('annual_return_pct', ascending=False)

print("\n" + "="*120)
print("DETAILED FILTER ANALYSIS WITH RETURN VERIFICATION")
print("="*120)
print("\nReturn Calculation Method:")
print("- Gross Return = (Exit Price / Entry Price - 1) * Direction")
print("- Net Return = Gross Return - 0.0002 (2bp round-trip cost)")
print("- Daily Return = Net Edge * Trades Per Day")
print("- Annual Return = Daily Return * 252 trading days")
print("\n" + "-"*120)

print("\nFilter                  | Trades | T/Day | Gross | Net   | Annual | Win%  || Long Count | Long Edge | Long Win% || Short Count | Short Edge | Short Win%")
print("------------------------|--------|-------|-------|-------|--------|-------||------------|-----------|-----------||-----------—|------------|----------")

for idx, row in filter_summary.iterrows():
    print(f"{idx:23s} | {row['total_trades']:6.0f} | {row['trades_per_day']:5.2f} | "
          f"{row['gross_edge_bps']:5.0f} | {row['net_edge_bps']:5.0f} | {row['annual_return_pct']:6.2f}% | {row['win_rate']:5.1f} || "
          f"{row['long_trades']:11.0f} | {row['long_edge_bps']:9.0f} | {row['long_win_rate']:9.1f} || "
          f"{row['short_trades']:11.0f} | {row['short_edge_bps']:10.0f} | {row['short_win_rate']:10.1f}")

# Show best by different criteria
print("\n\nBEST FILTERS BY DIFFERENT CRITERIA:")
print("="*80)

print("\n1. HIGHEST FREQUENCY (Most trades per day):")
for idx, row in filter_summary.nlargest(5, 'trades_per_day').iterrows():
    print(f"   {idx:30s}: {row['trades_per_day']:5.2f} t/day, {row['net_edge_bps']:5.0f} bps = {row['annual_return_pct']:6.2f}% annual")

print("\n2. HIGHEST EDGE (Best per-trade return):")
for idx, row in filter_summary.nlargest(5, 'net_edge_bps').iterrows():
    print(f"   {idx:30s}: {row['net_edge_bps']:5.0f} bps @ {row['trades_per_day']:5.2f} t/day = {row['annual_return_pct']:6.2f}% annual")

print("\n3. BEST ANNUAL RETURN (Edge * Frequency):")
for idx, row in filter_summary.nlargest(5, 'annual_return_pct').iterrows():
    print(f"   {idx:30s}: {row['net_edge_bps']:5.0f} bps @ {row['trades_per_day']:5.2f} t/day = {row['annual_return_pct']:6.2f}% annual")

print("\n4. LONG vs SHORT COMPARISON:")
long_only = filter_summary.loc['Longs Only']
short_only = filter_summary.loc['Shorts Only']
print(f"   Longs Only: {long_only['net_edge_bps']:5.0f} bps @ {long_only['trades_per_day']:5.2f} t/day = {long_only['annual_return_pct']:6.2f}% annual")
print(f"   Shorts Only: {short_only['net_edge_bps']:5.0f} bps @ {short_only['trades_per_day']:5.2f} t/day = {short_only['annual_return_pct']:6.2f}% annual")

# Verify calculations with example
print("\n\nCALCULATION VERIFICATION EXAMPLE:")
print("="*80)
example = filter_summary.iloc[0]
print(f"Filter: {filter_summary.index[0]}")
print(f"Net Edge: {example['net_edge_bps']:.2f} bps per trade")
print(f"Trades/Day: {example['trades_per_day']:.2f}")
print(f"Daily Return: {example['net_edge_bps']:.2f} * {example['trades_per_day']:.2f} = {example['daily_return_bps']:.2f} bps")
print(f"Annual Return: {example['daily_return_bps']:.2f} * 252 / 10000 = {example['annual_return_pct']:.2f}%")

print("\n\nKEY FINDINGS:")
print("="*80)
print("1. Check if more relaxed filters achieve target frequencies (2-3 t/day)")
print("2. Verify if long/short breakdown matches expectation (shorts outperform)")
print("3. Compare to target returns from config file")