"""Comprehensive filter analysis for 15-minute swing pivot bounce"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_ffbf0538")
signal_file = workspace / "traces/SPY_15m_1m/signals/swing_pivot_bounce/SPY_15m_compiled_strategy_0.parquet"

print("=== 15-MINUTE SWING PIVOT: COMPREHENSIVE FILTER ANALYSIS ===\n")

# Load signals
signals = pd.read_parquet(signal_file)

# Load 15-minute SPY data
spy_15m = pd.read_csv("./data/SPY_15m.csv")
spy_15m['timestamp'] = pd.to_datetime(spy_15m['timestamp'])
spy_15m.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                        'Close': 'close', 'Volume': 'volume'}, inplace=True)

# Use appropriate subset
max_idx = min(signals['idx'].max() + 5, len(spy_15m))
spy_subset = spy_15m.iloc[:max_idx].copy()

# Calculate comprehensive indicators
spy_subset['returns'] = spy_subset['close'].pct_change()

# Multiple volatility windows
spy_subset['volatility_10'] = spy_subset['returns'].rolling(10).std() * np.sqrt(26) * 100
spy_subset['volatility_20'] = spy_subset['returns'].rolling(20).std() * np.sqrt(26) * 100
spy_subset['volatility_40'] = spy_subset['returns'].rolling(40).std() * np.sqrt(26) * 100

# Volatility percentiles
for window in [10, 20, 40]:
    spy_subset[f'vol_percentile_{window}'] = spy_subset[f'volatility_{window}'].rolling(window=26*20).rank(pct=True) * 100

# Trend indicators
spy_subset['sma_20'] = spy_subset['close'].rolling(20).mean()
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
spy_subset['vwap_distance'] = (spy_subset['close'] - spy_subset['vwap']) / spy_subset['vwap'] * 100

# Daily range
spy_subset['daily_high'] = spy_subset.groupby('date')['high'].transform('max')
spy_subset['daily_low'] = spy_subset.groupby('date')['low'].transform('min')
spy_subset['daily_range'] = (spy_subset['daily_high'] - spy_subset['daily_low']) / spy_subset['daily_low'] * 100

# Time of day
spy_subset['hour'] = spy_subset['timestamp'].dt.hour
spy_subset['minute'] = spy_subset['timestamp'].dt.minute
spy_subset['time_slot'] = spy_subset['hour'] * 60 + spy_subset['minute']
spy_subset['morning'] = (spy_subset['time_slot'] >= 570) & (spy_subset['time_slot'] < 720)  # 9:30-12:00
spy_subset['afternoon'] = (spy_subset['time_slot'] >= 720) & (spy_subset['time_slot'] < 900)  # 12:00-15:00
spy_subset['last_hour'] = spy_subset['time_slot'] >= 900  # 15:00-16:00

# Price patterns
spy_subset['range_pct'] = (spy_subset['high'] - spy_subset['low']) / spy_subset['close'] * 100
spy_subset['body_pct'] = abs(spy_subset['close'] - spy_subset['open']) / spy_subset['close'] * 100
spy_subset['upper_wick'] = (spy_subset['high'] - spy_subset[['open', 'close']].max(axis=1)) / spy_subset['close'] * 100
spy_subset['lower_wick'] = (spy_subset[['open', 'close']].min(axis=1) - spy_subset['low']) / spy_subset['close'] * 100

# Volume
spy_subset['volume_sma'] = spy_subset['volume'].rolling(20).mean()
spy_subset['volume_ratio'] = spy_subset['volume'] / spy_subset['volume_sma']

# Collect trades
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
                'vol_percentile_10': entry_conditions.get('vol_percentile_10', 50),
                'vol_percentile_20': entry_conditions.get('vol_percentile_20', 50),
                'vol_percentile_40': entry_conditions.get('vol_percentile_40', 50),
                'vwap_distance': entry_conditions.get('vwap_distance', 0),
                'daily_range': entry_conditions.get('daily_range', 1),
                'morning': entry_conditions.get('morning', False),
                'afternoon': entry_conditions.get('afternoon', False),
                'last_hour': entry_conditions.get('last_hour', False),
                'range_pct': entry_conditions.get('range_pct', 0),
                'volume_ratio': entry_conditions.get('volume_ratio', 1),
                'upper_wick': entry_conditions.get('upper_wick', 0),
                'lower_wick': entry_conditions.get('lower_wick', 0)
            }
            trades.append(trade)
        
        if curr['val'] != 0:
            entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
        else:
            entry_data = None

trades_df = pd.DataFrame(trades)
total_days = 5673 / 26

print(f"Total trades: {len(trades_df)}")
print(f"Baseline performance: {trades_df['pct_return'].mean():.2f} bps on {len(trades_df)/total_days:.2f} tpd")

# Test comprehensive filters
print("\n\n=== SINGLE FILTER ANALYSIS ===")

single_filters = [
    # Volatility
    ("Vol20 > 70", trades_df['vol_percentile_20'] > 70),
    ("Vol20 > 80", trades_df['vol_percentile_20'] > 80),
    ("Vol10 > 80", trades_df['vol_percentile_10'] > 80),
    
    # Trend
    ("Trend up", trades_df['trend_up']),
    ("Trend down", trades_df['trend_down']),
    ("CT shorts", (trades_df['trend_up']) & (trades_df['direction'] == 'short')),
    ("CT longs", (trades_df['trend_down']) & (trades_df['direction'] == 'long')),
    
    # Time of day
    ("Morning", trades_df['morning']),
    ("Afternoon", trades_df['afternoon']),
    ("Last hour", trades_df['last_hour']),
    
    # Market conditions
    ("Daily range > 1%", trades_df['daily_range'] > 1),
    ("Daily range 1-2%", (trades_df['daily_range'] >= 1) & (trades_df['daily_range'] <= 2)),
    ("Far from VWAP", abs(trades_df['vwap_distance']) > 0.2),
    
    # Microstructure
    ("Wide range bars", trades_df['range_pct'] > 0.15),
    ("High volume", trades_df['volume_ratio'] > 1.5),
    ("Long wicks", (trades_df['upper_wick'] > 0.05) | (trades_df['lower_wick'] > 0.05)),
    
    # Hold duration
    ("Quick exits", trades_df['duration'] <= 2),
    ("Hold 3-5 bars", (trades_df['duration'] >= 3) & (trades_df['duration'] <= 5))
]

promising_filters = []

for filter_name, filter_mask in single_filters:
    filtered = trades_df[filter_mask]
    if len(filtered) >= 20:
        edge = filtered['pct_return'].mean()
        tpd = len(filtered) / total_days
        win_rate = (filtered['pct_return'] > 0).mean()
        
        if abs(edge) >= 0.5:  # Look for meaningful edge
            print(f"\n{filter_name}:")
            print(f"  Edge: {edge:.2f} bps")
            print(f"  Trades/day: {tpd:.2f}")
            print(f"  Win rate: {win_rate:.1%}")
            print(f"  Total trades: {len(filtered)}")
            
            if edge > 0:
                promising_filters.append((filter_name, filter_mask, edge))

# Test combinations of promising filters
if promising_filters:
    print("\n\n=== COMBINED FILTER ANALYSIS ===")
    
    # Try combinations
    for i, (name1, mask1, edge1) in enumerate(promising_filters):
        for name2, mask2, edge2 in promising_filters[i+1:]:
            combined_mask = mask1 & mask2
            combined = trades_df[combined_mask]
            
            if len(combined) >= 10:
                edge = combined['pct_return'].mean()
                tpd = len(combined) / total_days
                
                if edge >= 1.0:  # Looking for >=1 bps
                    print(f"\n{name1} + {name2}:")
                    print(f"  Edge: {edge:.2f} bps")
                    print(f"  Trades/day: {tpd:.2f}")
                    print(f"  Win rate: {(combined['pct_return'] > 0).mean():.1%}")

# Special combinations based on insights
print("\n\n=== TARGETED COMBINATIONS ===")

targeted_combos = [
    ("Morning + Vol>70", (trades_df['morning']) & (trades_df['vol_percentile_20'] > 70)),
    ("CT shorts + Vol>80", (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
     (trades_df['vol_percentile_20'] > 80)),
    ("Quick exits + Vol>70", (trades_df['duration'] <= 2) & (trades_df['vol_percentile_20'] > 70)),
    ("Last hour + Trend up", (trades_df['last_hour']) & (trades_df['trend_up'])),
    ("Wide bars + High vol", (trades_df['range_pct'] > 0.15) & (trades_df['volume_ratio'] > 1.5)),
    ("Far VWAP + Vol>70", (abs(trades_df['vwap_distance']) > 0.2) & (trades_df['vol_percentile_20'] > 70)),
    ("Morning + Far VWAP", (trades_df['morning']) & (abs(trades_df['vwap_distance']) > 0.2))
]

best_combos = []

for combo_name, combo_mask in targeted_combos:
    filtered = trades_df[combo_mask]
    if len(filtered) >= 10:
        edge = filtered['pct_return'].mean()
        tpd = len(filtered) / total_days
        
        print(f"\n{combo_name}:")
        print(f"  Edge: {edge:.2f} bps")
        print(f"  Trades/day: {tpd:.2f}")
        print(f"  Win rate: {(filtered['pct_return'] > 0).mean():.1%}")
        print(f"  Total trades: {len(filtered)}")
        
        if edge >= 1.0 and tpd >= 0.5:
            best_combos.append((combo_name, edge, tpd))

print("\n\n=== CONCLUSIONS ===")
if best_combos:
    print(f"\nFound {len(best_combos)} combinations with >=1 bps edge:")
    for name, edge, tpd in best_combos:
        print(f"  {name}: {edge:.2f} bps on {tpd:.2f} trades/day")
else:
    print("\nNo filter combinations achieve >=1 bps with reasonable frequency.")
    print("The 15-minute timeframe may be too slow for meaningful swing pivot signals.")
    
print("\n15-minute trades 3.35x/day but lacks edge. Consider:")
print("1. Stick with 5-minute timeframe (proven 2.18 bps)")
print("2. Try different strategies for 15-minute")
print("3. Use 15-minute for trend confirmation only")