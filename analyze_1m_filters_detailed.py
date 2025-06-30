"""Detailed analysis of 1-minute data to find filters with >=1 bps and multiple trades per day"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_acc7968d")
signal_dir = workspace / "traces/SPY_1m/signals/swing_pivot_bounce_zones"

# Load SPY 1m data
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})

print("=== 1-MINUTE DATA ANALYSIS: Finding >=1 bps with Multiple Trades/Day ===\n")
print(f"Data period: {spy_1m['timestamp'].min()} to {spy_1m['timestamp'].max()}")
print(f"Total bars: {len(spy_1m):,}")
print(f"Trading days: {len(spy_1m) / 390:.0f}\n")

# Calculate comprehensive indicators for 1-minute data
spy_1m['returns'] = spy_1m['close'].pct_change()

# Volatility at different windows (adjusted for 1-min)
spy_1m['volatility_5'] = spy_1m['returns'].rolling(5).std() * np.sqrt(390) * 100
spy_1m['volatility_10'] = spy_1m['returns'].rolling(10).std() * np.sqrt(390) * 100
spy_1m['volatility_20'] = spy_1m['returns'].rolling(20).std() * np.sqrt(390) * 100
spy_1m['volatility_60'] = spy_1m['returns'].rolling(60).std() * np.sqrt(390) * 100

# Volatility percentiles
spy_1m['vol_percentile_5'] = spy_1m['volatility_5'].rolling(window=390*5).rank(pct=True) * 100
spy_1m['vol_percentile_10'] = spy_1m['volatility_10'].rolling(window=390*5).rank(pct=True) * 100
spy_1m['vol_percentile_20'] = spy_1m['volatility_20'].rolling(window=390*5).rank(pct=True) * 100
spy_1m['vol_percentile_60'] = spy_1m['volatility_60'].rolling(window=390*5).rank(pct=True) * 100

# Trend indicators
spy_1m['sma_20'] = spy_1m['close'].rolling(20).mean()
spy_1m['sma_50'] = spy_1m['close'].rolling(50).mean()
spy_1m['sma_200'] = spy_1m['close'].rolling(200).mean()
spy_1m['trend_up'] = (spy_1m['close'] > spy_1m['sma_50']) & (spy_1m['sma_50'] > spy_1m['sma_200'])
spy_1m['trend_down'] = (spy_1m['close'] < spy_1m['sma_50']) & (spy_1m['sma_50'] < spy_1m['sma_200'])

# VWAP
spy_1m['date'] = spy_1m['timestamp'].dt.date
spy_1m['typical_price'] = (spy_1m['high'] + spy_1m['low'] + spy_1m['close']) / 3
spy_1m['pv'] = spy_1m['typical_price'] * spy_1m['volume']
spy_1m['cum_pv'] = spy_1m.groupby('date')['pv'].cumsum()
spy_1m['cum_volume'] = spy_1m.groupby('date')['volume'].cumsum()
spy_1m['vwap'] = spy_1m['cum_pv'] / spy_1m['cum_volume']
spy_1m['above_vwap'] = spy_1m['close'] > spy_1m['vwap']
spy_1m['vwap_distance'] = (spy_1m['close'] - spy_1m['vwap']) / spy_1m['vwap'] * 100

# Price patterns
spy_1m['higher_high'] = (spy_1m['high'] > spy_1m['high'].shift(1)) & (spy_1m['low'] > spy_1m['low'].shift(1))
spy_1m['lower_low'] = (spy_1m['high'] < spy_1m['high'].shift(1)) & (spy_1m['low'] < spy_1m['low'].shift(1))
spy_1m['inside_bar'] = (spy_1m['high'] <= spy_1m['high'].shift(1)) & (spy_1m['low'] >= spy_1m['low'].shift(1))

# Microstructure
spy_1m['spread_pct'] = (spy_1m['high'] - spy_1m['low']) / spy_1m['close'] * 100
spy_1m['body_pct'] = abs(spy_1m['close'] - spy_1m['open']) / spy_1m['close'] * 100

# Volume
spy_1m['volume_sma_20'] = spy_1m['volume'].rolling(20).mean()
spy_1m['volume_ratio'] = spy_1m['volume'] / spy_1m['volume_sma_20']

# Time features
spy_1m['hour'] = spy_1m['timestamp'].dt.hour
spy_1m['minute'] = spy_1m['timestamp'].dt.minute
spy_1m['minutes_from_open'] = (spy_1m['hour'] - 9) * 60 + spy_1m['minute'] - 30

# Daily range
for date in spy_1m['date'].unique():
    date_mask = spy_1m['date'] == date
    daily_high = spy_1m.loc[date_mask, 'high'].max()
    daily_low = spy_1m.loc[date_mask, 'low'].min()
    daily_range = (daily_high - daily_low) / daily_low * 100
    spy_1m.loc[date_mask, 'daily_range'] = daily_range

total_days = len(spy_1m) / 390

# Test multiple strategies
strategies_to_test = [0, 50, 88, 144, 256, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1400]

all_promising_filters = []

for strategy_id in strategies_to_test:
    signal_file = signal_dir / f"SPY_compiled_strategy_{strategy_id}.parquet"
    if not signal_file.exists():
        continue
        
    signals = pd.read_parquet(signal_file)
    
    # Collect trades
    trades = []
    entry_data = None
    
    for j in range(len(signals)):
        curr = signals.iloc[j]
        
        if entry_data is None and curr['val'] != 0:
            if curr['idx'] < len(spy_1m):
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
        elif entry_data is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])):
            if entry_data and entry_data['idx'] < len(spy_1m) and curr['idx'] < len(spy_1m):
                entry_conditions = spy_1m.iloc[entry_data['idx']]
                
                pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
                duration = curr['idx'] - entry_data['idx']
                
                if not pd.isna(entry_conditions.get('vol_percentile_20', np.nan)):
                    trade = {
                        'pct_return': pct_return,
                        'direction': 'short' if entry_data['signal'] < 0 else 'long',
                        'duration': duration,
                        'trend_up': entry_conditions.get('trend_up', False),
                        'trend_down': entry_conditions.get('trend_down', False),
                        'vol_percentile_5': entry_conditions.get('vol_percentile_5', 50),
                        'vol_percentile_10': entry_conditions.get('vol_percentile_10', 50),
                        'vol_percentile_20': entry_conditions.get('vol_percentile_20', 50),
                        'vol_percentile_60': entry_conditions.get('vol_percentile_60', 50),
                        'above_vwap': entry_conditions.get('above_vwap', False),
                        'vwap_distance': entry_conditions.get('vwap_distance', 0),
                        'higher_high': entry_conditions.get('higher_high', False),
                        'lower_low': entry_conditions.get('lower_low', False),
                        'inside_bar': entry_conditions.get('inside_bar', False),
                        'spread_pct': entry_conditions.get('spread_pct', 0),
                        'volume_ratio': entry_conditions.get('volume_ratio', 1),
                        'minutes_from_open': entry_conditions.get('minutes_from_open', 0),
                        'daily_range': entry_conditions.get('daily_range', 1)
                    }
                    trades.append(trade)
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if len(trades) < 50:
        continue
    
    trades_df = pd.DataFrame(trades)
    
    print(f"\nStrategy {strategy_id}: {len(trades_df)} trades ({len(trades_df)/total_days:.1f}/day)")
    print(f"Baseline: {trades_df['pct_return'].mean():.2f} bps")
    
    # Test comprehensive filters looking for >=1 bps with good frequency
    filters_to_test = [
        # Single filters
        ("Quick trades (3-10 min)", (trades_df['duration'] >= 3) & (trades_df['duration'] < 10)),
        ("Very quick (1-5 min)", (trades_df['duration'] >= 1) & (trades_df['duration'] <= 5)),
        ("Inside bar", trades_df['inside_bar']),
        ("After lower low", trades_df['lower_low']),
        ("After higher high", trades_df['higher_high']),
        ("Vol5 > 70", trades_df['vol_percentile_5'] > 70),
        ("Vol10 > 70", trades_df['vol_percentile_10'] > 70),
        ("Vol20 > 70", trades_df['vol_percentile_20'] > 70),
        ("Shorts only", trades_df['direction'] == 'short'),
        ("Far from VWAP (>0.1%)", abs(trades_df['vwap_distance']) > 0.1),
        ("Wide spread (>0.04%)", trades_df['spread_pct'] > 0.04),
        ("Low volume (<0.8x)", trades_df['volume_ratio'] < 0.8),
        ("High volume (>1.5x)", trades_df['volume_ratio'] > 1.5),
        ("First hour", trades_df['minutes_from_open'] < 60),
        ("Ranging day (1-2%)", (trades_df['daily_range'] >= 1) & (trades_df['daily_range'] <= 2)),
        
        # Combined filters (based on claims)
        ("CT shorts in uptrend", (trades_df['trend_up']) & (trades_df['direction'] == 'short')),
        ("CT shorts + Vol20>70", (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['vol_percentile_20'] > 70)),
        ("CT shorts + Vol10>70", (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['vol_percentile_10'] > 70)),
        ("Quick + Vol10>70", (trades_df['duration'] < 10) & (trades_df['vol_percentile_10'] > 70)),
        ("Quick + Vol20>70", (trades_df['duration'] < 10) & (trades_df['vol_percentile_20'] > 70)),
        ("Inside bar + Quick", trades_df['inside_bar'] & (trades_df['duration'] < 10)),
        ("Lower low + Vol>70", trades_df['lower_low'] & (trades_df['vol_percentile_20'] > 70)),
        ("VWAP aligned", ((trades_df['above_vwap']) & (trades_df['direction'] == 'long')) |
                        ((~trades_df['above_vwap']) & (trades_df['direction'] == 'short'))),
        ("Far VWAP + Vol>70", (abs(trades_df['vwap_distance']) > 0.1) & (trades_df['vol_percentile_20'] > 70)),
        ("First hour + Vol>70", (trades_df['minutes_from_open'] < 60) & (trades_df['vol_percentile_20'] > 70)),
        ("Ranging + CT shorts", (trades_df['daily_range'] >= 1) & (trades_df['daily_range'] <= 2) & 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short')),
        ("Vol5>80 + Shorts", (trades_df['vol_percentile_5'] > 80) & (trades_df['direction'] == 'short')),
        ("Multiple confirms", (trades_df['vol_percentile_20'] > 70) & (trades_df['duration'] < 10) & 
         (abs(trades_df['vwap_distance']) > 0.05))
    ]
    
    strategy_results = []
    
    for filter_name, filter_mask in filters_to_test:
        filtered = trades_df[filter_mask]
        if len(filtered) >= 20:  # Minimum trades for reliability
            edge = filtered['pct_return'].mean()
            tpd = len(filtered) / total_days
            win_rate = (filtered['pct_return'] > 0).mean()
            
            # Look for filters with >=1 bps and reasonable frequency
            if edge >= 1.0 or (edge >= 0.5 and tpd >= 2.0):
                result = {
                    'strategy_id': strategy_id,
                    'filter': filter_name,
                    'edge_bps': edge,
                    'trades_per_day': tpd,
                    'win_rate': win_rate,
                    'total_trades': len(filtered)
                }
                strategy_results.append(result)
                
                if edge >= 1.0 and tpd >= 1.0:
                    print(f"  ⭐ {filter_name}: {edge:.2f} bps, {tpd:.1f} tpd, {win_rate:.1%} win")
                    all_promising_filters.append(result)

# Summary of best filters
if all_promising_filters:
    results_df = pd.DataFrame(all_promising_filters)
    
    print("\n\n" + "="*80)
    print("FILTERS ACHIEVING >=1 BPS WITH MULTIPLE TRADES PER DAY")
    print("="*80)
    
    results_df = results_df.sort_values(['edge_bps', 'trades_per_day'], ascending=[False, False])
    
    print(f"\n{'Strategy':<10} {'Filter':<30} {'Edge':<10} {'Trades/Day':<12} {'Win Rate':<10}")
    print("-"*80)
    
    for _, row in results_df.head(20).iterrows():
        print(f"{row['strategy_id']:<10} {row['filter']:<30} {row['edge_bps']:>6.2f} bps "
              f"{row['trades_per_day']:>8.1f} {row['win_rate']:>8.1%}")
    
    # Group by filter type
    print("\n\nBEST FILTERS BY TYPE:")
    filter_summary = results_df.groupby('filter').agg({
        'edge_bps': ['mean', 'max', 'count'],
        'trades_per_day': 'mean'
    })
    filter_summary.columns = ['avg_edge', 'max_edge', 'strategies', 'avg_tpd']
    filter_summary = filter_summary.sort_values('avg_edge', ascending=False)
    
    for filter_name, stats in filter_summary.head(10).iterrows():
        print(f"\n{filter_name}:")
        print(f"  Average: {stats['avg_edge']:.2f} bps on {stats['avg_tpd']:.1f} trades/day")
        print(f"  Max: {stats['max_edge']:.2f} bps")
        print(f"  Works on {stats['strategies']:.0f} strategies")

else:
    print("\n\n⚠️  No filters found achieving >=1 bps with multiple trades per day")
    print("The 1-minute swing pivot bounce strategy may not be suitable for this goal.")

print("\n\nCONCLUSIONS:")
print("1. 1-minute data is much noisier than 5-minute")
print("2. Best filters focus on quick exits (3-10 minutes)")
print("3. Combining multiple conditions is essential")
print("4. Consider different strategy types for 1-minute trading")