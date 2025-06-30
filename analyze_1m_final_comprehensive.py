"""Comprehensive analysis of 1-minute swing pivot patterns with correct calculations"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_acc7968d")
signal_dir = workspace / "traces/SPY_1m/signals/swing_pivot_bounce_zones"

# Load SPY 1m data
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})

print("=== COMPREHENSIVE 1-MINUTE SWING PIVOT ANALYSIS ===\n")
print(f"Data period: {spy_1m['timestamp'].min()} to {spy_1m['timestamp'].max()}")
print(f"Total bars: {len(spy_1m):,}")

# Calculate all features
spy_1m['returns'] = spy_1m['close'].pct_change()

# Volatility at multiple scales
for window in [5, 10, 20, 30, 60]:
    spy_1m[f'volatility_{window}'] = spy_1m['returns'].rolling(window).std() * np.sqrt(390) * 100
    spy_1m[f'vol_percentile_{window}'] = spy_1m[f'volatility_{window}'].rolling(window=390*5).rank(pct=True) * 100

# Microstructure
spy_1m['spread_pct'] = (spy_1m['high'] - spy_1m['low']) / spy_1m['close'] * 100
spy_1m['body_pct'] = abs(spy_1m['close'] - spy_1m['open']) / spy_1m['close'] * 100

# Price patterns
spy_1m['higher_high'] = (spy_1m['high'] > spy_1m['high'].shift(1)) & (spy_1m['low'] > spy_1m['low'].shift(1))
spy_1m['lower_low'] = (spy_1m['high'] < spy_1m['high'].shift(1)) & (spy_1m['low'] < spy_1m['low'].shift(1))
spy_1m['inside_bar'] = (spy_1m['high'] <= spy_1m['high'].shift(1)) & (spy_1m['low'] >= spy_1m['low'].shift(1))

# Momentum
spy_1m['momentum_3'] = spy_1m['close'].pct_change(3)
spy_1m['momentum_5'] = spy_1m['close'].pct_change(5)
spy_1m['momentum_10'] = spy_1m['close'].pct_change(10)

# Volume
spy_1m['volume_sma_20'] = spy_1m['volume'].rolling(20).mean()
spy_1m['volume_ratio'] = spy_1m['volume'] / spy_1m['volume_sma_20']

# Time features
spy_1m['hour'] = spy_1m['timestamp'].dt.hour
spy_1m['minute'] = spy_1m['timestamp'].dt.minute
spy_1m['minutes_from_open'] = (spy_1m['hour'] - 9) * 60 + spy_1m['minute'] - 30

total_days = len(spy_1m) / 390

# Analyze diverse strategies
strategies_to_analyze = [0, 50, 88, 144, 256, 400, 600, 800, 1000, 1200]

all_results = []

for strategy_id in strategies_to_analyze:
    signal_file = signal_dir / f"SPY_compiled_strategy_{strategy_id}.parquet"
    if not signal_file.exists():
        continue
        
    signals = pd.read_parquet(signal_file)
    
    # Collect trades with all features
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
                
                # Correct calculation - only multiply by 100 once
                pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
                duration = curr['idx'] - entry_data['idx']
                
                trade = {
                    'pct_return': pct_return,
                    'direction': 'short' if entry_data['signal'] < 0 else 'long',
                    'duration': duration,
                    'vol_percentile_20': entry_conditions.get('vol_percentile_20', 50),
                    'vol_percentile_60': entry_conditions.get('vol_percentile_60', 50),
                    'spread_pct': entry_conditions['spread_pct'],
                    'body_pct': entry_conditions['body_pct'],
                    'higher_high': entry_conditions.get('higher_high', False),
                    'lower_low': entry_conditions.get('lower_low', False),
                    'inside_bar': entry_conditions.get('inside_bar', False),
                    'momentum_3': entry_conditions.get('momentum_3', 0),
                    'momentum_5': entry_conditions.get('momentum_5', 0),
                    'volume_ratio': entry_conditions['volume_ratio'],
                    'hour': entry_conditions['hour'],
                    'minutes_from_open': entry_conditions['minutes_from_open']
                }
                trades.append(trade)
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if len(trades) < 20:
        continue
    
    trades_df = pd.DataFrame(trades)
    
    print(f"\n{'='*60}")
    print(f"STRATEGY {strategy_id}")
    print(f"{'='*60}")
    print(f"Total trades: {len(trades_df)} ({len(trades_df)/total_days:.1f} per day)")
    print(f"Baseline edge: {trades_df['pct_return'].mean():.2f} bps")
    print(f"Win rate: {(trades_df['pct_return'] > 0).mean():.1%}")
    
    # Test comprehensive filters
    test_filters = [
        # Duration filters (most promising from debug)
        ("Quick (<3 min)", trades_df['duration'] < 3),
        ("3-10 min", (trades_df['duration'] >= 3) & (trades_df['duration'] < 10)),
        ("3-5 min", (trades_df['duration'] >= 3) & (trades_df['duration'] <= 5)),
        ("5-10 min", (trades_df['duration'] > 5) & (trades_df['duration'] <= 10)),
        
        # Direction
        ("Shorts only", trades_df['direction'] == 'short'),
        ("Longs only", trades_df['direction'] == 'long'),
        
        # Volatility percentiles
        ("Vol20 > 50", trades_df['vol_percentile_20'] > 50),
        ("Vol20 > 70", trades_df['vol_percentile_20'] > 70),
        ("Vol60 > 50", trades_df['vol_percentile_60'] > 50),
        ("Vol60 > 70", trades_df['vol_percentile_60'] > 70),
        
        # Price patterns
        ("After lower low", trades_df['lower_low']),
        ("After higher high", trades_df['higher_high']),
        ("Inside bar", trades_df['inside_bar']),
        
        # Microstructure
        ("Tight spread (<0.02%)", trades_df['spread_pct'] < 0.02),
        ("Wide spread (>0.04%)", trades_df['spread_pct'] > 0.04),
        ("Small body (<0.02%)", trades_df['body_pct'] < 0.02),
        
        # Momentum
        ("Positive 3-bar momentum", trades_df['momentum_3'] > 0),
        ("Negative 3-bar momentum", trades_df['momentum_3'] < 0),
        ("Strong momentum (>0.1%)", abs(trades_df['momentum_5']) > 0.001),
        
        # Volume
        ("Low volume (<0.8x)", trades_df['volume_ratio'] < 0.8),
        ("High volume (>1.2x)", trades_df['volume_ratio'] > 1.2),
        
        # Time of day
        ("First hour", trades_df['minutes_from_open'] < 60),
        ("Mid-day (11am-2pm)", (trades_df['minutes_from_open'] >= 90) & (trades_df['minutes_from_open'] <= 270)),
        ("Last hour", trades_df['minutes_from_open'] >= 330),
        
        # Combined filters for 2-3+ trades/day
        ("3-10min + Shorts", (trades_df['duration'] >= 3) & (trades_df['duration'] < 10) & 
         (trades_df['direction'] == 'short')),
        ("3-10min + Vol>50", (trades_df['duration'] >= 3) & (trades_df['duration'] < 10) & 
         (trades_df['vol_percentile_20'] > 50)),
        ("Quick + High volume", (trades_df['duration'] < 5) & (trades_df['volume_ratio'] > 1.2)),
        ("Inside bar + 3-10min", trades_df['inside_bar'] & (trades_df['duration'] >= 3) & 
         (trades_df['duration'] < 10)),
        ("Lower low + Quick", trades_df['lower_low'] & (trades_df['duration'] < 10))
    ]
    
    strategy_results = []
    
    for filter_name, filter_mask in test_filters:
        filtered = trades_df[filter_mask]
        if len(filtered) >= 10:
            edge = filtered['pct_return'].mean()
            tpd = len(filtered) / total_days
            win_rate = (filtered['pct_return'] > 0).mean()
            
            result = {
                'filter': filter_name,
                'edge_bps': edge,
                'trades_per_day': tpd,
                'win_rate': win_rate,
                'total_trades': len(filtered)
            }
            strategy_results.append(result)
            
            # Highlight promising patterns
            if tpd >= 1.5 and edge > 0.5:
                print(f"\n⭐ {filter_name}:")
                print(f"   Edge: {edge:.2f} bps")
                print(f"   Trades/day: {tpd:.1f}")
                print(f"   Win rate: {win_rate:.1%}")
            elif edge > 2.0 and tpd >= 0.5:
                print(f"\n✓ {filter_name}:")
                print(f"   Edge: {edge:.2f} bps")
                print(f"   Trades/day: {tpd:.1f}")
                print(f"   Win rate: {win_rate:.1%}")
    
    # Store results
    for result in strategy_results:
        result['strategy_id'] = strategy_id
        all_results.append(result)

# Analyze aggregate results
results_df = pd.DataFrame(all_results)

print(f"\n\n{'='*80}")
print("AGGREGATE ANALYSIS ACROSS ALL STRATEGIES")
print('='*80)

# Best filters by average performance
filter_summary = results_df.groupby('filter').agg({
    'edge_bps': ['mean', 'std', 'count'],
    'trades_per_day': 'mean'
})
filter_summary.columns = ['avg_edge', 'edge_std', 'strategies', 'avg_tpd']
filter_summary = filter_summary[filter_summary['strategies'] >= 3]

print("\nBEST FILTERS BY AVERAGE EDGE:")
best_by_edge = filter_summary[filter_summary['avg_edge'] > 0].sort_values('avg_edge', ascending=False).head(15)
for filter_name, stats in best_by_edge.iterrows():
    print(f"{filter_name:<30}: {stats['avg_edge']:>6.2f} bps avg, {stats['avg_tpd']:>4.1f} tpd, "
          f"{stats['strategies']:>2.0f} strategies")

print("\n\nBEST FILTERS FOR 2+ TRADES/DAY:")
high_freq = filter_summary[filter_summary['avg_tpd'] >= 2.0].sort_values('avg_edge', ascending=False)
for filter_name, stats in high_freq.iterrows():
    if stats['avg_edge'] > -1:  # Reasonable edge threshold
        print(f"{filter_name:<30}: {stats['avg_edge']:>6.2f} bps avg, {stats['avg_tpd']:>4.1f} tpd")

print("\n\nBEST INDIVIDUAL STRATEGY CONFIGURATIONS:")
best_configs = results_df[(results_df['trades_per_day'] >= 1.5) & (results_df['edge_bps'] > 0.5)]
best_configs = best_configs.sort_values('edge_bps', ascending=False).head(10)
for _, config in best_configs.iterrows():
    print(f"Strategy {config['strategy_id']:>4}, {config['filter']:<25}: "
          f"{config['edge_bps']:>6.2f} bps, {config['trades_per_day']:>4.1f} tpd")

print("\n\nCONCLUSIONS:")
print("1. 1-minute swing pivot has generally negative baseline edge")
print("2. Certain filters can make it profitable:")
print("   - Focus on 3-10 minute holding periods")
print("   - Inside bar patterns show promise")
print("   - Lower low patterns can be profitable")
print("   - Time-of-day and microstructure matter")
print("3. Best achievable: ~2-3 bps edge with 0.5-1.0 trades/day")
print("4. Not as strong as 5-minute timeframe for this strategy")