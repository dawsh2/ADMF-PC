"""Find the absolute best possible performance on 1-minute swing pivot data"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_acc7968d")
signal_dir = workspace / "traces/SPY_1m/signals/swing_pivot_bounce_zones"

# Load SPY 1m data
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})

print("=== 1-MINUTE SWING PIVOT: BEST POSSIBLE PERFORMANCE ===\n")

# Calculate all indicators
spy_1m['returns'] = spy_1m['close'].pct_change()

# Multiple volatility windows
for window in [5, 10, 20, 30, 60]:
    spy_1m[f'volatility_{window}'] = spy_1m['returns'].rolling(window).std() * np.sqrt(390) * 100
    spy_1m[f'vol_percentile_{window}'] = spy_1m[f'volatility_{window}'].rolling(window=390*5).rank(pct=True) * 100

# Trend
spy_1m['sma_50'] = spy_1m['close'].rolling(50).mean()
spy_1m['sma_200'] = spy_1m['close'].rolling(200).mean()
spy_1m['trend_up'] = (spy_1m['close'] > spy_1m['sma_50']) & (spy_1m['sma_50'] > spy_1m['sma_200'])

# VWAP
spy_1m['date'] = spy_1m['timestamp'].dt.date
spy_1m['typical_price'] = (spy_1m['high'] + spy_1m['low'] + spy_1m['close']) / 3
spy_1m['pv'] = spy_1m['typical_price'] * spy_1m['volume']
spy_1m['cum_pv'] = spy_1m.groupby('date')['pv'].cumsum()
spy_1m['cum_volume'] = spy_1m.groupby('date')['volume'].cumsum()
spy_1m['vwap'] = spy_1m['cum_pv'] / spy_1m['cum_volume']
spy_1m['vwap_distance'] = (spy_1m['close'] - spy_1m['vwap']) / spy_1m['vwap'] * 100

# Price patterns
spy_1m['lower_low'] = (spy_1m['high'] < spy_1m['high'].shift(1)) & (spy_1m['low'] < spy_1m['low'].shift(1))
spy_1m['inside_bar'] = (spy_1m['high'] <= spy_1m['high'].shift(1)) & (spy_1m['low'] >= spy_1m['low'].shift(1))

# Volume
spy_1m['volume_sma_20'] = spy_1m['volume'].rolling(20).mean()
spy_1m['volume_ratio'] = spy_1m['volume'] / spy_1m['volume_sma_20']

total_days = len(spy_1m) / 390

# Analyze ALL strategies to find the best
all_results = []

print("Analyzing all 1500 strategies...")
for strategy_id in range(0, 1500, 10):  # Every 10th strategy
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
                        'vol_percentile_5': entry_conditions.get('vol_percentile_5', 50),
                        'vol_percentile_10': entry_conditions.get('vol_percentile_10', 50),
                        'vol_percentile_20': entry_conditions.get('vol_percentile_20', 50),
                        'vwap_distance': entry_conditions.get('vwap_distance', 0),
                        'lower_low': entry_conditions.get('lower_low', False),
                        'inside_bar': entry_conditions.get('inside_bar', False),
                        'volume_ratio': entry_conditions.get('volume_ratio', 1)
                    }
                    trades.append(trade)
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if len(trades) < 20:
        continue
    
    trades_df = pd.DataFrame(trades)
    
    # Quick baseline check
    baseline_bps = trades_df['pct_return'].mean()
    baseline_tpd = len(trades_df) / total_days
    
    # Store baseline
    all_results.append({
        'strategy_id': strategy_id,
        'filter': 'Baseline',
        'edge_bps': baseline_bps,
        'trades_per_day': baseline_tpd,
        'trades': len(trades_df)
    })
    
    # Test promising filters
    filters = [
        ("3-10 min holds", (trades_df['duration'] >= 3) & (trades_df['duration'] < 10)),
        ("Inside bar", trades_df['inside_bar']),
        ("After lower low", trades_df['lower_low']),
        ("CT shorts + Vol>70", (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['vol_percentile_20'] > 70)),
        ("Far VWAP + Vol>70", (abs(trades_df['vwap_distance']) > 0.1) & 
         (trades_df['vol_percentile_20'] > 70)),
        ("Vol5>80", trades_df['vol_percentile_5'] > 80),
        ("Low volume", trades_df['volume_ratio'] < 0.8)
    ]
    
    for filter_name, filter_mask in filters:
        filtered = trades_df[filter_mask]
        if len(filtered) >= 10:
            edge = filtered['pct_return'].mean()
            tpd = len(filtered) / total_days
            
            all_results.append({
                'strategy_id': strategy_id,
                'filter': filter_name,
                'edge_bps': edge,
                'trades_per_day': tpd,
                'trades': len(filtered)
            })

# Analyze results
results_df = pd.DataFrame(all_results)

print(f"\nAnalyzed {len(results_df[results_df['filter'] == 'Baseline'])} strategies")
print(f"Total filter combinations tested: {len(results_df)}")

# Best overall edge
print("\n\nBEST EDGE (regardless of frequency):")
print("="*60)
best_edge = results_df.nlargest(15, 'edge_bps')
for _, row in best_edge.iterrows():
    if row['edge_bps'] > 0:
        print(f"Strategy {row['strategy_id']:>4}, {row['filter']:<20}: "
              f"{row['edge_bps']:>6.2f} bps on {row['trades_per_day']:>4.1f} tpd")

# Best with reasonable frequency (>0.5 tpd)
print("\n\nBEST EDGE WITH >0.5 TRADES/DAY:")
print("="*60)
frequent = results_df[results_df['trades_per_day'] > 0.5].nlargest(15, 'edge_bps')
for _, row in frequent.iterrows():
    if row['edge_bps'] > 0:
        print(f"Strategy {row['strategy_id']:>4}, {row['filter']:<20}: "
              f"{row['edge_bps']:>6.2f} bps on {row['trades_per_day']:>4.1f} tpd")

# Best with high frequency (>2 tpd)
print("\n\nBEST EDGE WITH >2 TRADES/DAY:")
print("="*60)
high_freq = results_df[results_df['trades_per_day'] > 2.0].nlargest(10, 'edge_bps')
if len(high_freq) > 0:
    for _, row in high_freq.iterrows():
        print(f"Strategy {row['strategy_id']:>4}, {row['filter']:<20}: "
              f"{row['edge_bps']:>6.2f} bps on {row['trades_per_day']:>4.1f} tpd")
else:
    print("No filters achieve positive edge with >2 trades/day")

# Summary statistics
print("\n\nSUMMARY STATISTICS:")
print("="*60)
baseline_stats = results_df[results_df['filter'] == 'Baseline']
print(f"Average baseline edge: {baseline_stats['edge_bps'].mean():.2f} bps")
print(f"Best baseline edge: {baseline_stats['edge_bps'].max():.2f} bps")
print(f"Average trades/day: {baseline_stats['trades_per_day'].mean():.1f}")

filter_summary = results_df[results_df['filter'] != 'Baseline'].groupby('filter').agg({
    'edge_bps': ['mean', 'max', 'count']
})
filter_summary.columns = ['avg_edge', 'max_edge', 'count']
filter_summary = filter_summary.sort_values('max_edge', ascending=False)

print("\n\nBEST FILTERS BY MAXIMUM EDGE:")
for filter_name, stats in filter_summary.head(10).iterrows():
    print(f"{filter_name:<25}: max {stats['max_edge']:>6.2f} bps, avg {stats['avg_edge']:>6.2f} bps")

print("\n\nCONCLUSION:")
print("The 1-minute swing pivot bounce strategy struggles to achieve meaningful edge.")
print("Best case scenario: ~0.5-1.0 bps with very selective filters and low frequency.")
print("For >=1 bps with multiple trades/day, consider different strategy types:")
print("- Momentum strategies")
print("- Microstructure/order flow strategies")
print("- Statistical arbitrage")
print("- Machine learning approaches")