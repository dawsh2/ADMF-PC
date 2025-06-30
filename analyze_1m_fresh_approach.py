"""Fresh analysis of 1-minute swing pivot bounce - open-minded approach"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_acc7968d")
signal_dir = workspace / "traces/SPY_1m/signals/swing_pivot_bounce_zones"

# Load SPY 1m data
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})

print("=== FRESH 1-MINUTE SWING PIVOT ANALYSIS ===\n")

# Let's first understand the data better
print("Data Overview:")
print(f"Total bars: {len(spy_1m):,}")
print(f"Date range: {spy_1m['timestamp'].min()} to {spy_1m['timestamp'].max()}")
print(f"Trading days: {len(spy_1m) / 390:.0f}")

# Calculate a broader set of features
print("\nCalculating comprehensive features...")

# Basic returns
spy_1m['returns'] = spy_1m['close'].pct_change()
spy_1m['log_returns'] = np.log(spy_1m['close'] / spy_1m['close'].shift(1))

# Multiple volatility windows
for window in [5, 10, 20, 30, 60]:
    spy_1m[f'volatility_{window}'] = spy_1m['returns'].rolling(window).std() * np.sqrt(390) * 100
    spy_1m[f'vol_percentile_{window}'] = spy_1m[f'volatility_{window}'].rolling(window=390*5).rank(pct=True) * 100

# Price action features
spy_1m['high_low_range'] = (spy_1m['high'] - spy_1m['low']) / spy_1m['close'] * 100
spy_1m['close_to_high'] = (spy_1m['high'] - spy_1m['close']) / spy_1m['close'] * 100
spy_1m['close_to_low'] = (spy_1m['close'] - spy_1m['low']) / spy_1m['close'] * 100

# Momentum
for period in [5, 10, 20]:
    spy_1m[f'momentum_{period}'] = spy_1m['close'].pct_change(period) * 100

# Volume analysis
spy_1m['volume_sma_20'] = spy_1m['volume'].rolling(20).mean()
spy_1m['volume_ratio'] = spy_1m['volume'] / spy_1m['volume_sma_20']
spy_1m['dollar_volume'] = spy_1m['close'] * spy_1m['volume']

# Microstructure
spy_1m['spread'] = spy_1m['high'] - spy_1m['low']
spy_1m['spread_pct'] = spy_1m['spread'] / spy_1m['close'] * 100

# Time features
spy_1m['hour'] = spy_1m['timestamp'].dt.hour
spy_1m['minute'] = spy_1m['timestamp'].dt.minute
spy_1m['time_of_day'] = spy_1m['hour'] * 60 + spy_1m['minute']

# Market open/close
spy_1m['first_15min'] = spy_1m['time_of_day'] <= 585  # 9:45
spy_1m['last_15min'] = spy_1m['time_of_day'] >= 945   # 15:45
spy_1m['midday'] = (spy_1m['time_of_day'] >= 690) & (spy_1m['time_of_day'] <= 810)  # 11:30-13:30

# Analyze multiple strategies with different perspectives
print("\nAnalyzing strategies with fresh perspective...\n")

# Sample more strategies
strategies_to_test = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

all_results = []

for strategy_id in strategies_to_test:
    signal_file = signal_dir / f"SPY_compiled_strategy_{strategy_id}.parquet"
    if not signal_file.exists():
        continue
        
    signals = pd.read_parquet(signal_file)
    
    # First, let's understand the signal patterns
    total_signals = len(signals[signals['val'] != 0])
    long_signals = len(signals[signals['val'] > 0])
    short_signals = len(signals[signals['val'] < 0])
    
    # Collect trades with more detailed info
    trades = []
    entry_data = None
    
    for j in range(len(signals)):
        curr = signals.iloc[j]
        
        if entry_data is None and curr['val'] != 0:
            if curr['idx'] < len(spy_1m):
                entry_data = {
                    'idx': curr['idx'],
                    'price': curr['px'],
                    'signal': curr['val'],
                    'entry_time': j
                }
        elif entry_data is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])):
            if entry_data and entry_data['idx'] < len(spy_1m) and curr['idx'] < len(spy_1m):
                entry_conditions = spy_1m.iloc[entry_data['idx']]
                exit_conditions = spy_1m.iloc[curr['idx']]
                
                pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
                trade_duration = curr['idx'] - entry_data['idx']
                
                trade = {
                    'pct_return': pct_return,
                    'direction': 'short' if entry_data['signal'] < 0 else 'long',
                    'duration_bars': trade_duration,
                    'entry_hour': entry_conditions['hour'],
                    'entry_minute': entry_conditions['minute'],
                    'time_of_day': entry_conditions['time_of_day'],
                    
                    # Volatility at different scales
                    'vol_5': entry_conditions.get('volatility_5', np.nan),
                    'vol_20': entry_conditions.get('volatility_20', np.nan),
                    'vol_60': entry_conditions.get('volatility_60', np.nan),
                    'vol_percentile_20': entry_conditions.get('vol_percentile_20', np.nan),
                    
                    # Price action
                    'high_low_range': entry_conditions['high_low_range'],
                    'momentum_5': entry_conditions.get('momentum_5', np.nan),
                    'momentum_20': entry_conditions.get('momentum_20', np.nan),
                    
                    # Volume
                    'volume_ratio': entry_conditions['volume_ratio'],
                    
                    # Time flags
                    'first_15min': entry_conditions['first_15min'],
                    'last_15min': entry_conditions['last_15min'],
                    'midday': entry_conditions['midday']
                }
                trades.append(trade)
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val'], 'entry_time': j}
            else:
                entry_data = None
    
    if not trades:
        continue
    
    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.dropna(subset=['vol_percentile_20'])
    
    if len(trades_df) < 20:  # Skip if too few trades
        continue
    
    total_days = len(spy_1m) / 390
    
    # Basic stats
    baseline_bps = trades_df['pct_return'].mean() * 100
    baseline_tpd = len(trades_df) / total_days
    win_rate = (trades_df['pct_return'] > 0).mean()
    
    print(f"\nStrategy {strategy_id} (SR={strategy_id//300 * 6 + 12}, touches={strategy_id%3 + 3}):")
    print(f"Signals: {total_signals} total ({long_signals} long, {short_signals} short)")
    print(f"Trades: {len(trades_df)} ({baseline_tpd:.1f}/day)")
    print(f"Baseline: {baseline_bps:.2f} bps, {win_rate:.1%} win rate")
    print(f"Avg duration: {trades_df['duration_bars'].mean():.1f} bars")
    
    # Store baseline
    all_results.append({
        'strategy_id': strategy_id,
        'filter': 'Baseline',
        'edge_bps': baseline_bps,
        'trades_per_day': baseline_tpd,
        'win_rate': win_rate,
        'trades': len(trades_df)
    })
    
    # Analyze by time of day
    print("\nBy time of day:")
    for period, mask in [('First 15min', trades_df['first_15min']), 
                         ('Midday', trades_df['midday']),
                         ('Last 15min', trades_df['last_15min'])]:
        period_trades = trades_df[mask]
        if len(period_trades) > 5:
            edge = period_trades['pct_return'].mean() * 100
            print(f"  {period}: {edge:.2f} bps on {len(period_trades)} trades")
    
    # Analyze by volatility regime (different approach)
    print("\nBy volatility characteristics:")
    
    # Low vs high short-term volatility
    low_vol_5 = trades_df[trades_df['vol_5'] < trades_df['vol_5'].quantile(0.3)]
    high_vol_5 = trades_df[trades_df['vol_5'] > trades_df['vol_5'].quantile(0.7)]
    
    if len(low_vol_5) > 5:
        print(f"  Low 5-bar vol: {low_vol_5['pct_return'].mean()*100:.2f} bps ({len(low_vol_5)} trades)")
    if len(high_vol_5) > 5:
        print(f"  High 5-bar vol: {high_vol_5['pct_return'].mean()*100:.2f} bps ({len(high_vol_5)} trades)")
    
    # Test completely different filters
    unique_filters = [
        # Duration-based
        ("Quick trades (<5 bars)", trades_df['duration_bars'] < 5),
        ("Medium trades (5-20 bars)", (trades_df['duration_bars'] >= 5) & (trades_df['duration_bars'] < 20)),
        ("Long trades (>20 bars)", trades_df['duration_bars'] >= 20),
        
        # Price action
        ("High range (>0.05%)", trades_df['high_low_range'] > 0.05),
        ("Low range (<0.02%)", trades_df['high_low_range'] < 0.02),
        
        # Momentum
        ("Positive momentum", trades_df['momentum_5'] > 0),
        ("Negative momentum", trades_df['momentum_5'] < 0),
        ("Strong momentum (abs>0.1%)", abs(trades_df['momentum_5']) > 0.1),
        
        # Volume
        ("Low volume (<0.8x)", trades_df['volume_ratio'] < 0.8),
        ("Normal volume", (trades_df['volume_ratio'] >= 0.8) & (trades_df['volume_ratio'] <= 1.2)),
        ("High volume (>1.2x)", trades_df['volume_ratio'] > 1.2),
        
        # Combined
        ("Morning + Low vol", trades_df['first_15min'] & (trades_df['vol_5'] < trades_df['vol_5'].median())),
        ("Midday + High range", trades_df['midday'] & (trades_df['high_low_range'] > 0.03)),
        ("Low vol regime", trades_df['vol_percentile_20'] < 30),
        ("Medium vol regime", (trades_df['vol_percentile_20'] >= 30) & (trades_df['vol_percentile_20'] < 70)),
        ("High vol regime", trades_df['vol_percentile_20'] > 70)
    ]
    
    for filter_name, filter_mask in unique_filters:
        filtered = trades_df[filter_mask]
        if len(filtered) >= 10:
            edge = filtered['pct_return'].mean() * 100
            tpd = len(filtered) / total_days
            wr = (filtered['pct_return'] > 0).mean()
            
            # Store promising results
            if edge > 0.5 or (edge > 0 and tpd > 1):
                all_results.append({
                    'strategy_id': strategy_id,
                    'filter': filter_name,
                    'edge_bps': edge,
                    'trades_per_day': tpd,
                    'win_rate': wr,
                    'trades': len(filtered)
                })
                
                # Highlight good finds
                if edge > 1.0 and tpd > 0.5:
                    print(f"  ⭐ {filter_name}: {edge:.2f} bps, {tpd:.1f} tpd, {wr:.1%} win")

# Analyze all results
results_df = pd.DataFrame(all_results)

print("\n\n=== SUMMARY OF BEST FINDINGS ===")
print("="*80)

# Best edge with reasonable frequency
good_freq = results_df[results_df['trades_per_day'] >= 0.5]
if len(good_freq) > 0:
    print("\nBest edge with >0.5 trades/day:")
    best = good_freq.sort_values('edge_bps', ascending=False).head(10)
    for _, row in best.iterrows():
        print(f"Strategy {row['strategy_id']:>4}, {row['filter']:<30}: {row['edge_bps']:>6.2f} bps, {row['trades_per_day']:>4.1f} tpd, {row['win_rate']:>5.1%} win")

# Best filters across strategies
print("\n\nMost consistent filters:")
filter_stats = results_df[results_df['filter'] != 'Baseline'].groupby('filter').agg({
    'edge_bps': ['mean', 'std', 'count'],
    'trades_per_day': 'mean'
})
filter_stats.columns = ['avg_edge', 'edge_std', 'count', 'avg_tpd']
filter_stats = filter_stats[filter_stats['count'] >= 3]  # At least 3 strategies
filter_stats = filter_stats.sort_values('avg_edge', ascending=False).head(10)

for filter_name, stats in filter_stats.iterrows():
    if stats['avg_edge'] > 0:
        print(f"{filter_name:<30}: {stats['avg_edge']:>6.2f} bps avg, {stats['avg_tpd']:>4.1f} tpd, {stats['count']:>2.0f} strategies")

print("\n\nKey insights from fresh analysis:")
print("- Check if certain times of day work better")
print("- Look at trade duration patterns")
print("- Consider microstructure (spread, range)")
print("- Test completely different volatility approaches")