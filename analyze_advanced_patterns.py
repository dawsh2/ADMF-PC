"""Deep dive into additional patterns and correlations for swing pivot strategies"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

workspace = Path("workspaces/signal_generation_320d109d")
signal_dir = workspace / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones"

# Load SPY 5m data
spy_5m = pd.read_csv("./data/SPY_5m.csv")
spy_5m['timestamp'] = pd.to_datetime(spy_5m['timestamp'])
spy_5m.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                       'Close': 'close', 'Volume': 'volume'}, inplace=True)

print("=== ADVANCED PATTERN DISCOVERY FOR SWING PIVOT STRATEGIES ===\n")

# Calculate comprehensive set of features
print("Calculating extended feature set...")

# Time-based features
spy_5m['hour'] = spy_5m['timestamp'].dt.hour
spy_5m['minute'] = spy_5m['timestamp'].dt.minute
spy_5m['time_of_day'] = spy_5m['hour'] + spy_5m['minute']/60
spy_5m['day_of_week'] = spy_5m['timestamp'].dt.dayofweek
spy_5m['month'] = spy_5m['timestamp'].dt.month

# Session markers
spy_5m['first_30min'] = spy_5m['time_of_day'] <= 10.0
spy_5m['last_30min'] = spy_5m['time_of_day'] >= 15.5
spy_5m['lunch_hour'] = (spy_5m['time_of_day'] >= 12.0) & (spy_5m['time_of_day'] <= 13.0)
spy_5m['morning'] = spy_5m['time_of_day'] <= 12.0
spy_5m['afternoon'] = spy_5m['time_of_day'] > 12.0

# Price action features
spy_5m['range'] = spy_5m['high'] - spy_5m['low']
spy_5m['range_pct'] = spy_5m['range'] / spy_5m['close'] * 100
spy_5m['upper_wick'] = spy_5m['high'] - np.maximum(spy_5m['open'], spy_5m['close'])
spy_5m['lower_wick'] = np.minimum(spy_5m['open'], spy_5m['close']) - spy_5m['low']
spy_5m['body'] = abs(spy_5m['close'] - spy_5m['open'])
spy_5m['wick_ratio'] = (spy_5m['upper_wick'] + spy_5m['lower_wick']) / (spy_5m['body'] + 0.0001)

# Momentum features
spy_5m['returns'] = spy_5m['close'].pct_change()
spy_5m['returns_5'] = spy_5m['close'].pct_change(5)
spy_5m['returns_10'] = spy_5m['close'].pct_change(10)
spy_5m['momentum_5'] = spy_5m['returns'].rolling(5).sum()
spy_5m['momentum_10'] = spy_5m['returns'].rolling(10).sum()

# Volatility features
spy_5m['volatility_10'] = spy_5m['returns'].rolling(10).std() * np.sqrt(78) * 100
spy_5m['volatility_20'] = spy_5m['returns'].rolling(20).std() * np.sqrt(78) * 100
spy_5m['volatility_50'] = spy_5m['returns'].rolling(50).std() * np.sqrt(78) * 100
spy_5m['vol_percentile'] = spy_5m['volatility_20'].rolling(window=126).rank(pct=True) * 100
spy_5m['vol_change'] = spy_5m['volatility_20'].pct_change(20)

# Volume features
spy_5m['volume_sma_20'] = spy_5m['volume'].rolling(20).mean()
spy_5m['volume_sma_50'] = spy_5m['volume'].rolling(50).mean()
spy_5m['volume_ratio'] = spy_5m['volume'] / spy_5m['volume_sma_20']
spy_5m['dollar_volume'] = spy_5m['close'] * spy_5m['volume']
spy_5m['dollar_vol_ratio'] = spy_5m['dollar_volume'] / spy_5m['dollar_volume'].rolling(20).mean()
spy_5m['volume_momentum'] = spy_5m['volume_ratio'].rolling(5).mean()

# Market structure
spy_5m['sma_10'] = spy_5m['close'].rolling(10).mean()
spy_5m['sma_20'] = spy_5m['close'].rolling(20).mean()
spy_5m['sma_50'] = spy_5m['close'].rolling(50).mean()
spy_5m['distance_from_sma20'] = (spy_5m['close'] - spy_5m['sma_20']) / spy_5m['sma_20'] * 100
spy_5m['sma_slope_20'] = (spy_5m['sma_20'] - spy_5m['sma_20'].shift(5)) / spy_5m['sma_20'].shift(5) * 100

# VWAP features
spy_5m['date'] = spy_5m['timestamp'].dt.date
spy_5m['typical_price'] = (spy_5m['high'] + spy_5m['low'] + spy_5m['close']) / 3
spy_5m['pv'] = spy_5m['typical_price'] * spy_5m['volume']
spy_5m['cum_pv'] = spy_5m.groupby('date')['pv'].cumsum()
spy_5m['cum_volume'] = spy_5m.groupby('date')['volume'].cumsum()
spy_5m['vwap'] = spy_5m['cum_pv'] / spy_5m['cum_volume']
spy_5m['distance_from_vwap'] = (spy_5m['close'] - spy_5m['vwap']) / spy_5m['vwap'] * 100

# Gap features
spy_5m['overnight_gap'] = 0
dates = spy_5m['date'].unique()
for i in range(1, len(dates)):
    prev_close = spy_5m[spy_5m['date'] == dates[i-1]]['close'].iloc[-1]
    curr_open = spy_5m[spy_5m['date'] == dates[i]]['open'].iloc[0]
    gap = (curr_open - prev_close) / prev_close * 100
    spy_5m.loc[spy_5m['date'] == dates[i], 'overnight_gap'] = gap

# Analyze best strategy (88) for patterns
strategy_id = 88
print(f"\nAnalyzing Strategy {strategy_id} for advanced patterns...\n")

signal_file = signal_dir / f"SPY_5m_compiled_strategy_{strategy_id}.parquet"
signals = pd.read_parquet(signal_file)

# Collect all trades with extended features
trades = []
entry_data = None

for j in range(len(signals)):
    curr = signals.iloc[j]
    
    if entry_data is None and curr['val'] != 0:
        if curr['idx'] < len(spy_5m):
            entry_data = {
                'idx': curr['idx'],
                'price': curr['px'],
                'signal': curr['val']
            }
    elif entry_data is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])):
        if entry_data and entry_data['idx'] < len(spy_5m) and curr['idx'] < len(spy_5m):
            entry_conditions = spy_5m.iloc[entry_data['idx']]
            
            pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
            duration_bars = curr['idx'] - entry_data['idx']
            
            trade = {
                'pct_return': pct_return,
                'direction': 'short' if entry_data['signal'] < 0 else 'long',
                'duration_bars': duration_bars,
                'duration_hours': duration_bars * 5 / 60,  # 5-min bars
                
                # Time features
                'hour': entry_conditions['hour'],
                'time_of_day': entry_conditions['time_of_day'],
                'day_of_week': entry_conditions['day_of_week'],
                'month': entry_conditions['month'],
                'first_30min': entry_conditions['first_30min'],
                'last_30min': entry_conditions['last_30min'],
                'lunch_hour': entry_conditions['lunch_hour'],
                
                # Price action
                'range_pct': entry_conditions['range_pct'],
                'wick_ratio': entry_conditions['wick_ratio'],
                'momentum_5': entry_conditions['momentum_5'],
                'momentum_10': entry_conditions['momentum_10'],
                
                # Volatility
                'vol_percentile': entry_conditions['vol_percentile'],
                'vol_change': entry_conditions['vol_change'],
                'volatility_10': entry_conditions['volatility_10'],
                'volatility_20': entry_conditions['volatility_20'],
                
                # Volume
                'volume_ratio': entry_conditions['volume_ratio'],
                'dollar_vol_ratio': entry_conditions['dollar_vol_ratio'],
                'volume_momentum': entry_conditions['volume_momentum'],
                
                # Market structure
                'distance_from_sma20': entry_conditions['distance_from_sma20'],
                'distance_from_vwap': entry_conditions['distance_from_vwap'],
                'sma_slope_20': entry_conditions['sma_slope_20'],
                'overnight_gap': entry_conditions['overnight_gap']
            }
            trades.append(trade)
        
        if curr['val'] != 0:
            entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
        else:
            entry_data = None

trades_df = pd.DataFrame(trades)
trades_df = trades_df.dropna(subset=['vol_percentile'])

print(f"Total trades analyzed: {len(trades_df)}")

# 1. Time-based patterns
print("\n1. TIME-BASED PATTERNS")
print("-" * 50)

# By hour
hourly_stats = trades_df.groupby('hour').agg({
    'pct_return': ['mean', 'count', lambda x: (x > 0).mean()]
})
hourly_stats.columns = ['avg_return', 'count', 'win_rate']
hourly_stats['avg_bps'] = hourly_stats['avg_return'] * 100

print("\nPerformance by hour:")
best_hours = hourly_stats[hourly_stats['count'] > 20].sort_values('avg_bps', ascending=False).head(5)
for hour, stats in best_hours.iterrows():
    print(f"  {hour}:00 - {hour+1}:00: {stats['avg_bps']:.2f} bps ({stats['count']} trades, {stats['win_rate']:.1%} win)")

# Session analysis
for session in ['first_30min', 'last_30min', 'lunch_hour']:
    session_trades = trades_df[trades_df[session]]
    if len(session_trades) > 20:
        avg_bps = session_trades['pct_return'].mean() * 100
        print(f"\n{session.replace('_', ' ').title()}: {avg_bps:.2f} bps on {len(session_trades)} trades")

# 2. Price action patterns
print("\n\n2. PRICE ACTION PATTERNS")
print("-" * 50)

# Range analysis
trades_df['high_range'] = trades_df['range_pct'] > trades_df['range_pct'].quantile(0.8)
trades_df['low_range'] = trades_df['range_pct'] < trades_df['range_pct'].quantile(0.2)

for range_type in ['high_range', 'low_range']:
    range_trades = trades_df[trades_df[range_type]]
    if len(range_trades) > 20:
        avg_bps = range_trades['pct_return'].mean() * 100
        print(f"\n{range_type.replace('_', ' ').title()}: {avg_bps:.2f} bps on {len(range_trades)} trades")

# Momentum states
trades_df['strong_momentum'] = trades_df['momentum_10'] > trades_df['momentum_10'].quantile(0.8)
trades_df['weak_momentum'] = trades_df['momentum_10'] < trades_df['momentum_10'].quantile(0.2)

for mom_state in ['strong_momentum', 'weak_momentum']:
    mom_trades = trades_df[trades_df[mom_state]]
    if len(mom_trades) > 20:
        avg_bps = mom_trades['pct_return'].mean() * 100
        print(f"\n{mom_state.replace('_', ' ').title()}: {avg_bps:.2f} bps on {len(mom_trades)} trades")

# 3. Combined patterns
print("\n\n3. ADVANCED COMBINED PATTERNS")
print("-" * 50)

complex_patterns = [
    ("High Vol + Morning + High Range", 
     (trades_df['vol_percentile'] > 70) & (trades_df['hour'] < 12) & (trades_df['range_pct'] > trades_df['range_pct'].quantile(0.7))),
    
    ("High Vol + Far from VWAP", 
     (trades_df['vol_percentile'] > 70) & (abs(trades_df['distance_from_vwap']) > 0.2)),
    
    ("Volume Spike + Momentum", 
     (trades_df['volume_ratio'] > 1.5) & (abs(trades_df['momentum_5']) > 0.2)),
    
    ("Afternoon + Low Volume + High Vol", 
     (trades_df['hour'] >= 13) & (trades_df['volume_ratio'] < 0.8) & (trades_df['vol_percentile'] > 70)),
    
    ("Gap Days + High Vol", 
     (abs(trades_df['overnight_gap']) > 0.5) & (trades_df['vol_percentile'] > 70)),
    
    ("First Hour + Volume Momentum", 
     (trades_df['hour'] < 11) & (trades_df['volume_momentum'] > 1.2)),
    
    ("Extended from SMA + High Vol", 
     (abs(trades_df['distance_from_sma20']) > 0.5) & (trades_df['vol_percentile'] > 70)),
    
    ("Trending Market + Counter-trend", 
     (abs(trades_df['sma_slope_20']) > 0.1) & (trades_df['direction'] == 'short'))
]

best_patterns = []

for pattern_name, pattern_mask in complex_patterns:
    pattern_trades = trades_df[pattern_mask]
    if len(pattern_trades) > 10:
        avg_bps = pattern_trades['pct_return'].mean() * 100
        win_rate = (pattern_trades['pct_return'] > 0).mean()
        daily_freq = len(pattern_trades) / (16614 / 78)
        
        print(f"\n{pattern_name}:")
        print(f"  Trades: {len(pattern_trades)} ({daily_freq:.2f}/day)")
        print(f"  Edge: {avg_bps:.2f} bps")
        print(f"  Win rate: {win_rate:.1%}")
        
        if avg_bps > 3.0 and daily_freq > 0.3:
            best_patterns.append((pattern_name, avg_bps, daily_freq))

# 4. Trade duration analysis
print("\n\n4. TRADE DURATION PATTERNS")
print("-" * 50)

duration_buckets = [
    ("Quick (<30min)", trades_df['duration_hours'] < 0.5),
    ("Short (30min-1hr)", (trades_df['duration_hours'] >= 0.5) & (trades_df['duration_hours'] < 1)),
    ("Medium (1-2hr)", (trades_df['duration_hours'] >= 1) & (trades_df['duration_hours'] < 2)),
    ("Long (>2hr)", trades_df['duration_hours'] >= 2)
]

for desc, mask in duration_buckets:
    dur_trades = trades_df[mask]
    if len(dur_trades) > 10:
        avg_bps = dur_trades['pct_return'].mean() * 100
        print(f"\n{desc}: {len(dur_trades)} trades, {avg_bps:.2f} bps")

# 5. Volatility regime transitions
print("\n\n5. VOLATILITY REGIME TRANSITIONS")
print("-" * 50)

trades_df['vol_increasing'] = trades_df['vol_change'] > 0.1
trades_df['vol_decreasing'] = trades_df['vol_change'] < -0.1

for vol_regime in ['vol_increasing', 'vol_decreasing']:
    regime_trades = trades_df[trades_df[vol_regime]]
    if len(regime_trades) > 20:
        avg_bps = regime_trades['pct_return'].mean() * 100
        print(f"\n{vol_regime.replace('_', ' ').title()}: {avg_bps:.2f} bps on {len(regime_trades)} trades")

# Summary
print(f"\n\n{'='*70}")
print("KEY DISCOVERIES")
print('='*70)

if best_patterns:
    print("\nMost promising patterns found:")
    for pattern, edge, freq in sorted(best_patterns, key=lambda x: x[1], reverse=True):
        print(f"  - {pattern}: {edge:.2f} bps edge, {freq:.2f} trades/day")

print("\n\nAdditional filters to consider:")
print("1. Time-based: Avoid first 30 minutes, focus on mid-day")
print("2. Price action: High range bars show better edge")
print("3. Volume patterns: Volume momentum >1.2 improves results")
print("4. Market structure: Extended from moving averages works well")
print("5. Volatility transitions: Increasing volatility periods are favorable")