"""Deep dive into 1-minute swing pivot patterns"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_acc7968d")
signal_dir = workspace / "traces/SPY_1m/signals/swing_pivot_bounce_zones"

# Load SPY 1m data
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})

print("=== DEEP DIVE: 1-MINUTE SWING PIVOT PATTERNS ===\n")

# Calculate comprehensive features
spy_1m['returns'] = spy_1m['close'].pct_change()

# Short-term volatility (more relevant for 1-min)
spy_1m['volatility_5'] = spy_1m['returns'].rolling(5).std() * np.sqrt(390) * 100
spy_1m['volatility_10'] = spy_1m['returns'].rolling(10).std() * np.sqrt(390) * 100
spy_1m['volatility_30'] = spy_1m['returns'].rolling(30).std() * np.sqrt(390) * 100

# Realized volatility patterns
spy_1m['vol_expanding'] = spy_1m['volatility_5'] > spy_1m['volatility_30']
spy_1m['vol_contracting'] = spy_1m['volatility_5'] < spy_1m['volatility_30']

# Microstructure
spy_1m['spread_pct'] = (spy_1m['high'] - spy_1m['low']) / spy_1m['close'] * 100
spy_1m['body_pct'] = abs(spy_1m['close'] - spy_1m['open']) / spy_1m['close'] * 100
spy_1m['upper_wick'] = (spy_1m['high'] - np.maximum(spy_1m['open'], spy_1m['close'])) / spy_1m['close'] * 100
spy_1m['lower_wick'] = (np.minimum(spy_1m['open'], spy_1m['close']) - spy_1m['low']) / spy_1m['close'] * 100

# Price patterns
spy_1m['higher_high'] = (spy_1m['high'] > spy_1m['high'].shift(1)) & (spy_1m['low'] > spy_1m['low'].shift(1))
spy_1m['lower_low'] = (spy_1m['high'] < spy_1m['high'].shift(1)) & (spy_1m['low'] < spy_1m['low'].shift(1))
spy_1m['inside_bar'] = (spy_1m['high'] <= spy_1m['high'].shift(1)) & (spy_1m['low'] >= spy_1m['low'].shift(1))

# Short-term momentum
spy_1m['momentum_1'] = spy_1m['returns'] * 100
spy_1m['momentum_3'] = spy_1m['close'].pct_change(3) * 100
spy_1m['momentum_5'] = spy_1m['close'].pct_change(5) * 100

# Volume patterns
spy_1m['volume_sma_10'] = spy_1m['volume'].rolling(10).mean()
spy_1m['volume_spike'] = spy_1m['volume'] > spy_1m['volume_sma_10'] * 2
spy_1m['low_volume'] = spy_1m['volume'] < spy_1m['volume_sma_10'] * 0.5

# Time of day
spy_1m['hour'] = spy_1m['timestamp'].dt.hour
spy_1m['minute'] = spy_1m['timestamp'].dt.minute
spy_1m['minutes_from_open'] = (spy_1m['hour'] - 9) * 60 + spy_1m['minute'] - 30
spy_1m['minutes_to_close'] = 960 - spy_1m['minutes_from_open']  # 16:00 close

# Opening range
spy_1m['date'] = spy_1m['timestamp'].dt.date
for date in spy_1m['date'].unique():
    date_mask = spy_1m['date'] == date
    opening_5min = spy_1m[date_mask & (spy_1m['minutes_from_open'] <= 5)]
    if len(opening_5min) > 0:
        or_high = opening_5min['high'].max()
        or_low = opening_5min['low'].min()
        spy_1m.loc[date_mask, 'or_high'] = or_high
        spy_1m.loc[date_mask, 'or_low'] = or_low
        spy_1m.loc[date_mask, 'above_or'] = spy_1m.loc[date_mask, 'close'] > or_high
        spy_1m.loc[date_mask, 'below_or'] = spy_1m.loc[date_mask, 'close'] < or_low

# Test more diverse strategy selections
strategies_to_test = [0, 50, 88, 144, 256, 512, 777, 999, 1234, 1499]

best_configs = []

for strategy_id in strategies_to_test:
    signal_file = signal_dir / f"SPY_compiled_strategy_{strategy_id}.parquet"
    if not signal_file.exists():
        continue
        
    signals = pd.read_parquet(signal_file)
    
    # Collect detailed trades
    trades = []
    entry_data = None
    
    for j in range(len(signals)):
        curr = signals.iloc[j]
        
        if entry_data is None and curr['val'] != 0:
            if curr['idx'] < len(spy_1m):
                entry_data = {
                    'idx': curr['idx'],
                    'price': curr['px'],
                    'signal': curr['val']
                }
        elif entry_data is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])):
            if entry_data and entry_data['idx'] < len(spy_1m) and curr['idx'] < len(spy_1m):
                entry_conditions = spy_1m.iloc[entry_data['idx']]
                
                pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
                duration = curr['idx'] - entry_data['idx']
                
                # Collect all features
                trade = {
                    'pct_return': pct_return,
                    'direction': 'short' if entry_data['signal'] < 0 else 'long',
                    'duration': duration,
                    
                    # Volatility state
                    'vol_5': entry_conditions.get('volatility_5', np.nan),
                    'vol_expanding': entry_conditions.get('vol_expanding', False),
                    
                    # Microstructure
                    'spread_pct': entry_conditions['spread_pct'],
                    'body_pct': entry_conditions['body_pct'],
                    'upper_wick': entry_conditions['upper_wick'],
                    'lower_wick': entry_conditions['lower_wick'],
                    
                    # Price patterns
                    'higher_high': entry_conditions.get('higher_high', False),
                    'lower_low': entry_conditions.get('lower_low', False),
                    'inside_bar': entry_conditions.get('inside_bar', False),
                    
                    # Momentum
                    'momentum_1': entry_conditions['momentum_1'],
                    'momentum_3': entry_conditions.get('momentum_3', np.nan),
                    'momentum_5': entry_conditions.get('momentum_5', np.nan),
                    
                    # Volume
                    'volume_spike': entry_conditions.get('volume_spike', False),
                    'low_volume': entry_conditions.get('low_volume', False),
                    
                    # Time
                    'minutes_from_open': entry_conditions['minutes_from_open'],
                    'minutes_to_close': entry_conditions['minutes_to_close'],
                    
                    # Opening range
                    'above_or': entry_conditions.get('above_or', np.nan),
                    'below_or': entry_conditions.get('below_or', np.nan)
                }
                trades.append(trade)
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if len(trades) < 50:
        continue
    
    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.dropna(subset=['momentum_3'])
    
    total_days = len(spy_1m) / 390
    
    print(f"\nStrategy {strategy_id}:")
    print(f"Total trades: {len(trades_df)} ({len(trades_df)/total_days:.1f}/day)")
    print(f"Baseline: {trades_df['pct_return'].mean()*100:.2f} bps")
    
    # Test unique 1-minute specific patterns
    patterns = [
        # Microstructure patterns
        ("Tight spread (<0.02%)", trades_df['spread_pct'] < 0.02),
        ("Wide spread (>0.05%)", trades_df['spread_pct'] > 0.05),
        ("Large body (>0.03%)", trades_df['body_pct'] > 0.03),
        ("Doji pattern", trades_df['body_pct'] < 0.01),
        ("Upper wick rejection", trades_df['upper_wick'] > trades_df['body_pct']),
        ("Lower wick support", trades_df['lower_wick'] > trades_df['body_pct']),
        
        # Price patterns
        ("After higher high", trades_df['higher_high']),
        ("After lower low", trades_df['lower_low']),
        ("Inside bar setup", trades_df['inside_bar']),
        
        # Momentum patterns
        ("Momentum reversal", (trades_df['momentum_1'] * trades_df['momentum_3']) < 0),
        ("Momentum continuation", (trades_df['momentum_1'] * trades_df['momentum_3']) > 0),
        ("Flat momentum", abs(trades_df['momentum_1']) < 0.01),
        ("Strong 1min move", abs(trades_df['momentum_1']) > 0.05),
        
        # Volatility patterns  
        ("Vol expansion", trades_df['vol_expanding']),
        ("Vol contraction", trades_df['vol_expanding'] == False),
        ("Very low vol", trades_df['vol_5'] < 5),
        
        # Volume patterns
        ("Volume spike entry", trades_df['volume_spike']),
        ("Low volume entry", trades_df['low_volume']),
        
        # Time patterns
        ("First 5 minutes", trades_df['minutes_from_open'] <= 5),
        ("First 30 minutes", trades_df['minutes_from_open'] <= 30),
        ("Last 30 minutes", trades_df['minutes_to_close'] <= 30),
        ("Mid-morning (10-11am)", (trades_df['minutes_from_open'] >= 30) & (trades_df['minutes_from_open'] <= 90)),
        ("Lunch (12-1pm)", (trades_df['minutes_from_open'] >= 150) & (trades_df['minutes_from_open'] <= 210)),
        
        # Opening range
        ("Above OR", trades_df['above_or'] == True),
        ("Below OR", trades_df['below_or'] == True),
        ("Inside OR", (trades_df['above_or'] == False) & (trades_df['below_or'] == False)),
        
        # Duration filters
        ("Scalp trades (<3 min)", trades_df['duration'] < 3),
        ("Quick trades (3-10 min)", (trades_df['duration'] >= 3) & (trades_df['duration'] < 10)),
        
        # Combined patterns
        ("Reversal + wide spread", (trades_df['momentum_1'] * trades_df['momentum_3'] < 0) & (trades_df['spread_pct'] > 0.04)),
        ("Spike + momentum", trades_df['volume_spike'] & (abs(trades_df['momentum_1']) > 0.03)),
        ("Morning + tight spread", (trades_df['minutes_from_open'] <= 60) & (trades_df['spread_pct'] < 0.02)),
        ("Lunch + low vol", (trades_df['minutes_from_open'] >= 150) & (trades_df['minutes_from_open'] <= 210) & trades_df['low_volume'])
    ]
    
    for pattern_name, mask in patterns:
        filtered = trades_df[mask]
        if len(filtered) >= 20:
            edge = filtered['pct_return'].mean() * 100
            tpd = len(filtered) / total_days
            win_rate = (filtered['pct_return'] > 0).mean()
            
            if edge > 0.5:  # Positive edge
                best_configs.append({
                    'strategy_id': strategy_id,
                    'pattern': pattern_name,
                    'edge_bps': edge,
                    'trades_per_day': tpd,
                    'win_rate': win_rate,
                    'total_trades': len(filtered)
                })
                
                if edge > 1.0:
                    print(f"  ✓ {pattern_name}: {edge:.2f} bps, {tpd:.1f} tpd, {win_rate:.1%} win")

# Analyze results
if best_configs:
    results_df = pd.DataFrame(best_configs)
    
    print("\n\n=== BEST 1-MINUTE PATTERNS DISCOVERED ===")
    print("="*80)
    
    # Group by pattern
    pattern_stats = results_df.groupby('pattern').agg({
        'edge_bps': ['mean', 'max', 'count'],
        'trades_per_day': 'mean'
    }).round(2)
    pattern_stats.columns = ['avg_edge', 'max_edge', 'strategies', 'avg_tpd']
    pattern_stats = pattern_stats.sort_values('avg_edge', ascending=False)
    
    print("\nMost consistent positive-edge patterns:")
    for pattern, stats in pattern_stats.head(15).iterrows():
        print(f"{pattern:<35}: {stats['avg_edge']:>6.2f} bps avg, {stats['max_edge']:>6.2f} bps max, {stats['avg_tpd']:>4.1f} tpd")
    
    # Best high-frequency patterns
    high_freq = results_df[results_df['trades_per_day'] >= 1.0].sort_values('edge_bps', ascending=False)
    if len(high_freq) > 0:
        print("\n\nBest patterns with 1+ trades/day:")
        for _, row in high_freq.head(10).iterrows():
            print(f"Strategy {row['strategy_id']:>4}: {row['pattern']:<30} = {row['edge_bps']:>6.2f} bps, {row['trades_per_day']:>4.1f} tpd")

print("\n\nKEY INSIGHTS:")
print("- Focus on microstructure patterns (spread, wicks)")
print("- Time-of-day matters more for 1-minute")
print("- Opening range breakouts/failures")
print("- Very short-term momentum patterns")
print("- Volume spikes more meaningful at 1-min level")