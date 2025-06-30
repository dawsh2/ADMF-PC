"""Analyze volume relationships for 5-minute swing pivot strategies"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_320d109d")
signal_dir = workspace / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones"

# Load SPY 5m data
spy_5m = pd.read_csv("./data/SPY_5m.csv")
spy_5m['timestamp'] = pd.to_datetime(spy_5m['timestamp'])
spy_5m.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                       'Close': 'close', 'Volume': 'volume'}, inplace=True)

print("=== VOLUME ANALYSIS: 5-MINUTE SWING PIVOT STRATEGIES ===\n")

# Calculate volume metrics
print("Calculating volume indicators...")

# Basic volume metrics
spy_5m['volume_sma_20'] = spy_5m['volume'].rolling(20).mean()
spy_5m['volume_sma_50'] = spy_5m['volume'].rolling(50).mean()
spy_5m['volume_ratio_20'] = spy_5m['volume'] / spy_5m['volume_sma_20']
spy_5m['volume_ratio_50'] = spy_5m['volume'] / spy_5m['volume_sma_50']

# Volume percentiles
spy_5m['volume_percentile'] = spy_5m['volume'].rolling(window=390).rank(pct=True) * 100  # ~1 week

# Price-volume metrics
spy_5m['dollar_volume'] = spy_5m['close'] * spy_5m['volume']
spy_5m['dollar_vol_sma'] = spy_5m['dollar_volume'].rolling(20).mean()
spy_5m['dollar_vol_ratio'] = spy_5m['dollar_volume'] / spy_5m['dollar_vol_sma']

# Volume spikes
spy_5m['volume_spike'] = spy_5m['volume_ratio_20'] > 2.0
spy_5m['high_volume'] = spy_5m['volume_ratio_20'] > 1.5
spy_5m['low_volume'] = spy_5m['volume_ratio_20'] < 0.5

# Other indicators for context
spy_5m['returns'] = spy_5m['close'].pct_change()
spy_5m['volatility_20'] = spy_5m['returns'].rolling(20).std() * np.sqrt(78) * 100
spy_5m['vol_percentile'] = spy_5m['volatility_20'].rolling(window=126).rank(pct=True) * 100

# Trend
spy_5m['sma_50'] = spy_5m['close'].rolling(50).mean()
spy_5m['sma_200'] = spy_5m['close'].rolling(200).mean()
spy_5m['trend_up'] = (spy_5m['close'] > spy_5m['sma_50']) & (spy_5m['sma_50'] > spy_5m['sma_200'])

# VWAP
spy_5m['date'] = spy_5m['timestamp'].dt.date
spy_5m['typical_price'] = (spy_5m['high'] + spy_5m['low'] + spy_5m['close']) / 3
spy_5m['pv'] = spy_5m['typical_price'] * spy_5m['volume']
spy_5m['cum_pv'] = spy_5m.groupby('date')['pv'].cumsum()
spy_5m['cum_volume'] = spy_5m.groupby('date')['volume'].cumsum()
spy_5m['vwap'] = spy_5m['cum_pv'] / spy_5m['cum_volume']
spy_5m['above_vwap'] = spy_5m['close'] > spy_5m['vwap']

# Analyze best strategies
strategies_to_analyze = [88, 48, 80]

for strategy_id in strategies_to_analyze:
    print(f"\n{'='*80}")
    print(f"STRATEGY {strategy_id} - VOLUME ANALYSIS")
    print('='*80)
    
    signal_file = signal_dir / f"SPY_5m_compiled_strategy_{strategy_id}.parquet"
    signals = pd.read_parquet(signal_file)
    
    # Collect trades with volume data
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
                
                if not pd.isna(entry_conditions['volume_ratio_20']):
                    trade = {
                        'pct_return': pct_return,
                        'direction': 'long' if entry_data['signal'] > 0 else 'short',
                        'volume': entry_conditions['volume'],
                        'volume_ratio_20': entry_conditions['volume_ratio_20'],
                        'volume_ratio_50': entry_conditions['volume_ratio_50'],
                        'volume_percentile': entry_conditions['volume_percentile'],
                        'dollar_volume': entry_conditions['dollar_volume'],
                        'dollar_vol_ratio': entry_conditions['dollar_vol_ratio'],
                        'volume_spike': entry_conditions['volume_spike'],
                        'high_volume': entry_conditions['high_volume'],
                        'low_volume': entry_conditions['low_volume'],
                        'vol_percentile': entry_conditions['vol_percentile'],
                        'trend_up': entry_conditions['trend_up'],
                        'above_vwap': entry_conditions['above_vwap']
                    }
                    trades.append(trade)
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if not trades:
        continue
    
    trades_df = pd.DataFrame(trades)
    
    # 1. Volume ratio analysis
    print("\n1. PERFORMANCE BY VOLUME RATIO (20-bar)")
    print("-" * 50)
    
    volume_buckets = [
        ("Very Low (<0.5x)", trades_df['volume_ratio_20'] < 0.5),
        ("Low (0.5-0.8x)", (trades_df['volume_ratio_20'] >= 0.5) & (trades_df['volume_ratio_20'] < 0.8)),
        ("Normal (0.8-1.2x)", (trades_df['volume_ratio_20'] >= 0.8) & (trades_df['volume_ratio_20'] < 1.2)),
        ("High (1.2-2.0x)", (trades_df['volume_ratio_20'] >= 1.2) & (trades_df['volume_ratio_20'] < 2.0)),
        ("Spike (>2.0x)", trades_df['volume_ratio_20'] >= 2.0)
    ]
    
    for desc, condition in volume_buckets:
        bucket_trades = trades_df[condition]
        if len(bucket_trades) > 0:
            avg_ret = bucket_trades['pct_return'].mean()
            win_rate = (bucket_trades['pct_return'] > 0).mean()
            print(f"\n{desc}: {len(bucket_trades)} trades ({len(bucket_trades)/len(trades_df)*100:.1f}%)")
            print(f"  Average: {avg_ret*100:.2f} bps")
            print(f"  Win rate: {win_rate:.1%}")
            
            # By direction
            for direction in ['long', 'short']:
                dir_trades = bucket_trades[bucket_trades['direction'] == direction]
                if len(dir_trades) > 0:
                    print(f"  {direction}: {len(dir_trades)} trades, {dir_trades['pct_return'].mean()*100:.2f} bps")
    
    # 2. Volume percentile analysis
    print("\n\n2. PERFORMANCE BY VOLUME PERCENTILE")
    print("-" * 50)
    
    percentile_buckets = [
        ("Bottom 20%", trades_df['volume_percentile'] < 20),
        ("20-40%", (trades_df['volume_percentile'] >= 20) & (trades_df['volume_percentile'] < 40)),
        ("40-60%", (trades_df['volume_percentile'] >= 40) & (trades_df['volume_percentile'] < 60)),
        ("60-80%", (trades_df['volume_percentile'] >= 60) & (trades_df['volume_percentile'] < 80)),
        ("Top 20%", trades_df['volume_percentile'] >= 80)
    ]
    
    for desc, condition in percentile_buckets:
        bucket_trades = trades_df[condition]
        if len(bucket_trades) > 0:
            avg_ret = bucket_trades['pct_return'].mean()
            print(f"\n{desc}: {len(bucket_trades)} trades, {avg_ret*100:.2f} bps, {(bucket_trades['pct_return'] > 0).mean():.1%} win")
    
    # 3. Volume + Volatility interaction
    print("\n\n3. VOLUME × VOLATILITY INTERACTION")
    print("-" * 50)
    
    combinations = [
        ("High Vol + High Volume", (trades_df['vol_percentile'] > 70) & (trades_df['volume_ratio_20'] > 1.2)),
        ("High Vol + Low Volume", (trades_df['vol_percentile'] > 70) & (trades_df['volume_ratio_20'] < 0.8)),
        ("Low Vol + High Volume", (trades_df['vol_percentile'] < 30) & (trades_df['volume_ratio_20'] > 1.2)),
        ("Low Vol + Low Volume", (trades_df['vol_percentile'] < 30) & (trades_df['volume_ratio_20'] < 0.8))
    ]
    
    for desc, condition in combinations:
        combo_trades = trades_df[condition]
        if len(combo_trades) > 0:
            avg_ret = combo_trades['pct_return'].mean()
            win_rate = (combo_trades['pct_return'] > 0).mean()
            print(f"\n{desc}:")
            print(f"  Trades: {len(combo_trades)} ({len(combo_trades)/len(trades_df)*100:.1f}%)")
            print(f"  Average: {avg_ret*100:.2f} bps")
            print(f"  Win rate: {win_rate:.1%}")
    
    # 4. Volume patterns by market condition
    print("\n\n4. VOLUME PATTERNS BY MARKET CONDITION")
    print("-" * 50)
    
    # Trend + Volume
    print("\nIn Uptrend:")
    uptrend_trades = trades_df[trades_df['trend_up']]
    if len(uptrend_trades) > 0:
        for vol_level in ["Low (<0.8x)", "Normal (0.8-1.2x)", "High (>1.2x)"]:
            if "Low" in vol_level:
                vol_trades = uptrend_trades[uptrend_trades['volume_ratio_20'] < 0.8]
            elif "Normal" in vol_level:
                vol_trades = uptrend_trades[(uptrend_trades['volume_ratio_20'] >= 0.8) & 
                                           (uptrend_trades['volume_ratio_20'] < 1.2)]
            else:
                vol_trades = uptrend_trades[uptrend_trades['volume_ratio_20'] >= 1.2]
            
            if len(vol_trades) > 0:
                print(f"  {vol_level}: {len(vol_trades)} trades, {vol_trades['pct_return'].mean()*100:.2f} bps")
    
    # VWAP + Volume
    print("\nAbove VWAP:")
    above_vwap_trades = trades_df[trades_df['above_vwap']]
    if len(above_vwap_trades) > 0:
        for vol_level in ["Low volume", "High volume"]:
            if "Low" in vol_level:
                vol_trades = above_vwap_trades[above_vwap_trades['volume_ratio_20'] < 0.8]
            else:
                vol_trades = above_vwap_trades[above_vwap_trades['volume_ratio_20'] > 1.2]
            
            if len(vol_trades) > 0:
                longs = vol_trades[vol_trades['direction'] == 'long']
                shorts = vol_trades[vol_trades['direction'] == 'short']
                if len(longs) > 0:
                    print(f"  {vol_level} longs: {len(longs)} trades, {longs['pct_return'].mean()*100:.2f} bps")
                if len(shorts) > 0:
                    print(f"  {vol_level} shorts: {len(shorts)} trades, {shorts['pct_return'].mean()*100:.2f} bps")
    
    # 5. Best volume-based filters
    print("\n\n5. OPTIMAL VOLUME FILTERS")
    print("-" * 50)
    
    volume_filters = [
        ("Volume > 1.0x average", trades_df['volume_ratio_20'] > 1.0),
        ("Volume > 1.2x average", trades_df['volume_ratio_20'] > 1.2),
        ("Volume > 1.5x average", trades_df['volume_ratio_20'] > 1.5),
        ("Volume > 2.0x average", trades_df['volume_ratio_20'] > 2.0),
        ("Volume > 1.2x + High Vol", (trades_df['volume_ratio_20'] > 1.2) & (trades_df['vol_percentile'] > 70)),
        ("Volume spike + Any direction", trades_df['volume_spike']),
        ("Dollar volume > 1.5x", trades_df['dollar_vol_ratio'] > 1.5)
    ]
    
    best_filter = None
    best_bps = -100
    
    for desc, condition in volume_filters:
        filtered = trades_df[condition]
        if len(filtered) > 10:
            avg_ret = filtered['pct_return'].mean()
            win_rate = (filtered['pct_return'] > 0).mean()
            
            # Annual projection
            filter_ratio = len(filtered) / len(trades_df)
            
            print(f"\n{desc}:")
            print(f"  Trades: {len(filtered)} ({filter_ratio*100:.1f}%)")
            print(f"  Average: {avg_ret*100:.2f} bps")
            print(f"  Win rate: {win_rate:.1%}")
            
            if avg_ret * 100 > best_bps:
                best_bps = avg_ret * 100
                best_filter = desc
    
    if best_filter:
        print(f"\n*** Best volume filter: {best_filter} with {best_bps:.2f} bps ***")

# Summary
print(f"\n\n{'='*80}")
print("SUMMARY: VOLUME RELATIONSHIPS")
print('='*80)

print("\nKey findings from volume analysis:")
print("1. Check if high volume (>1.2x) improves performance like on 1-minute")
print("2. Look for volume spike patterns")
print("3. Examine volume + volatility combinations")
print("4. Compare to 1-minute where volume > 1.0x was crucial")