"""Verify the combined filter performance claims on 5-minute data"""
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

print("=== VERIFYING COMBINED FILTER CLAIMS ===\n")

# Calculate all indicators
# Volatility
spy_5m['returns'] = spy_5m['close'].pct_change()
spy_5m['volatility_20'] = spy_5m['returns'].rolling(20).std() * np.sqrt(78) * 100
spy_5m['vol_percentile'] = spy_5m['volatility_20'].rolling(window=126).rank(pct=True) * 100

# Volume
spy_5m['volume_sma_20'] = spy_5m['volume'].rolling(20).mean()
spy_5m['volume_ratio'] = spy_5m['volume'] / spy_5m['volume_sma_20']
spy_5m['high_volume'] = spy_5m['volume_ratio'] > 1.5

# Trend
spy_5m['sma_50'] = spy_5m['close'].rolling(50).mean()
spy_5m['sma_200'] = spy_5m['close'].rolling(200).mean()
spy_5m['trend_up'] = (spy_5m['close'] > spy_5m['sma_50']) & (spy_5m['sma_50'] > spy_5m['sma_200'])
spy_5m['trend_down'] = (spy_5m['close'] < spy_5m['sma_50']) & (spy_5m['sma_50'] < spy_5m['sma_200'])

# VWAP
spy_5m['date'] = spy_5m['timestamp'].dt.date
spy_5m['typical_price'] = (spy_5m['high'] + spy_5m['low'] + spy_5m['close']) / 3
spy_5m['pv'] = spy_5m['typical_price'] * spy_5m['volume']
spy_5m['cum_pv'] = spy_5m.groupby('date')['pv'].cumsum()
spy_5m['cum_volume'] = spy_5m.groupby('date')['volume'].cumsum()
spy_5m['vwap'] = spy_5m['cum_pv'] / spy_5m['cum_volume']
spy_5m['below_vwap'] = spy_5m['close'] < spy_5m['vwap']

# Analyze the best performing strategy from our analysis (88)
# And also check a high-frequency one (80) for comparison
strategies_to_check = [88, 80, 40]

all_results = []

for strategy_id in strategies_to_check:
    print(f"\n{'='*70}")
    print(f"STRATEGY {strategy_id}")
    print('='*70)
    
    signal_file = signal_dir / f"SPY_5m_compiled_strategy_{strategy_id}.parquet"
    signals = pd.read_parquet(signal_file)
    
    # Collect all trades
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
                
                if not pd.isna(entry_conditions['vol_percentile']):
                    trade = {
                        'pct_return': pct_return,
                        'direction': 'short' if entry_data['signal'] < 0 else 'long',
                        'vol_percentile': entry_conditions['vol_percentile'],
                        'high_volume': entry_conditions['high_volume'],
                        'volume_ratio': entry_conditions['volume_ratio'],
                        'trend_down': entry_conditions['trend_down'],
                        'below_vwap': entry_conditions['below_vwap']
                    }
                    trades.append(trade)
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if not trades:
        continue
    
    trades_df = pd.DataFrame(trades)
    
    # Test the claimed filter combinations
    print(f"\nBaseline: {len(trades_df)} trades, {trades_df['pct_return'].mean()*100:.2f} bps")
    
    filters_to_test = [
        ("Vol>85", trades_df['vol_percentile'] > 85),
        ("Vol>90", trades_df['vol_percentile'] > 90),
        ("Vol>95", trades_df['vol_percentile'] > 95),
        ("Shorts only", trades_df['direction'] == 'short'),
        ("High Volume (>1.5x)", trades_df['high_volume']),
        ("Vol>85 + Shorts", (trades_df['vol_percentile'] > 85) & (trades_df['direction'] == 'short')),
        ("Vol>90 + Shorts", (trades_df['vol_percentile'] > 90) & (trades_df['direction'] == 'short')),
        ("Vol>85 + HighVolume", (trades_df['vol_percentile'] > 85) & (trades_df['high_volume'])),
        ("Vol>90 + HighVolume", (trades_df['vol_percentile'] > 90) & (trades_df['high_volume'])),
        ("Vol>85 + Shorts + HighVolume", 
         (trades_df['vol_percentile'] > 85) & (trades_df['direction'] == 'short') & (trades_df['high_volume'])),
        ("Vol>90 + Shorts + HighVolume", 
         (trades_df['vol_percentile'] > 90) & (trades_df['direction'] == 'short') & (trades_df['high_volume'])),
        ("Vol>90 + Downtrend + HighVolume", 
         (trades_df['vol_percentile'] > 90) & (trades_df['trend_down']) & (trades_df['high_volume'])),
        ("Vol>90 + BelowVWAP + HighVolume", 
         (trades_df['vol_percentile'] > 90) & (trades_df['below_vwap']) & (trades_df['high_volume']))
    ]
    
    strategy_results = {'strategy_id': strategy_id}
    
    for filter_name, filter_mask in filters_to_test:
        filtered = trades_df[filter_mask]
        if len(filtered) > 0:
            avg_return_bps = filtered['pct_return'].mean() * 100
            win_rate = (filtered['pct_return'] > 0).mean()
            
            # Calculate daily frequency
            total_days = 16614 / 78  # Total 5-min bars / bars per day
            trades_per_day = len(filtered) / total_days
            
            print(f"\n{filter_name}:")
            print(f"  Trades: {len(filtered)} ({len(filtered)/len(trades_df)*100:.1f}%)")
            print(f"  Average: {avg_return_bps:.2f} bps")
            print(f"  Win rate: {win_rate:.1%}")
            print(f"  Trades/day: {trades_per_day:.1f}")
            
            # Store key results
            if "Vol>90 + Shorts + HighVolume" in filter_name:
                strategy_results['best_filter_bps'] = avg_return_bps
                strategy_results['best_filter_trades'] = len(filtered)
                strategy_results['best_filter_tpd'] = trades_per_day
    
    all_results.append(strategy_results)

# Summary comparison
print(f"\n\n{'='*70}")
print("REALITY CHECK: CLAIMED VS ACTUAL")
print('='*70)

print("\nCLAIMED Performance:")
print("- Vol>90 + Shorts + HighVolume: 17.86 bps/trade")
print("- 16.4 trades/day")
print("- 52.9% win rate")

print("\nACTUAL Performance (Best case from our analysis):")
best_result = max(all_results, key=lambda x: x.get('best_filter_bps', 0))
if 'best_filter_bps' in best_result:
    print(f"- Strategy {best_result['strategy_id']}: {best_result['best_filter_bps']:.2f} bps/trade")
    print(f"- {best_result['best_filter_tpd']:.1f} trades/day")
    print(f"- {best_result['best_filter_trades']} total trades")

print("\nLIKELY ISSUES with claimed results:")
print("1. Different parameter optimization (SR period, zones)")
print("2. Different data period or quality")
print("3. Possible look-ahead bias in testing")
print("4. Different calculation methodology")

# Let's also check if ANY combination gets close to 17+ bps
print("\n\nSearching for ANY filter combination that achieves >10 bps...")
for strategy_id in strategies_to_check:
    signal_file = signal_dir / f"SPY_5m_compiled_strategy_{strategy_id}.parquet"
    signals = pd.read_parquet(signal_file)
    
    # Quick re-process for this strategy
    trades = []
    entry_data = None
    
    for j in range(len(signals)):
        curr = signals.iloc[j]
        
        if entry_data is None and curr['val'] != 0:
            if curr['idx'] < len(spy_5m):
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
        elif entry_data is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])):
            if entry_data and entry_data['idx'] < len(spy_5m) and curr['idx'] < len(spy_5m):
                entry_conditions = spy_5m.iloc[entry_data['idx']]
                pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
                
                if not pd.isna(entry_conditions['vol_percentile']):
                    trades.append({
                        'pct_return': pct_return,
                        'vol_percentile': entry_conditions['vol_percentile'],
                        'volume_ratio': entry_conditions['volume_ratio'],
                        'is_short': entry_data['signal'] < 0
                    })
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Try extreme filters
        extreme_filter = (trades_df['vol_percentile'] > 95) & (trades_df['volume_ratio'] > 2.0) & trades_df['is_short']
        extreme_trades = trades_df[extreme_filter]
        
        if len(extreme_trades) > 5:
            print(f"\nStrategy {strategy_id} - Most extreme filter (Vol>95 + VolRatio>2 + Shorts):")
            print(f"  {len(extreme_trades)} trades, {extreme_trades['pct_return'].mean()*100:.2f} bps")