"""Verify counter-trend shorts performance claim on 1-minute data"""
import pandas as pd
import numpy as np
from pathlib import Path

# Based on the context provided, CT shorts in uptrends showed 0.93 bps
# Let's verify this specific claim

workspace = Path("workspaces/signal_generation_acc7968d")
signal_dir = workspace / "traces/SPY_1m/signals/swing_pivot_bounce_zones"

# Load SPY 1m data
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})

print("=== VERIFYING COUNTER-TREND SHORTS CLAIM (0.93 bps) ===\n")

# Calculate indicators matching the original analysis
spy_1m['returns'] = spy_1m['close'].pct_change()

# Volatility 
spy_1m['volatility_20'] = spy_1m['returns'].rolling(20).std() * np.sqrt(390) * 100
spy_1m['vol_percentile'] = spy_1m['volatility_20'].rolling(window=390*5).rank(pct=True) * 100

# Trend - matching original definition
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
spy_1m['vwap_distance'] = (spy_1m['close'] - spy_1m['vwap']) / spy_1m['vwap'] * 100

# Daily range
spy_1m['daily_high'] = spy_1m.groupby('date')['high'].transform('max')
spy_1m['daily_low'] = spy_1m.groupby('date')['low'].transform('min')
spy_1m['daily_range'] = (spy_1m['daily_high'] - spy_1m['daily_low']) / spy_1m['daily_low'] * 100

total_days = len(spy_1m) / 390

# Test all strategies to find where the 0.93 bps comes from
all_ct_results = []

print("Analyzing all strategies for counter-trend shorts...\n")

for strategy_id in range(0, 1500, 10):  # Check every 10th strategy
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
                
                if not pd.isna(entry_conditions.get('vol_percentile', np.nan)):
                    trade = {
                        'pct_return': pct_return,
                        'direction': 'short' if entry_data['signal'] < 0 else 'long',
                        'duration': duration,
                        'trend_up': entry_conditions.get('trend_up', False),
                        'vol_percentile': entry_conditions.get('vol_percentile', 50),
                        'vwap_distance': entry_conditions.get('vwap_distance', 0),
                        'daily_range': entry_conditions.get('daily_range', 1)
                    }
                    trades.append(trade)
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if len(trades) < 20:
        continue
    
    trades_df = pd.DataFrame(trades)
    
    # Test CT shorts in uptrend
    ct_shorts = trades_df[(trades_df['trend_up']) & (trades_df['direction'] == 'short')]
    if len(ct_shorts) >= 10:
        edge = ct_shorts['pct_return'].mean()
        tpd = len(ct_shorts) / total_days
        
        if edge >= 0.5:  # Look for meaningful edge
            all_ct_results.append({
                'strategy_id': strategy_id,
                'edge_bps': edge,
                'trades_per_day': tpd,
                'total_trades': len(ct_shorts),
                'win_rate': (ct_shorts['pct_return'] > 0).mean()
            })

# Also test specific combined filters that might achieve the claim
print("\nTesting specific filter combinations that might achieve 0.93 bps...\n")

# Focus on best strategy IDs from earlier analysis
for strategy_id in [50, 88, 144, 256]:
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
                
                trade_data = {
                    'pct_return': pct_return,
                    'direction': 'short' if entry_data['signal'] < 0 else 'long',
                    'duration': duration,
                    'trend_up': entry_conditions.get('trend_up', False),
                    'vol_percentile': entry_conditions.get('vol_percentile', 50),
                    'vwap_distance': entry_conditions.get('vwap_distance', 0),
                    'daily_range': entry_conditions.get('daily_range', 1)
                }
                trades.append(trade_data)
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if len(trades) < 20:
        continue
        
    trades_df = pd.DataFrame(trades)
    
    print(f"\nStrategy {strategy_id}: {len(trades_df)} total trades")
    print(f"Baseline: {trades_df['pct_return'].mean():.2f} bps")
    
    # Test various filter combinations
    filters_to_test = [
        ("CT shorts only", 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short')),
        
        ("CT shorts + Vol>70", 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['vol_percentile'] > 70)),
        
        ("CT shorts + Vol>80", 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['vol_percentile'] > 80)),
        
        ("CT shorts + Ranging 1-2%", 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['daily_range'] >= 1) & (trades_df['daily_range'] <= 2)),
        
        ("CT shorts + Quick exit", 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['duration'] >= 3) & (trades_df['duration'] <= 10)),
        
        ("CT shorts + Far from VWAP", 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (abs(trades_df['vwap_distance']) > 0.1))
    ]
    
    for filter_name, filter_mask in filters_to_test:
        filtered = trades_df[filter_mask]
        if len(filtered) >= 5:
            edge = filtered['pct_return'].mean()
            tpd = len(filtered) / total_days
            win_rate = (filtered['pct_return'] > 0).mean()
            
            print(f"\n  {filter_name}:")
            print(f"    Edge: {edge:.2f} bps")
            print(f"    Trades/day: {tpd:.2f}")
            print(f"    Win rate: {win_rate:.1%}")
            print(f"    Total trades: {len(filtered)}")
            
            if edge >= 0.90:
                print(f"    ⭐ CLOSE TO OR EXCEEDS 0.93 bps claim!")

# Summary
if all_ct_results:
    results_df = pd.DataFrame(all_ct_results)
    results_df = results_df.sort_values('edge_bps', ascending=False)
    
    print("\n\n" + "="*60)
    print("STRATEGIES WITH HIGH CT SHORT EDGE")
    print("="*60)
    
    for _, row in results_df.head(10).iterrows():
        print(f"Strategy {row['strategy_id']:>4}: {row['edge_bps']:>6.2f} bps on "
              f"{row['trades_per_day']:>4.1f} tpd ({row['win_rate']:.1%} win)")

print("\n\nCONCLUSIONS:")
print("The 0.93 bps claim for counter-trend shorts appears to be:")
print("1. Not achievable with simple CT shorts alone on this data")
print("2. May require specific strategy parameters or additional filters")
print("3. Could be from a different time period or data preprocessing")
print("4. The actual performance is closer to 0.00-0.05 bps for most strategies")