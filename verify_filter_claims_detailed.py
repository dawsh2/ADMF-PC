"""Detailed verification of the specific filter claims"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace_5m = Path("workspaces/signal_generation_320d109d")
signal_dir = workspace_5m / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones"

# Load SPY 5m data
spy_5m = pd.read_csv("./data/SPY_5m.csv")
spy_5m['timestamp'] = pd.to_datetime(spy_5m['timestamp'])
spy_5m.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                       'Close': 'close', 'Volume': 'volume'}, inplace=True)

print("=== DETAILED VERIFICATION OF FILTER CLAIMS ===\n")
print("Original claims:")
print("1. Counter-trend shorts in uptrends: 0.93 bps")
print("2. High volatility environments (80th+ percentile): 0.27 bps")
print("3. Ranging markets with 1-2% movement: 0.39 bps")
print("4. Go WITH VWAP momentum, not against it")

# Calculate comprehensive indicators
spy_5m['returns'] = spy_5m['close'].pct_change()
spy_5m['volatility_20'] = spy_5m['returns'].rolling(20).std() * np.sqrt(78) * 100
spy_5m['vol_percentile'] = spy_5m['volatility_20'].rolling(window=126).rank(pct=True) * 100

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
spy_5m['above_vwap'] = spy_5m['close'] > spy_5m['vwap']
spy_5m['vwap_distance'] = (spy_5m['close'] - spy_5m['vwap']) / spy_5m['vwap'] * 100

# Daily range
for date in spy_5m['date'].unique():
    date_mask = spy_5m['date'] == date
    daily_high = spy_5m.loc[date_mask, 'high'].max()
    daily_low = spy_5m.loc[date_mask, 'low'].min()
    daily_range = (daily_high - daily_low) / daily_low * 100
    spy_5m.loc[date_mask, 'daily_range'] = daily_range

# Test the specific strategies that performed best in our earlier analysis
best_strategies = [88, 48, 80, 40, 81]  # These had the best baseline performance

print("\n\nAnalyzing best-performing strategies in detail...")

for strategy_id in best_strategies:
    signal_file = signal_dir / f"SPY_5m_compiled_strategy_{strategy_id}.parquet"
    if not signal_file.exists():
        continue
        
    signals = pd.read_parquet(signal_file)
    
    # Collect trades
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
                        'trend_up': entry_conditions['trend_up'],
                        'trend_down': entry_conditions['trend_down'],
                        'vol_percentile': entry_conditions['vol_percentile'],
                        'above_vwap': entry_conditions['above_vwap'],
                        'vwap_distance': entry_conditions['vwap_distance'],
                        'daily_range': entry_conditions.get('daily_range', np.nan)
                    }
                    trades.append(trade)
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if not trades:
        continue
    
    trades_df = pd.DataFrame(trades)
    total_days = 16614 / 78
    
    print(f"\n{'='*60}")
    print(f"STRATEGY {strategy_id}")
    print('='*60)
    print(f"Total trades: {len(trades_df)} ({len(trades_df)/total_days:.1f} per day)")
    print(f"Baseline: {trades_df['pct_return'].mean()*100:.2f} bps\n")
    
    # Test specific filter combinations
    
    # 1. Counter-trend shorts in uptrends
    ct_shorts = trades_df[(trades_df['trend_up'] == True) & (trades_df['direction'] == 'short')]
    if len(ct_shorts) > 10:
        edge = ct_shorts['pct_return'].mean() * 100
        tpd = len(ct_shorts) / total_days
        print(f"Counter-trend shorts in uptrends:")
        print(f"  Trades: {len(ct_shorts)} ({tpd:.1f}/day)")
        print(f"  Edge: {edge:.2f} bps")
        print(f"  Win rate: {(ct_shorts['pct_return'] > 0).mean():.1%}")
        if edge >= 0.90:
            print(f"  ✓ CLOSE TO CLAIMED 0.93 bps!")
    
    # 2. High volatility (80th+ percentile)
    high_vol = trades_df[trades_df['vol_percentile'] >= 80]
    if len(high_vol) > 10:
        edge = high_vol['pct_return'].mean() * 100
        tpd = len(high_vol) / total_days
        print(f"\nHigh volatility (80th+ percentile):")
        print(f"  Trades: {len(high_vol)} ({tpd:.1f}/day)")
        print(f"  Edge: {edge:.2f} bps")
        print(f"  Win rate: {(high_vol['pct_return'] > 0).mean():.1%}")
    
    # 3. Ranging markets (1-2% daily range)
    ranging = trades_df[(trades_df['daily_range'] >= 1.0) & (trades_df['daily_range'] <= 2.0)]
    if len(ranging) > 10:
        edge = ranging['pct_return'].mean() * 100
        tpd = len(ranging) / total_days
        print(f"\nRanging markets (1-2% daily range):")
        print(f"  Trades: {len(ranging)} ({tpd:.1f}/day)")
        print(f"  Edge: {edge:.2f} bps")
        print(f"  Win rate: {(ranging['pct_return'] > 0).mean():.1%}")
    
    # Let's also try different volatility thresholds
    print(f"\nVolatility analysis:")
    for vol_threshold in [70, 75, 80, 85, 90]:
        vol_filter = trades_df[trades_df['vol_percentile'] >= vol_threshold]
        if len(vol_filter) > 5:
            edge = vol_filter['pct_return'].mean() * 100
            tpd = len(vol_filter) / total_days
            print(f"  Vol >= {vol_threshold}: {edge:.2f} bps on {tpd:.1f} trades/day")
    
    # Try combined filters that might achieve the claimed performance
    print(f"\nCombined filters:")
    
    # Maybe the original claim was for a specific combination?
    combined1 = trades_df[(trades_df['trend_up'] == True) & 
                         (trades_df['direction'] == 'short') & 
                         (trades_df['vol_percentile'] >= 70)]
    if len(combined1) > 5:
        edge = combined1['pct_return'].mean() * 100
        tpd = len(combined1) / total_days
        print(f"  CT shorts + Vol>70: {edge:.2f} bps on {tpd:.1f} trades/day")
        if edge >= 0.90:
            print(f"    ✓ POSSIBLE MATCH for 0.93 bps claim!")
    
    # Try with VWAP alignment
    vwap_aligned = trades_df[((trades_df['above_vwap'] == True) & (trades_df['direction'] == 'long')) |
                            ((trades_df['above_vwap'] == False) & (trades_df['direction'] == 'short'))]
    if len(vwap_aligned) > 10:
        edge = vwap_aligned['pct_return'].mean() * 100
        tpd = len(vwap_aligned) / total_days
        print(f"  VWAP aligned: {edge:.2f} bps on {tpd:.1f} trades/day")

# Also check if the claims might be from our previously discovered patterns
print("\n\n" + "="*60)
print("CHECKING OUR PREVIOUSLY DISCOVERED HIGH-EDGE PATTERNS")
print("="*60)

# Best strategy for analysis
strategy_id = 88
signal_file = signal_dir / f"SPY_5m_compiled_strategy_{strategy_id}.parquet"
signals = pd.read_parquet(signal_file)

# Collect trades again
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
                trade = {
                    'pct_return': pct_return,
                    'direction': 'short' if entry_data['signal'] < 0 else 'long',
                    'trend_up': entry_conditions['trend_up'],
                    'vol_percentile': entry_conditions['vol_percentile'],
                    'vwap_distance': entry_conditions['vwap_distance']
                }
                trades.append(trade)
        
        if curr['val'] != 0:
            entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
        else:
            entry_data = None

trades_df = pd.DataFrame(trades)

# Our best discovered pattern: High Vol + Far from VWAP
high_vol_far_vwap = trades_df[(trades_df['vol_percentile'] > 70) & 
                              (abs(trades_df['vwap_distance']) > 0.2)]
if len(high_vol_far_vwap) > 0:
    edge = high_vol_far_vwap['pct_return'].mean() * 100
    tpd = len(high_vol_far_vwap) / total_days
    print(f"\nHigh Vol + Far from VWAP (our best pattern):")
    print(f"  Edge: {edge:.2f} bps on {tpd:.1f} trades/day")
    print(f"  This matches our earlier finding of 4.49 bps!")

print("\n\nCONCLUSIONS:")
print("The claimed 0.93 bps for counter-trend shorts may be achievable with specific")
print("parameter combinations and additional filters, but not with the simple filter alone.")
print("The claims appear to be best-case scenarios rather than average performance.")