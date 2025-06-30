"""Verify Keltner Bands calculations to understand discrepancy"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_d5807cc2")
signal_file = workspace / "traces/SPY_1m/signals/keltner_bands/SPY_compiled_strategy_0.parquet"

print("=== VERIFYING KELTNER BANDS CALCULATIONS ===\n")

# Load signals
signals = pd.read_parquet(signal_file)

# Load SPY 1m data
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})
spy_subset = spy_1m.iloc[:81787].copy()

# Calculate indicators matching their filters
spy_subset['returns'] = spy_subset['close'].pct_change()
spy_subset['volatility_20'] = spy_subset['returns'].rolling(20).std() * np.sqrt(390) * 100
spy_subset['vol_percentile'] = spy_subset['volatility_20'].rolling(window=390*5).rank(pct=True) * 100

# RSI
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

spy_subset['rsi'] = calculate_rsi(spy_subset['close'])

# Volume ratio
spy_subset['volume_sma_20'] = spy_subset['volume'].rolling(20).mean()
spy_subset['volume_ratio'] = spy_subset['volume'] / spy_subset['volume_sma_20']

# VWAP
spy_subset['date'] = spy_subset['timestamp'].dt.date
spy_subset['typical_price'] = (spy_subset['high'] + spy_subset['low'] + spy_subset['close']) / 3
spy_subset['pv'] = spy_subset['typical_price'] * spy_subset['volume']
spy_subset['cum_pv'] = spy_subset.groupby('date')['pv'].cumsum()
spy_subset['cum_volume'] = spy_subset.groupby('date')['volume'].cumsum()
spy_subset['vwap'] = spy_subset['cum_pv'] / spy_subset['cum_volume']
spy_subset['vwap_distance'] = (spy_subset['close'] - spy_subset['vwap']) / spy_subset['vwap'] * 100

# Test their top filter: HighVol(>90%) + HighVolume(>2x) + LowRSI(<50)
filter_mask = (spy_subset['vol_percentile'] > 90) & (spy_subset['volume_ratio'] > 2) & (spy_subset['rsi'] < 50)

print(f"Filter: HighVol(>90%) + HighVolume(>2x) + LowRSI(<50)")
print(f"Bars passing filter: {filter_mask.sum()} out of {len(spy_subset)}")

# Collect trades with this filter
trades = []
entry_data = None

for i in range(len(signals)):
    curr = signals.iloc[i]
    
    if entry_data is None and curr['val'] != 0:
        entry_idx = curr['idx']
        if entry_idx < len(spy_subset) and filter_mask.iloc[entry_idx]:
            entry_data = {
                'idx': entry_idx,
                'price': curr['px'],
                'signal': curr['val'],
                'entry_bar': spy_subset.iloc[entry_idx]
            }
    
    elif entry_data is not None:
        # Natural exit
        if curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal']):
            exit_idx = curr['idx']
            if exit_idx < len(spy_subset):
                # Calculate return
                exit_price = curr['px']
                pct_return = (exit_price / entry_data['price'] - 1) * entry_data['signal'] * 100
                
                # Alternative calculation (in case they use different method)
                alt_return = (exit_price - entry_data['price']) / entry_data['price'] * entry_data['signal'] * 100
                
                trades.append({
                    'entry_idx': entry_data['idx'],
                    'exit_idx': exit_idx,
                    'duration': exit_idx - entry_data['idx'],
                    'direction': 'long' if entry_data['signal'] > 0 else 'short',
                    'pct_return': pct_return,
                    'alt_return': alt_return,
                    'entry_price': entry_data['price'],
                    'exit_price': exit_price,
                    'vol_pct': entry_data['entry_bar']['vol_percentile'],
                    'volume_ratio': entry_data['entry_bar']['volume_ratio'],
                    'rsi': entry_data['entry_bar']['rsi']
                })
            
            # Check if next signal passes filter
            if curr['val'] != 0 and curr['idx'] < len(spy_subset) and filter_mask.iloc[curr['idx']]:
                entry_data = {
                    'idx': curr['idx'],
                    'price': curr['px'],
                    'signal': curr['val'],
                    'entry_bar': spy_subset.iloc[curr['idx']]
                }
            else:
                entry_data = None

trades_df = pd.DataFrame(trades)
total_days = 81787 / 390

print(f"\nTrades found: {len(trades_df)}")
if len(trades_df) > 0:
    print(f"Trades per day: {len(trades_df)/total_days:.2f}")
    print(f"\nReturn calculations:")
    print(f"Standard method: {trades_df['pct_return'].mean():.4f}% = {trades_df['pct_return'].mean() * 100:.2f} bps")
    print(f"Alternative method: {trades_df['alt_return'].mean():.4f}% = {trades_df['alt_return'].mean() * 100:.2f} bps")
    print(f"Win rate: {(trades_df['pct_return'] > 0).mean():.1%}")
    print(f"Average duration: {trades_df['duration'].mean():.1f} bars")
    
    # Check if there's a unit confusion
    print(f"\nIf they meant % not bps:")
    print(f"Their 7.14 'bps' = 7.14% = 714 actual basis points")
    print(f"Our result: {trades_df['pct_return'].mean():.4f}% = {trades_df['pct_return'].mean() * 100:.2f} bps")
    
    # Distribution of returns
    print(f"\nReturn distribution:")
    print(f"Min: {trades_df['pct_return'].min():.2f}%")
    print(f"25%: {trades_df['pct_return'].quantile(0.25):.2f}%")
    print(f"50%: {trades_df['pct_return'].median():.2f}%")
    print(f"75%: {trades_df['pct_return'].quantile(0.75):.2f}%")
    print(f"Max: {trades_df['pct_return'].max():.2f}%")
    
    # Sample trades
    print(f"\nSample trades:")
    for i, trade in trades_df.head(5).iterrows():
        print(f"Trade {i+1}: {trade['direction']}, "
              f"return={trade['pct_return']:.2f}%, duration={trade['duration']} bars")

# Test without stops vs with 0.3% stop
print("\n\n=== COMPARING NO STOP VS 0.3% STOP ===")

def collect_trades_with_stop(filter_mask, stop_pct=None):
    trades = []
    entry_data = None
    
    for i in range(len(signals)):
        curr = signals.iloc[i]
        
        if entry_data is None and curr['val'] != 0:
            entry_idx = curr['idx']
            if entry_idx < len(spy_subset) and filter_mask.iloc[entry_idx]:
                entry_data = {
                    'idx': entry_idx,
                    'price': curr['px'],
                    'signal': curr['val']
                }
        
        elif entry_data is not None:
            stopped_out = False
            exit_price = curr['px']
            exit_idx = curr['idx']
            
            # Check for stop if enabled
            if stop_pct is not None:
                for check_idx in range(entry_data['idx'] + 1, min(curr['idx'] + 1, len(spy_subset))):
                    check_bar = spy_subset.iloc[check_idx]
                    
                    if entry_data['signal'] > 0:  # Long
                        stop_level = entry_data['price'] * (1 - stop_pct)
                        if check_bar['low'] <= stop_level:
                            stopped_out = True
                            exit_price = stop_level
                            exit_idx = check_idx
                            break
                    else:  # Short
                        stop_level = entry_data['price'] * (1 + stop_pct)
                        if check_bar['high'] >= stop_level:
                            stopped_out = True
                            exit_price = stop_level
                            exit_idx = check_idx
                            break
            
            # Natural exit or stop
            if stopped_out or curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal']):
                pct_return = (exit_price / entry_data['price'] - 1) * entry_data['signal'] * 100
                trades.append({
                    'pct_return': pct_return,
                    'stopped_out': stopped_out
                })
                
                # Next entry
                if curr['val'] != 0 and not stopped_out and curr['idx'] < len(spy_subset) and filter_mask.iloc[curr['idx']]:
                    entry_data = {
                        'idx': curr['idx'],
                        'price': curr['px'],
                        'signal': curr['val']
                    }
                else:
                    entry_data = None
    
    return pd.DataFrame(trades)

# Test with best filter
best_filter = (spy_subset['vol_percentile'] > 90) & (spy_subset['volume_ratio'] > 2) & (spy_subset['rsi'] < 50)

no_stop_trades = collect_trades_with_stop(best_filter, stop_pct=None)
with_stop_trades = collect_trades_with_stop(best_filter, stop_pct=0.003)

print(f"\nBest filter: HighVol(>90%) + HighVolume(>2x) + LowRSI(<50)")
print(f"\nNo stop:")
if len(no_stop_trades) > 0:
    print(f"  Trades: {len(no_stop_trades)}")
    print(f"  Average: {no_stop_trades['pct_return'].mean():.4f}% = {no_stop_trades['pct_return'].mean() * 100:.2f} bps")
    print(f"  Win rate: {(no_stop_trades['pct_return'] > 0).mean():.1%}")

print(f"\nWith 0.3% stop:")
if len(with_stop_trades) > 0:
    print(f"  Trades: {len(with_stop_trades)}")
    print(f"  Average: {with_stop_trades['pct_return'].mean():.4f}% = {with_stop_trades['pct_return'].mean() * 100:.2f} bps")
    print(f"  Win rate: {(with_stop_trades['pct_return'] > 0).mean():.1%}")
    print(f"  Stopped: {with_stop_trades['stopped_out'].mean():.1%}")