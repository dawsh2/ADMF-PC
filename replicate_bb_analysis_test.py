"""Replicate the Bollinger RSI analysis with proper market conditions on test data"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load market data
print("Loading market data...")
market_df = pd.read_csv('data/SPY_1m.csv')
market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])

# Get test portion (last 20%)
test_start_idx = int(len(market_df) * 0.8)
print(f"Using test data from index {test_start_idx} onwards")
test_market_df = market_df.iloc[test_start_idx:].copy()
# Parse timestamps and convert to UTC
test_market_df['timestamp'] = pd.to_datetime(test_market_df['timestamp'], utc=True)
test_market_df.set_index('timestamp', inplace=True)

print(f"Test data range: {test_market_df.index.min()} to {test_market_df.index.max()}")

# Calculate indicators on test data
print("Calculating market indicators...")

# ATR (14 period)
high_low = test_market_df['High'] - test_market_df['Low']
high_close = abs(test_market_df['High'] - test_market_df['Close'].shift(1))
low_close = abs(test_market_df['Low'] - test_market_df['Close'].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
test_market_df['ATR'] = true_range.rolling(14).mean()
test_market_df['ATR_pct'] = (test_market_df['ATR'] / test_market_df['Close']) * 100

# VWAP distance
test_market_df['VWAP'] = test_market_df['vwap']
test_market_df['VWAP_distance'] = ((test_market_df['Close'] - test_market_df['VWAP']) / test_market_df['VWAP']) * 100

# Simple trend detection
test_market_df['SMA20'] = test_market_df['Close'].rolling(20).mean()
test_market_df['SMA50'] = test_market_df['Close'].rolling(50).mean()
test_market_df['trend'] = 'sideways'
test_market_df.loc[test_market_df['SMA20'] > test_market_df['SMA50'] * 1.002, 'trend'] = 'uptrend'
test_market_df.loc[test_market_df['SMA20'] < test_market_df['SMA50'] * 0.998, 'trend'] = 'downtrend'

# Volume ratio
test_market_df['Volume_SMA'] = test_market_df['Volume'].rolling(20).mean()
test_market_df['Volume_ratio'] = test_market_df['Volume'] / test_market_df['Volume_SMA']

# Load signal data
workspace = Path("workspaces/signal_generation_238d9851")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

print(f"\nSignal data range: {signals_df['ts'].min()} to {signals_df['ts'].max()}")
print(f"Total signals: {len(signals_df)}")

# Convert to trades with market conditions
trades = []
current_position = 0

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    new_signal = row['val']
    
    if current_position != 0 and new_signal != current_position:
        entry_idx = i - 1
        entry_row = signals_df.iloc[entry_idx]
        
        # Calculate trade metrics
        entry_price = entry_row['px']
        exit_price = row['px']
        pnl_pct = (exit_price / entry_price - 1) * current_position * 100
        bars_held = row['idx'] - entry_row['idx']
        
        # Get market conditions at entry
        entry_time = pd.to_datetime(entry_row['ts'])
        
        # Find market data at entry time
        # Both should now be in UTC
        nearest_idx = test_market_df.index.get_indexer([entry_time], method='nearest')[0]
        time_diff = abs(test_market_df.index[nearest_idx] - entry_time)
        
        if time_diff < pd.Timedelta(minutes=5):  # Within 5 minutes
            market_at_entry = test_market_df.iloc[nearest_idx]
            
            # Volatility regime based on ATR
            atr_pct = market_at_entry['ATR_pct']
            if pd.isna(atr_pct):
                vol_regime = 'unknown'
            elif atr_pct < 0.5:
                vol_regime = 'low'
            elif atr_pct > 1.0:
                vol_regime = 'high'
            else:
                vol_regime = 'medium'
            
            # VWAP position
            vwap_dist = market_at_entry['VWAP_distance']
            if pd.isna(vwap_dist):
                vwap_position = 'unknown'
            elif vwap_dist < -0.2:
                vwap_position = 'far_below'
            elif vwap_dist < -0.05:
                vwap_position = 'below'
            elif vwap_dist > 0.2:
                vwap_position = 'far_above'
            elif vwap_dist > 0.05:
                vwap_position = 'above'
            else:
                vwap_position = 'near'
            
            trend = market_at_entry['trend']
            volume_ratio = market_at_entry['Volume_ratio']
            
        else:
            vol_regime = 'unknown'
            vwap_position = 'unknown'
            trend = 'unknown'
            volume_ratio = 1.0
            atr_pct = np.nan
            vwap_dist = np.nan
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': pd.to_datetime(row['ts']),
            'direction': 'long' if current_position > 0 else 'short',
            'pnl_pct': pnl_pct,
            'bars_held': bars_held,
            'volatility': vol_regime,
            'atr_pct': atr_pct,
            'vwap_position': vwap_position,
            'vwap_distance': vwap_dist,
            'trend': trend,
            'volume_ratio': volume_ratio,
            'hour': entry_time.hour
        })
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

# Overall Performance
print(f"\n=== Overall Performance ===")
print(f"Total trades: {len(trades_df)}")
print(f"Average return: {trades_df['pnl_pct'].mean():.3f}%")
print(f"Win rate: {(trades_df['pnl_pct'] > 0).mean():.1%}")

# Performance by Volatility
print(f"\n=== Performance by Volatility Regime ===")
vol_counts = trades_df['volatility'].value_counts()
print(f"Volatility distribution: {vol_counts.to_dict()}")

for vol in ['low', 'medium', 'high']:
    vol_trades = trades_df[trades_df['volatility'] == vol]
    if len(vol_trades) > 0:
        print(f"{vol.capitalize()} volatility: {len(vol_trades)} trades, "
              f"avg: {vol_trades['pnl_pct'].mean():.3f}%, "
              f"win rate: {(vol_trades['pnl_pct'] > 0).mean():.1%}")

# Direction by Volatility
print(f"\n=== Direction Performance by Volatility ===")
for vol in ['low', 'medium', 'high']:
    vol_trades = trades_df[trades_df['volatility'] == vol]
    if len(vol_trades) > 0:
        # Longs
        vol_longs = vol_trades[vol_trades['direction'] == 'long']
        if len(vol_longs) > 0:
            print(f"{vol.capitalize()} vol - Longs: {len(vol_longs)} trades, "
                  f"avg: {vol_longs['pnl_pct'].mean():.3f}%, "
                  f"win rate: {(vol_longs['pnl_pct'] > 0).mean():.1%}")
        # Shorts
        vol_shorts = vol_trades[vol_trades['direction'] == 'short']
        if len(vol_shorts) > 0:
            print(f"{vol.capitalize()} vol - Shorts: {len(vol_shorts)} trades, "
                  f"avg: {vol_shorts['pnl_pct'].mean():.3f}%, "
                  f"win rate: {(vol_shorts['pnl_pct'] > 0).mean():.1%}")

# Performance by Trend
print(f"\n=== Performance by Trend Regime ===")
for trend in ['uptrend', 'downtrend', 'sideways']:
    trend_trades = trades_df[trades_df['trend'] == trend]
    if len(trend_trades) > 0:
        print(f"{trend.capitalize()}: {len(trend_trades)} trades, "
              f"avg: {trend_trades['pnl_pct'].mean():.3f}%, "
              f"win rate: {(trend_trades['pnl_pct'] > 0).mean():.1%}")

# Performance by VWAP Position
print(f"\n=== Performance by VWAP Position ===")
for vwap_pos in ['far_below', 'below', 'near', 'above', 'far_above']:
    vwap_trades = trades_df[trades_df['vwap_position'] == vwap_pos]
    if len(vwap_trades) > 0:
        print(f"{vwap_pos.replace('_', ' ').capitalize()}: {len(vwap_trades)} trades, "
              f"avg: {vwap_trades['pnl_pct'].mean():.3f}%, "
              f"win rate: {(vwap_trades['pnl_pct'] > 0).mean():.1%}")

# Best market conditions
print(f"\n=== Best Market Condition Combinations ===")
# Group by all conditions
valid_trades = trades_df[trades_df['volatility'] != 'unknown']
if len(valid_trades) > 0:
    condition_groups = valid_trades.groupby(['volatility', 'trend', 'vwap_position'])
    results = []
    for (vol, trend, vwap), group in condition_groups:
        if len(group) >= 3:  # At least 3 trades
            avg_return = group['pnl_pct'].mean()
            win_rate = (group['pnl_pct'] > 0).mean()
            results.append({
                'conditions': f"{vol} vol + {trend} + {vwap}",
                'trades': len(group),
                'avg_return': avg_return,
                'win_rate': win_rate
            })
    
    # Sort by average return
    results_df = pd.DataFrame(results).sort_values('avg_return', ascending=False)
    for _, row in results_df.head(10).iterrows():
        print(f"{row['conditions']}: {row['trades']} trades, "
              f"avg: {row['avg_return']:.3f}%, win rate: {row['win_rate']:.1%}")

# Check ATR statistics
print(f"\n=== ATR Statistics ===")
atr_values = trades_df['atr_pct'].dropna()
if len(atr_values) > 0:
    print(f"Mean ATR%: {atr_values.mean():.3f}%")
    print(f"ATR% percentiles: 25th={atr_values.quantile(0.25):.3f}%, "
          f"50th={atr_values.quantile(0.5):.3f}%, "
          f"75th={atr_values.quantile(0.75):.3f}%")

# Save detailed results
trades_df.to_csv('bb_trades_with_conditions_test.csv', index=False)
print(f"\nDetailed results saved to bb_trades_with_conditions_test.csv")