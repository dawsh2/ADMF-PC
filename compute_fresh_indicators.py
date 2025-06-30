"""Compute fresh VWAP and other indicators for proper analysis"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("Loading market data...")
market_df = pd.read_csv('data/SPY_1m.csv')
market_df['timestamp'] = pd.to_datetime(market_df['timestamp'], utc=True)

# Get test portion (last 20%)
test_start_idx = int(len(market_df) * 0.8)
print(f"Using test data from index {test_start_idx} onwards")
test_df = market_df.iloc[test_start_idx:].copy()
test_df.set_index('timestamp', inplace=True)

print(f"Test data range: {test_df.index.min()} to {test_df.index.max()}")

# Compute fresh VWAP for each trading day
test_df['date'] = test_df.index.date
test_df['time'] = test_df.index.time

# Calculate cumulative values for VWAP within each day
test_df['cum_volume'] = test_df.groupby('date')['Volume'].cumsum()
test_df['cum_pv'] = test_df.groupby('date').apply(
    lambda x: (x['Close'] * x['Volume']).cumsum()
).reset_index(level=0, drop=True)

# Calculate VWAP
test_df['VWAP_fresh'] = test_df['cum_pv'] / test_df['cum_volume']
test_df['VWAP_distance'] = ((test_df['Close'] - test_df['VWAP_fresh']) / test_df['VWAP_fresh']) * 100

# Calculate ATR properly
high_low = test_df['High'] - test_df['Low']
high_close = abs(test_df['High'] - test_df['Close'].shift(1))
low_close = abs(test_df['Low'] - test_df['Close'].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
test_df['ATR'] = true_range.rolling(14).mean()
test_df['ATR_pct'] = (test_df['ATR'] / test_df['Close']) * 100

# Calculate historical volatility (20-period)
test_df['returns'] = test_df['Close'].pct_change()
test_df['hist_vol'] = test_df['returns'].rolling(20).std() * np.sqrt(390) * 100  # Annualized

# Trend detection using multiple timeframes
test_df['SMA20'] = test_df['Close'].rolling(20).mean()
test_df['SMA50'] = test_df['Close'].rolling(50).mean()
test_df['SMA200'] = test_df['Close'].rolling(200).mean()

# More sophisticated trend detection
test_df['trend'] = 'sideways'
test_df.loc[(test_df['Close'] > test_df['SMA20']) & 
            (test_df['SMA20'] > test_df['SMA50']), 'trend'] = 'uptrend'
test_df.loc[(test_df['Close'] < test_df['SMA20']) & 
            (test_df['SMA20'] < test_df['SMA50']), 'trend'] = 'downtrend'

# Volume analysis
test_df['Volume_SMA20'] = test_df['Volume'].rolling(20).mean()
test_df['Volume_ratio'] = test_df['Volume'] / test_df['Volume_SMA20']

# Load signal data and merge with market data
workspace = Path("workspaces/signal_generation_238d9851")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'], utc=True)

print(f"\nSignal data: {len(signals_df)} changes")

# Convert signals to trades
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
        
        # Find market data
        try:
            market_data = test_df.loc[entry_time]
            
            # ATR-based volatility
            atr_pct = market_data['ATR_pct']
            hist_vol = market_data['hist_vol']
            
            # VWAP position
            vwap_dist = market_data['VWAP_distance']
            
            # Trend
            trend = market_data['trend']
            
            # Volume
            vol_ratio = market_data['Volume_ratio']
            
        except KeyError:
            # Try nearest timestamp
            nearest_idx = test_df.index.get_indexer([entry_time], method='nearest')[0]
            if abs(test_df.index[nearest_idx] - entry_time) < pd.Timedelta(minutes=5):
                market_data = test_df.iloc[nearest_idx]
                atr_pct = market_data['ATR_pct']
                hist_vol = market_data['hist_vol']
                vwap_dist = market_data['VWAP_distance']
                trend = market_data['trend']
                vol_ratio = market_data['Volume_ratio']
            else:
                atr_pct = np.nan
                hist_vol = np.nan
                vwap_dist = np.nan
                trend = 'unknown'
                vol_ratio = np.nan
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': pd.to_datetime(row['ts']),
            'direction': 'long' if current_position > 0 else 'short',
            'pnl_pct': pnl_pct,
            'bars_held': bars_held,
            'atr_pct': atr_pct,
            'hist_vol': hist_vol,
            'vwap_distance': vwap_dist,
            'trend': trend,
            'volume_ratio': vol_ratio,
            'hour': entry_time.hour
        })
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

# Remove trades with missing data
print(f"\nTotal trades: {len(trades_df)}")
trades_clean = trades_df.dropna(subset=['atr_pct', 'vwap_distance'])
print(f"Trades with complete data: {len(trades_clean)}")

# Categorize volatility using both ATR and historical vol
# Use percentiles for categorization
atr_33 = trades_clean['atr_pct'].quantile(0.33)
atr_67 = trades_clean['atr_pct'].quantile(0.67)

trades_clean['volatility'] = 'medium'
trades_clean.loc[trades_clean['atr_pct'] < atr_33, 'volatility'] = 'low'
trades_clean.loc[trades_clean['atr_pct'] > atr_67, 'volatility'] = 'high'

# VWAP position categories
trades_clean['vwap_position'] = 'near'
trades_clean.loc[trades_clean['vwap_distance'] < -0.2, 'vwap_position'] = 'far_below'
trades_clean.loc[trades_clean['vwap_distance'] < -0.05, 'vwap_position'] = 'below'
trades_clean.loc[trades_clean['vwap_distance'] > 0.2, 'vwap_position'] = 'far_above'
trades_clean.loc[trades_clean['vwap_distance'] > 0.05, 'vwap_position'] = 'above'

# Overall Performance
print(f"\n=== Overall Performance ===")
print(f"Average return: {trades_clean['pnl_pct'].mean():.3f}%")
print(f"Win rate: {(trades_clean['pnl_pct'] > 0).mean():.1%}")

# Performance by Volatility
print(f"\n=== Performance by Volatility Regime ===")
for vol in ['low', 'medium', 'high']:
    vol_trades = trades_clean[trades_clean['volatility'] == vol]
    if len(vol_trades) > 0:
        print(f"{vol.capitalize()} volatility: {len(vol_trades)} trades, "
              f"avg: {vol_trades['pnl_pct'].mean():.3f}%, "
              f"win rate: {(vol_trades['pnl_pct'] > 0).mean():.1%}")

# Performance by VWAP Position
print(f"\n=== Performance by VWAP Position ===")
for vwap_pos in ['far_below', 'below', 'near', 'above', 'far_above']:
    vwap_trades = trades_clean[trades_clean['vwap_position'] == vwap_pos]
    if len(vwap_trades) > 0:
        print(f"{vwap_pos.replace('_', ' ').capitalize()}: {len(vwap_trades)} trades, "
              f"avg: {vwap_trades['pnl_pct'].mean():.3f}%, "
              f"win rate: {(vwap_trades['pnl_pct'] > 0).mean():.1%}")

# Performance by Trend
print(f"\n=== Performance by Trend Regime ===")
for trend in ['uptrend', 'downtrend', 'sideways']:
    trend_trades = trades_clean[trades_clean['trend'] == trend]
    if len(trend_trades) > 0:
        print(f"{trend.capitalize()}: {len(trend_trades)} trades, "
              f"avg: {trend_trades['pnl_pct'].mean():.3f}%, "
              f"win rate: {(trend_trades['pnl_pct'] > 0).mean():.1%}")

# Best combinations
print(f"\n=== Best Market Condition Combinations ===")
condition_groups = trades_clean.groupby(['volatility', 'trend', 'vwap_position'])
results = []
for (vol, trend, vwap), group in condition_groups:
    if len(group) >= 3:
        avg_return = group['pnl_pct'].mean()
        win_rate = (group['pnl_pct'] > 0).mean()
        results.append({
            'conditions': f"{vol} vol + {trend} + {vwap}",
            'trades': len(group),
            'avg_return': avg_return,
            'win_rate': win_rate
        })

results_df = pd.DataFrame(results).sort_values('avg_return', ascending=False)
for _, row in results_df.head(10).iterrows():
    print(f"{row['conditions']}: {row['trades']} trades, "
          f"avg: {row['avg_return']:.3f}%, win rate: {row['win_rate']:.1%}")

# Save for further analysis
trades_clean.to_csv('bb_trades_fresh_indicators.csv', index=False)
print(f"\nSaved {len(trades_clean)} trades with fresh indicators to bb_trades_fresh_indicators.csv")