"""Replicate the Bollinger RSI analysis with proper market conditions"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load market data
print("Loading market data...")
market_df = pd.read_csv('data/SPY_1m.csv')
market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
market_df.set_index('timestamp', inplace=True)

# Calculate indicators on full data
print("Calculating market indicators...")

# ATR (14 period)
high_low = market_df['High'] - market_df['Low']
high_close = abs(market_df['High'] - market_df['Close'].shift(1))
low_close = abs(market_df['Low'] - market_df['Close'].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
market_df['ATR'] = true_range.rolling(14).mean()
market_df['ATR_pct'] = (market_df['ATR'] / market_df['Close']) * 100

# VWAP distance
market_df['VWAP'] = market_df['vwap']  # Already in data
market_df['VWAP_distance'] = ((market_df['Close'] - market_df['VWAP']) / market_df['VWAP']) * 100

# Simple trend detection (20 period)
market_df['SMA20'] = market_df['Close'].rolling(20).mean()
market_df['SMA50'] = market_df['Close'].rolling(50).mean()
market_df['trend'] = 'sideways'
market_df.loc[market_df['SMA20'] > market_df['SMA50'] * 1.002, 'trend'] = 'uptrend'
market_df.loc[market_df['SMA20'] < market_df['SMA50'] * 0.998, 'trend'] = 'downtrend'

# Volume ratio
market_df['Volume_SMA'] = market_df['Volume'].rolling(20).mean()
market_df['Volume_ratio'] = market_df['Volume'] / market_df['Volume_SMA']

# Load signal data
workspace = Path("workspaces/signal_generation_238d9851")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

print(f"\nSignal data range: {signals_df['ts'].min()} to {signals_df['ts'].max()}")
print(f"Total signals: {len(signals_df)}")

# Convert to trades
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
        
        # Find closest market data
        try:
            market_at_entry = market_df.loc[market_df.index.get_indexer([entry_time], method='nearest')[0]]
            
            # Volatility regime based on ATR
            atr_pct = market_at_entry['ATR_pct']
            if atr_pct < 0.5:
                vol_regime = 'low'
            elif atr_pct > 1.0:
                vol_regime = 'high'
            else:
                vol_regime = 'medium'
            
            # VWAP position
            vwap_dist = market_at_entry['VWAP_distance']
            if vwap_dist < -0.2:
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
            
        except:
            vol_regime = 'unknown'
            vwap_position = 'unknown'
            trend = 'unknown'
            volume_ratio = 1.0
            atr_pct = 0
            vwap_dist = 0
        
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
condition_groups = trades_df.groupby(['volatility', 'trend', 'vwap_position'])
for (vol, trend, vwap), group in condition_groups:
    if len(group) >= 3:  # At least 3 trades
        avg_return = group['pnl_pct'].mean()
        win_rate = (group['pnl_pct'] > 0).mean()
        print(f"{vol} vol + {trend} + {vwap}: "
              f"{len(group)} trades, avg: {avg_return:.3f}%, win rate: {win_rate:.1%}")

# Save detailed results
trades_df.to_csv('bb_trades_with_conditions.csv', index=False)
print(f"\nDetailed results saved to bb_trades_with_conditions.csv")