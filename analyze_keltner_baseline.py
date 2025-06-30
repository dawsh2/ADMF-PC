"""Analyze Keltner baseline performance"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load signals
signal_file = Path("workspaces/signal_generation_1e32d562/traces/SPY_1m/signals/mean_reversion/SPY_keltner_baseline.parquet")
signals_df = pd.read_parquet(signal_file)

print("Loaded signal data:")
print(f"Total signal changes: {len(signals_df)}")
print(f"Date range: {signals_df['ts'].min()} to {signals_df['ts'].max()}")

# Convert timestamp and create proper datetime index
signals_df['timestamp'] = pd.to_datetime(signals_df['ts'])
signals_df = signals_df.sort_values('timestamp')

# Load SPY data
spy_data = pd.read_csv("./data/SPY.csv")
spy_data['timestamp'] = pd.to_datetime(spy_data['timestamp'], utc=True)
# Rename columns to lowercase
spy_data.columns = spy_data.columns.str.lower()

# Ensure both have UTC timezone
signals_df['timestamp'] = signals_df['timestamp'].dt.tz_convert('UTC')

# Merge signals with price data
merged = signals_df.merge(
    spy_data[['timestamp', 'close']], 
    on='timestamp', 
    how='left',
    suffixes=('_signal', '_price')
)

print(f"\nMerged {len(merged)} signal changes with price data")
print(f"Missing prices: {merged['close'].isna().sum()}")

# Simple backtest
positions = []
current_position = None

for idx, row in merged.iterrows():
    signal = row['val']
    price = row['close'] if not pd.isna(row['close']) else row['px']  # Use signal price if market price missing
    ts = row['timestamp']
    
    if signal != 0 and current_position is None:
        # Enter position
        current_position = {
            'entry_time': ts,
            'entry_price': price,
            'direction': 1 if signal > 0 else -1
        }
    elif signal == 0 and current_position is not None:
        # Exit position
        ret = (price - current_position['entry_price']) / current_position['entry_price']
        if current_position['direction'] < 0:
            ret = -ret
            
        current_position['exit_time'] = ts
        current_position['exit_price'] = price
        current_position['return'] = ret
        current_position['bps'] = ret * 10000
        current_position['duration_min'] = (ts - current_position['entry_time']).total_seconds() / 60
        
        positions.append(current_position)
        current_position = None
    elif signal != 0 and current_position is not None and np.sign(signal) != current_position['direction']:
        # Close and reverse
        ret = (price - current_position['entry_price']) / current_position['entry_price']
        if current_position['direction'] < 0:
            ret = -ret
            
        current_position['exit_time'] = ts
        current_position['exit_price'] = price
        current_position['return'] = ret
        current_position['bps'] = ret * 10000
        current_position['duration_min'] = (ts - current_position['entry_time']).total_seconds() / 60
        
        positions.append(current_position)
        
        # Open new position
        current_position = {
            'entry_time': ts,
            'entry_price': price,
            'direction': 1 if signal > 0 else -1
        }

if positions:
    trades_df = pd.DataFrame(positions)
    
    # Calculate metrics
    total_trades = len(trades_df)
    avg_bps = trades_df['bps'].mean()
    win_rate = (trades_df['bps'] > 0).mean() * 100
    
    # Trading days
    trading_days = (signals_df['timestamp'].max() - signals_df['timestamp'].min()).days
    trades_per_day = total_trades / trading_days if trading_days > 0 else 0
    
    print("\n=== BASELINE KELTNER BANDS PERFORMANCE ===")
    print(f"Total trades: {total_trades}")
    print(f"Average bps per trade: {avg_bps:.2f}")
    print(f"Trades per day: {trades_per_day:.2f}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Avg duration: {trades_df['duration_min'].mean():.1f} minutes")
    
    # After costs
    cost_bps = 1.0
    net_bps = avg_bps - cost_bps
    annual_return = net_bps * trades_per_day * 252 / 100
    
    print(f"\nAfter {cost_bps} bp cost:")
    print(f"Net bps: {net_bps:.2f}")
    print(f"Annual return: {annual_return:.1f}%")
    
    # Distribution
    print(f"\nReturn distribution:")
    print(f"Best trade: {trades_df['bps'].max():.2f} bps")
    print(f"Worst trade: {trades_df['bps'].min():.2f} bps")
    print(f"Std dev: {trades_df['bps'].std():.2f} bps")
    
    # By direction
    longs = trades_df[trades_df['direction'] > 0]
    shorts = trades_df[trades_df['direction'] < 0]
    
    print(f"\nLongs: {len(longs)} trades, {longs['bps'].mean():.2f} bps avg")
    print(f"Shorts: {len(shorts)} trades, {shorts['bps'].mean():.2f} bps avg")
else:
    print("\nNo completed trades found")