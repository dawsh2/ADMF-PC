"""Analyze training data with 0.01% stop loss"""
import pandas as pd
import numpy as np
from pathlib import Path

# Training workspace path
workspace = Path("workspaces/signal_generation_a033b74d")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"

# Read the sparse signal data
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

# Convert to trades
trades = []
current_position = 0

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    new_signal = row['val']
    
    if current_position != 0 and new_signal != current_position:
        entry_idx = i - 1
        entry_row = signals_df.iloc[entry_idx]
        
        entry_price = entry_row['px']
        exit_price = row['px']
        pnl_pct = (exit_price / entry_price - 1) * current_position * 100
        bars_held = row['idx'] - entry_row['idx']
        
        trades.append({
            'entry_time': entry_row['ts'],
            'exit_time': row['ts'],
            'direction': 'long' if current_position > 0 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'bars_held': bars_held
        })
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

print("=== Training Data with Different Stop Loss Levels ===\n")

# Test different stop loss levels
stop_levels = [0.01, 0.05, 0.10, 0.15, 0.20]
trades_per_year = 699  # From previous analysis

print(f"Total trades: {len(trades_df)}")
print(f"Original avg return: {trades_df['pnl_pct'].mean():.3f}%")
print(f"Trades per year: {trades_per_year}")

print(f"\n{'Stop Loss':<12} {'Avg Return':<12} {'Trades Hit':<12} {'Net Annual (1bp)':<15}")
print("-" * 55)

for stop_pct in stop_levels:
    trades_with_stop = trades_df.copy()
    trades_hit_stop = len(trades_df[trades_df['pnl_pct'] < -stop_pct])
    trades_with_stop.loc[trades_with_stop['pnl_pct'] < -stop_pct, 'pnl_pct'] = -stop_pct
    
    avg_return = trades_with_stop['pnl_pct'].mean()
    
    # Calculate net annual with 1bp cost
    exec_cost = 0.0002
    net_return = avg_return / 100 - exec_cost
    
    if net_return > 0:
        annual_net = (1 + net_return) ** trades_per_year - 1
        annual_str = f"{annual_net*100:.1f}%"
    else:
        annual_str = "LOSS"
    
    print(f"{stop_pct:.2f}%       {avg_return:>8.3f}%    {trades_hit_stop:>10}    {annual_str:>14}")

# Detailed analysis of 0.01% stop
print(f"\n=== Detailed Analysis: 0.01% Stop Loss ===")
trades_001_stop = trades_df.copy()
trades_001_stop.loc[trades_001_stop['pnl_pct'] < -0.01, 'pnl_pct'] = -0.01

print(f"Average return: {trades_001_stop['pnl_pct'].mean():.3f}%")
print(f"Trades hitting stop: {len(trades_df[trades_df['pnl_pct'] < -0.01])}")
print(f"Percentage of trades stopped: {len(trades_df[trades_df['pnl_pct'] < -0.01])/len(trades_df)*100:.1f}%")

# By direction
for direction in ['long', 'short']:
    dir_trades = trades_df[trades_df['direction'] == direction]
    dir_trades_stop = dir_trades.copy()
    dir_trades_stop.loc[dir_trades_stop['pnl_pct'] < -0.01, 'pnl_pct'] = -0.01
    
    print(f"\n{direction.capitalize()}s:")
    print(f"  Original avg: {dir_trades['pnl_pct'].mean():.3f}%")
    print(f"  With 0.01% stop: {dir_trades_stop['pnl_pct'].mean():.3f}%")
    print(f"  Trades hit stop: {len(dir_trades[dir_trades['pnl_pct'] < -0.01])}")

# Compare stop effectiveness
print(f"\n=== Stop Loss Effectiveness ===")
print(f"0.01% stop improves avg return by: {((trades_001_stop['pnl_pct'].mean() - trades_df['pnl_pct'].mean()) / trades_df['pnl_pct'].mean() * 100):.0f}%")
print(f"0.10% stop improves avg return by: {((0.019 - trades_df['pnl_pct'].mean()) / trades_df['pnl_pct'].mean() * 100):.0f}%")

# Save for comparison
trades_df.to_csv('training_trades_no_vwap.csv', index=False)
print(f"\nSaved {len(trades_df)} trades to training_trades_no_vwap.csv for comparison")