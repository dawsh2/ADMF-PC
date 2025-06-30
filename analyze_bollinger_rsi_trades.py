"""Analyze Bollinger RSI Simple Signals trades from sparse trace data"""
import pandas as pd
import numpy as np
from pathlib import Path

# Workspace path
workspace = Path("workspaces/signal_generation_238d9851")

# Find the signal file
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"

# Read the sparse signal data
signals_df = pd.read_parquet(signal_file)
print(f"=== Bollinger RSI Simple Signals Analysis ===")
print(f"Total signal changes: {len(signals_df)}")
print(f"Date range: {signals_df['ts'].min()} to {signals_df['ts'].max()}")
print(f"\nSignal columns: {signals_df.columns.tolist()}")

# Convert sparse signals to trades
trades = []
current_position = 0

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    new_signal = row['val']
    
    # If we have a position and signal changes
    if current_position != 0 and new_signal != current_position:
        # Find entry
        entry_idx = i - 1
        entry_row = signals_df.iloc[entry_idx]
        
        # Calculate PnL
        entry_price = entry_row['px']
        exit_price = row['px']
        pnl_pct = (exit_price / entry_price - 1) * current_position * 100
        bars_held = row['idx'] - entry_row['idx']
        
        trades.append({
            'entry_time': entry_row['ts'],
            'exit_time': row['ts'],
            'entry_bar': entry_row['idx'],
            'exit_bar': row['idx'],
            'direction': 'long' if current_position > 0 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'bars_held': bars_held,
            'hour': pd.to_datetime(entry_row['ts']).hour
        })
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)
print(f"\nTotal completed trades: {len(trades_df)}")

if len(trades_df) > 0:
    # Overall performance
    print(f"\n=== Overall Performance ===")
    print(f"Average return per trade: {trades_df['pnl_pct'].mean():.3f}%")
    print(f"Win rate: {(trades_df['pnl_pct'] > 0).mean():.1%}")
    print(f"Average bars held: {trades_df['bars_held'].mean():.1f}")
    
    # Winners vs Losers
    winners = trades_df[trades_df['pnl_pct'] > 0]
    losers = trades_df[trades_df['pnl_pct'] < 0]
    
    print(f"\n=== Winners vs Losers ===")
    print(f"Winners: {len(winners)} trades, avg return: {winners['pnl_pct'].mean():.3f}%, avg bars: {winners['bars_held'].mean():.1f}")
    print(f"Losers: {len(losers)} trades, avg return: {losers['pnl_pct'].mean():.3f}%, avg bars: {losers['bars_held'].mean():.1f}")
    
    # Exit speed analysis
    print(f"\n=== Exit Speed Analysis ===")
    quick_winners = winners[winners['bars_held'] < 5]
    quick_losers = losers[losers['bars_held'] < 5]
    print(f"Winners exiting <5 bars: {len(quick_winners)} ({len(quick_winners)/len(winners) if len(winners) > 0 else 0:.1%})")
    print(f"Losers exiting <5 bars: {len(quick_losers)} ({len(quick_losers)/len(losers) if len(losers) > 0 else 0:.1%})")
    
    # Long vs Short
    longs = trades_df[trades_df['direction'] == 'long']
    shorts = trades_df[trades_df['direction'] == 'short']
    
    print(f"\n=== Long vs Short Performance ===")
    print(f"Longs: {len(longs)} trades, avg: {longs['pnl_pct'].mean():.3f}%, win rate: {(longs['pnl_pct'] > 0).mean():.1%}")
    print(f"Shorts: {len(shorts)} trades, avg: {shorts['pnl_pct'].mean():.3f}%, win rate: {(shorts['pnl_pct'] > 0).mean():.1%}")
    
    # What-if scenarios
    print(f"\n=== What-If Scenarios ===")
    
    # Stop loss at -0.1%
    trades_with_stop = trades_df.copy()
    trades_with_stop.loc[trades_with_stop['pnl_pct'] < -0.1, 'pnl_pct'] = -0.1
    print(f"With -0.1% stop loss: avg return would be {trades_with_stop['pnl_pct'].mean():.3f}%")
    
    # Minimum bars between trades
    print(f"\n=== Trade Spacing Analysis ===")
    trades_sorted = trades_df.sort_values('entry_bar')
    trades_sorted['bars_since_last_exit'] = trades_sorted['entry_bar'] - trades_sorted['exit_bar'].shift(1)
    
    for min_bars in [10, 20, 30]:
        valid_trades = trades_sorted[trades_sorted['bars_since_last_exit'] >= min_bars].dropna()
        if len(valid_trades) > 0:
            print(f"Min {min_bars} bars between trades: {len(valid_trades)} trades, "
                  f"avg: {valid_trades['pnl_pct'].mean():.3f}%, "
                  f"win rate: {(valid_trades['pnl_pct'] > 0).mean():.1%}")
    
    # Time of day analysis
    print(f"\n=== Time of Day Analysis ===")
    for hour in sorted(trades_df['hour'].unique()):
        hour_trades = trades_df[trades_df['hour'] == hour]
        if len(hour_trades) >= 5:  # Only show hours with meaningful data
            print(f"Hour {hour:02d}: {len(hour_trades)} trades, "
                  f"avg: {hour_trades['pnl_pct'].mean():.3f}%, "
                  f"win rate: {(hour_trades['pnl_pct'] > 0).mean():.1%}")