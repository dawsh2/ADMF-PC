"""Analyze trades from the Bollinger RSI Simple Signals workspace"""
import pandas as pd
import numpy as np
import duckdb

# Connect to the analytics database
db_path = "workspaces/signal_generation_238d9851/analytics.duckdb"
conn = duckdb.connect(db_path, read_only=True)

# Get all signals
signals_df = conn.execute("""
    SELECT * FROM signals 
    WHERE strategy_type = 'bollinger_rsi_simple_signals'
    ORDER BY timestamp
""").df()

print(f"=== Bollinger RSI Simple Signals Analysis ===")
print(f"Total signals: {len(signals_df)}")
print(f"Date range: {signals_df['timestamp'].min()} to {signals_df['timestamp'].max()}")

# Convert signal changes to trades
trades = []
position = 0
entry_bar = None
entry_price = None
entry_time = None

for idx, row in signals_df.iterrows():
    signal = row['signal']
    
    # Entry logic
    if position == 0 and signal != 0:
        position = signal
        entry_bar = row['bar_index']
        entry_price = row['close']
        entry_time = row['timestamp']
    
    # Exit logic (signal changes or goes to 0)
    elif position != 0 and (signal != position or signal == 0):
        exit_bar = row['bar_index']
        exit_price = row['close']
        exit_time = row['timestamp']
        
        # Calculate trade metrics
        pnl_pct = (exit_price / entry_price - 1) * position * 100
        bars_held = exit_bar - entry_bar
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': 'long' if position > 0 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'bars_held': bars_held,
            'hour': pd.to_datetime(entry_time).hour
        })
        
        # Reset position
        position = signal if signal != 0 else 0
        if position != 0:
            entry_bar = row['bar_index']
            entry_price = row['close']
            entry_time = row['timestamp']

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
    
    # Quick exit analysis
    print(f"\n=== Exit Speed Analysis ===")
    quick_winners = winners[winners['bars_held'] < 5]
    quick_losers = losers[losers['bars_held'] < 5]
    print(f"Winners exiting <5 bars: {len(quick_winners)} ({len(quick_winners)/len(winners):.1%})")
    print(f"Losers exiting <5 bars: {len(quick_losers)} ({len(quick_losers)/len(losers):.1%})")
    
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
    # Calculate time between trade exits and next entries
    trades_df['next_entry_time'] = trades_df['entry_time'].shift(-1)
    trades_df['bars_to_next'] = (trades_df['next_entry_time'] - trades_df['exit_time']).dt.total_seconds() / 60  # Convert to minutes
    
    for min_bars in [10, 20, 30]:
        valid_trades = trades_df[trades_df['bars_to_next'] >= min_bars].dropna()
        if len(valid_trades) > 0:
            print(f"Min {min_bars} bars between trades: {len(valid_trades)} trades, "
                  f"avg: {valid_trades['pnl_pct'].mean():.3f}%, "
                  f"win rate: {(valid_trades['pnl_pct'] > 0).mean():.1%}")

conn.close()