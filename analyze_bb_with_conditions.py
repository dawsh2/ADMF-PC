"""Analyze Bollinger RSI trades with market conditions"""
import pandas as pd
import numpy as np
from pathlib import Path

# Workspace path
workspace = Path("workspaces/signal_generation_238d9851")

# Find the signal file
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"

# Read the sparse signal data
signals_df = pd.read_parquet(signal_file)

# We need market data to calculate conditions - let's get it from the prices in signals
# First, let's see what data we have
print("Signal data structure:")
print(signals_df.head())
print(f"\nTotal signal changes: {len(signals_df)}")

# Convert sparse signals to trades with more detail
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
            'hour': pd.to_datetime(entry_row['ts']).hour,
            'entry_idx': entry_idx  # Store index for market condition analysis
        })
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

# Basic performance metrics
print(f"\n=== Overall Performance ===")
print(f"Total trades: {len(trades_df)}")
print(f"Average return: {trades_df['pnl_pct'].mean():.3f}%")
print(f"Win rate: {(trades_df['pnl_pct'] > 0).mean():.1%}")
print(f"Avg bars held: {trades_df['bars_held'].mean():.1f}")

# Winners vs Losers holding period
winners = trades_df[trades_df['pnl_pct'] > 0]
losers = trades_df[trades_df['pnl_pct'] < 0]
print(f"\n=== Holding Period Analysis ===")
print(f"Winners: {len(winners)} trades, avg bars: {winners['bars_held'].mean():.1f}")
print(f"Losers: {len(losers)} trades, avg bars: {losers['bars_held'].mean():.1f}")
print(f"Ratio: {losers['bars_held'].mean() / winners['bars_held'].mean():.1f}x longer for losers")

# Direction analysis
longs = trades_df[trades_df['direction'] == 'long']
shorts = trades_df[trades_df['direction'] == 'short']
print(f"\n=== Direction Analysis ===")
print(f"Longs: {len(longs)} trades, avg: {longs['pnl_pct'].mean():.3f}%, win rate: {(longs['pnl_pct'] > 0).mean():.1%}")
print(f"Shorts: {len(shorts)} trades, avg: {shorts['pnl_pct'].mean():.3f}%, win rate: {(shorts['pnl_pct'] > 0).mean():.1%}")

# Now let's estimate market conditions from price movements
# We'll use the signal data to estimate volatility
print(f"\n=== Volatility Analysis (Estimated) ===")

# Calculate price changes between signals for volatility proxy
price_changes = []
for i in range(1, len(signals_df)):
    prev_price = signals_df.iloc[i-1]['px']
    curr_price = signals_df.iloc[i]['px']
    bars_between = signals_df.iloc[i]['idx'] - signals_df.iloc[i-1]['idx']
    if bars_between > 0:
        pct_change = abs((curr_price / prev_price - 1) * 100)
        price_changes.append({
            'idx': i,
            'pct_change': pct_change,
            'bars': bars_between,
            'pct_per_bar': pct_change / bars_between
        })

changes_df = pd.DataFrame(price_changes)

# Categorize trades by estimated volatility
for idx, trade in trades_df.iterrows():
    # Find volatility around entry
    entry_idx = trade['entry_idx']
    nearby_changes = changes_df[(changes_df['idx'] >= entry_idx - 5) & (changes_df['idx'] <= entry_idx + 5)]
    
    if len(nearby_changes) > 0:
        avg_vol = nearby_changes['pct_per_bar'].mean()
        # Categorize volatility
        if avg_vol < 0.01:  # Less than 0.01% per bar
            vol_category = 'low'
        elif avg_vol > 0.02:  # More than 0.02% per bar
            vol_category = 'high'
        else:
            vol_category = 'medium'
    else:
        vol_category = 'unknown'
    
    trades_df.loc[idx, 'volatility'] = vol_category

# Analyze by volatility
print("\nPerformance by volatility regime:")
for vol in ['low', 'medium', 'high']:
    vol_trades = trades_df[trades_df['volatility'] == vol]
    if len(vol_trades) > 0:
        print(f"{vol.capitalize()} volatility: {len(vol_trades)} trades, "
              f"avg: {vol_trades['pnl_pct'].mean():.3f}%, "
              f"win rate: {(vol_trades['pnl_pct'] > 0).mean():.1%}")

# Direction by volatility
print("\n=== Direction Performance by Volatility ===")
for vol in ['low', 'medium', 'high']:
    vol_trades = trades_df[trades_df['volatility'] == vol]
    if len(vol_trades) > 0:
        vol_longs = vol_trades[vol_trades['direction'] == 'long']
        vol_shorts = vol_trades[vol_trades['direction'] == 'short']
        if len(vol_longs) > 0:
            print(f"{vol.capitalize()} vol - Longs: {len(vol_longs)} trades, "
                  f"avg: {vol_longs['pnl_pct'].mean():.3f}%, "
                  f"win rate: {(vol_longs['pnl_pct'] > 0).mean():.1%}")
        if len(vol_shorts) > 0:
            print(f"{vol.capitalize()} vol - Shorts: {len(vol_shorts)} trades, "
                  f"avg: {vol_shorts['pnl_pct'].mean():.3f}%, "
                  f"win rate: {(vol_shorts['pnl_pct'] > 0).mean():.1%}")