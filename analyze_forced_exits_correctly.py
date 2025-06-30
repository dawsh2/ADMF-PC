"""Correctly analyze forced exits and their impact on win rate"""
import pandas as pd
import numpy as np
from pathlib import Path

# Workspace path
workspace = Path("workspaces/signal_generation_7ecda4b8")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"

# Read the sparse signal data
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

# Load market data for accurate intermediate pricing
market_df = pd.read_parquet('./data/SPY_1m.parquet')
market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
market_df = market_df.set_index('timestamp').sort_index()

print("=== Forced Exit Analysis - Win Rate Impact ===\n")

# Convert sparse signals to trades with intermediate price tracking
trades = []
current_position = 0
entry_time = None
entry_price = None
entry_idx = None

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    new_signal = row['val']
    
    # Close existing position if changing
    if current_position != 0 and new_signal != current_position:
        if entry_time is not None:
            exit_time = row['ts']
            exit_price = row['px']
            pnl_pct = (exit_price / entry_price - 1) * current_position * 100
            bars_held = row['idx'] - entry_idx
            
            # Get intermediate prices for forced exit calculation
            mask = (market_df.index >= entry_time) & (market_df.index <= exit_time)
            trade_bars = market_df[mask]
            
            # Calculate P&L at various exit points
            pnl_at_bars = {}
            if len(trade_bars) > 0:
                for exit_bar in [5, 10, 15, 20, 30]:
                    if len(trade_bars) > exit_bar:
                        # Use close price at that bar
                        exit_px = trade_bars.iloc[exit_bar]['close']
                        pnl_at_bar = (exit_px / entry_price - 1) * current_position * 100
                        pnl_at_bars[f'pnl_at_{exit_bar}'] = pnl_at_bar
            
            trade_info = {
                'entry_time': entry_time,
                'exit_time': exit_time,
                'direction': 'long' if current_position > 0 else 'short',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'bars_held': bars_held
            }
            trade_info.update(pnl_at_bars)
            trades.append(trade_info)
    
    # Update position
    if new_signal != 0 and new_signal != current_position:
        entry_time = row['ts']
        entry_price = row['px']
        entry_idx = row['idx']
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

# Calculate win rates for different exit strategies
print("=== Win Rate Analysis by Exit Strategy ===\n")
print(f"{'Exit Strategy':<20} {'Win Rate':<12} {'Avg Return':<12} {'# Trades':<10}")
print("-" * 60)

# Original (no forced exit)
original_wr = (trades_df['pnl_pct'] > 0).mean()
original_avg = trades_df['pnl_pct'].mean()
print(f"{'Original (no limit)':<20} {original_wr*100:>10.1f}%  {original_avg:>10.3f}%  {len(trades_df):>8}")

# Forced exits at various bars
for max_bars in [5, 10, 15, 20, 30]:
    col_name = f'pnl_at_{max_bars}'
    
    # Create modified returns
    modified_returns = []
    trades_affected = 0
    
    for idx, trade in trades_df.iterrows():
        if trade['bars_held'] <= max_bars:
            # Trade naturally exited before max_bars
            modified_returns.append(trade['pnl_pct'])
        else:
            # Trade would be forced to exit
            trades_affected += 1
            if col_name in trade and pd.notna(trade[col_name]):
                modified_returns.append(trade[col_name])
            else:
                # Approximate if we don't have exact data
                modified_returns.append(trade['pnl_pct'] * (max_bars / trade['bars_held']))
    
    modified_returns = pd.Series(modified_returns)
    forced_wr = (modified_returns > 0).mean()
    forced_avg = modified_returns.mean()
    
    print(f"{f'Force exit at {max_bars}':<20} {forced_wr*100:>10.1f}%  {forced_avg:>10.3f}%  {len(trades_df):>8}")

# Detailed breakdown for 10-bar forced exit
print("\n=== Detailed Analysis: Force Exit at 10 Bars ===")

trades_over_10 = trades_df[trades_df['bars_held'] > 10]
print(f"\nTrades lasting >10 bars: {len(trades_over_10)} ({len(trades_over_10)/len(trades_df)*100:.1f}%)")
print(f"Original win rate for these: {(trades_over_10['pnl_pct'] > 0).mean()*100:.1f}%")
print(f"Original avg return for these: {trades_over_10['pnl_pct'].mean():.3f}%")

if 'pnl_at_10' in trades_over_10.columns:
    trades_with_10bar_data = trades_over_10.dropna(subset=['pnl_at_10'])
    if len(trades_with_10bar_data) > 0:
        print(f"\nIf forced to exit at 10 bars:")
        print(f"Win rate at bar 10: {(trades_with_10bar_data['pnl_at_10'] > 0).mean()*100:.1f}%")
        print(f"Avg return at bar 10: {trades_with_10bar_data['pnl_at_10'].mean():.3f}%")
        
        # Compare to final outcome
        improved = ((trades_with_10bar_data['pnl_at_10'] > 0) & (trades_with_10bar_data['pnl_pct'] <= 0)).sum()
        worsened = ((trades_with_10bar_data['pnl_at_10'] <= 0) & (trades_with_10bar_data['pnl_pct'] > 0)).sum()
        
        print(f"\nTrades that would improve: {improved} (losers that were winning at bar 10)")
        print(f"Trades that would worsen: {worsened} (winners that were losing at bar 10)")

# Win rate improvement calculation
print("\n=== Win Rate Improvement Summary ===")
for max_bars in [5, 10, 15, 20]:
    # Approximate calculation
    trades_affected = len(trades_df[trades_df['bars_held'] > max_bars])
    # Assume trades forced to exit have similar win rate to trades that naturally exit at that time
    natural_exits = trades_df[(trades_df['bars_held'] > max_bars-5) & (trades_df['bars_held'] <= max_bars+5)]
    if len(natural_exits) > 0:
        expected_wr_at_exit = (natural_exits['pnl_pct'] > 0).mean()
        print(f"\nMax {max_bars} bars:")
        print(f"  Affects {trades_affected} trades ({trades_affected/len(trades_df)*100:.1f}%)")
        print(f"  Expected win rate for affected trades: ~{expected_wr_at_exit*100:.1f}%")