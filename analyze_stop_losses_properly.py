"""Properly analyze stop losses by checking intra-trade price movements"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load market data
market_df = pd.read_parquet('./data/SPY_1m.parquet')
market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
market_df = market_df.set_index('timestamp').sort_index()

# Load signals
workspace = Path("workspaces/signal_generation_7ecda4b8")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

print("=== Proper Stop Loss Analysis - Checking Intra-Trade Movements ===\n")

# Convert sparse signals to trades
trades = []
current_position = 0
entry_time = None
entry_price = None

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    new_signal = row['val']
    
    if current_position != 0 and new_signal != current_position:
        if entry_time is not None:
            exit_time = row['ts']
            exit_price = row['px']
            pnl_pct = (exit_price / entry_price - 1) * current_position * 100
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'direction': 'long' if current_position > 0 else 'short',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'position': current_position
            })
    
    if new_signal != 0 and new_signal != current_position:
        entry_time = row['ts']
        entry_price = row['px']
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

# Function to check if trade would be stopped out
def check_stop_hit(trade, stop_pct, market_data):
    """Check if trade would hit stop loss during its lifetime"""
    # Get market data during trade
    mask = (market_data.index >= trade['entry_time']) & (market_data.index <= trade['exit_time'])
    trade_bars = market_data[mask]
    
    if len(trade_bars) == 0:
        return False, trade['pnl_pct']  # No data, assume not stopped
    
    if trade['direction'] == 'long':
        # For longs, check if low price hits stop
        min_low = trade_bars['low'].min()
        worst_drawdown = (min_low / trade['entry_price'] - 1) * 100
        if worst_drawdown < -stop_pct:
            # Would have been stopped out
            return True, -stop_pct
    else:  # short
        # For shorts, check if high price hits stop
        max_high = trade_bars['high'].max()
        worst_drawdown = (trade['entry_price'] / max_high - 1) * 100
        if worst_drawdown < -stop_pct:
            # Would have been stopped out
            return True, -stop_pct
    
    return False, trade['pnl_pct']

# Test different stop levels
stop_levels = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05]

print(f"Original performance: {len(trades_df)} trades, {trades_df['pnl_pct'].mean():.3f}% avg, {(trades_df['pnl_pct'] > 0).mean()*100:.1f}% win rate\n")

print(f"{'Stop Loss':<10} {'Trades Hit':<12} {'Winners Hit':<12} {'Win Rate':<10} {'Avg Return':<12} {'Annual Net':<12}")
print("-" * 80)

# Calculate annualization factors
date_range_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
trades_per_year = len(trades_df) / date_range_days * 252
exec_cost = 0.0001

results = []

for stop_pct in stop_levels:
    trades_stopped = 0
    winners_stopped = 0
    modified_returns = []
    
    for idx, trade in trades_df.iterrows():
        was_stopped, final_pnl = check_stop_hit(trade, stop_pct, market_df)
        
        if was_stopped:
            trades_stopped += 1
            if trade['pnl_pct'] > 0:  # Was originally a winner
                winners_stopped += 1
        
        modified_returns.append(final_pnl)
    
    modified_returns = pd.Series(modified_returns)
    avg_return = modified_returns.mean()
    win_rate = (modified_returns > 0).mean()
    
    # Annual return
    net_return_per_trade = (avg_return / 100) - (2 * exec_cost)
    if net_return_per_trade > -1:
        annual_net = (1 + net_return_per_trade) ** trades_per_year - 1
        annual_net_str = f"{annual_net*100:.1f}%"
    else:
        annual_net_str = "LOSS"
    
    results.append({
        'stop_pct': stop_pct,
        'trades_stopped': trades_stopped,
        'winners_stopped': winners_stopped,
        'avg_return': avg_return,
        'win_rate': win_rate,
        'annual_net': annual_net_str
    })
    
    print(f"{stop_pct:.3f}%     {trades_stopped:<12} {winners_stopped:<12} {win_rate*100:>8.1f}%   {avg_return:>10.3f}%   {annual_net_str:>11}")

# Detailed analysis of winners that would be stopped
print("\n=== Impact on Winning Trades ===")
print("Analyzing how many eventual winners would be stopped out...\n")

winning_trades = trades_df[trades_df['pnl_pct'] > 0]
print(f"Total winning trades: {len(winning_trades)}")

for stop_pct in [0.001, 0.005, 0.01]:
    winners_hit = 0
    for idx, trade in winning_trades.iterrows():
        was_stopped, _ = check_stop_hit(trade, stop_pct, market_df)
        if was_stopped:
            winners_hit += 1
    
    print(f"\n{stop_pct:.3f}% stop loss:")
    print(f"  Would stop {winners_hit} winning trades ({winners_hit/len(winning_trades)*100:.1f}%)")
    print(f"  These trades eventually made {winning_trades.iloc[:winners_hit]['pnl_pct'].mean():.3f}% on average")

# Sample some trades that would be stopped
print("\n=== Example: Winners That Would Be Stopped (0.005% stop) ===")
stop_test = 0.005
examples_shown = 0

for idx, trade in winning_trades.iterrows():
    if examples_shown >= 5:
        break
        
    was_stopped, _ = check_stop_hit(trade, stop_test, market_df)
    if was_stopped:
        # Get the worst drawdown
        mask = (market_df.index >= trade['entry_time']) & (market_df.index <= trade['exit_time'])
        trade_bars = market_df[mask]
        
        if trade['direction'] == 'long':
            worst_price = trade_bars['low'].min()
            worst_dd = (worst_price / trade['entry_price'] - 1) * 100
        else:
            worst_price = trade_bars['high'].max()
            worst_dd = (trade['entry_price'] / worst_price - 1) * 100
        
        print(f"\n{trade['direction']} trade:")
        print(f"  Entry: ${trade['entry_price']:.2f}")
        print(f"  Worst drawdown: {worst_dd:.3f}%")
        print(f"  Final P&L: {trade['pnl_pct']:.3f}%")
        print(f"  Would have been stopped at -0.005%")
        
        examples_shown += 1