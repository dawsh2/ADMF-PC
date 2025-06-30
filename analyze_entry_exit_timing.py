"""Analyze the impact of delaying entry and/or exit by X bars"""
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

print("=== Entry/Exit Timing Analysis ===\n")

# First, create the original trades for comparison
original_trades = []
current_position = 0
entry_time = None
entry_price = None
entry_idx = None

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    new_signal = row['val']
    
    if current_position != 0 and new_signal != current_position:
        if entry_time is not None:
            original_trades.append({
                'entry_time': entry_time,
                'exit_time': row['ts'],
                'entry_idx': entry_idx,
                'exit_idx': row['idx'],
                'direction': 'long' if current_position > 0 else 'short',
                'entry_price': entry_price,
                'exit_price': row['px'],
                'pnl_pct': (row['px'] / entry_price - 1) * current_position * 100
            })
    
    if new_signal != 0 and new_signal != current_position:
        entry_time = row['ts']
        entry_price = row['px']
        entry_idx = row['idx']
    
    current_position = new_signal

original_df = pd.DataFrame(original_trades)

print(f"Original Performance:")
print(f"  Trades: {len(original_df)}")
print(f"  Win Rate: {(original_df['pnl_pct'] > 0).mean()*100:.1f}%")
print(f"  Avg Return: {original_df['pnl_pct'].mean():.3f}%\n")

# Function to calculate performance with delayed entry/exit
def calculate_delayed_performance(entry_delay=0, exit_delay=0):
    """Calculate performance with delayed entry and/or exit"""
    trades = []
    
    for _, trade in original_df.iterrows():
        # Get market data for the trade period (extended for delays)
        start_time = trade['entry_time'] - pd.Timedelta(minutes=max(0, -entry_delay))
        end_time = trade['exit_time'] + pd.Timedelta(minutes=max(0, exit_delay))
        
        mask = (market_df.index >= start_time) & (market_df.index <= end_time)
        trade_bars = market_df[mask]
        
        if len(trade_bars) == 0:
            continue
        
        # Find entry index with delay
        entry_mask = market_df.index >= trade['entry_time']
        entry_bars = market_df[entry_mask]
        
        if len(entry_bars) <= abs(entry_delay):
            continue
            
        # Adjust entry (negative delay = enter earlier, positive = enter later)
        if entry_delay < 0:  # Enter earlier
            # Need to find bars before the signal
            pre_entry_mask = (market_df.index < trade['entry_time']) & \
                            (market_df.index >= trade['entry_time'] - pd.Timedelta(minutes=-entry_delay))
            pre_bars = market_df[pre_entry_mask]
            if len(pre_bars) >= -entry_delay:
                entry_price = pre_bars.iloc[entry_delay]['close']  # negative index
                entry_time = pre_bars.iloc[entry_delay].name
            else:
                continue
        else:  # Enter later
            if len(entry_bars) > entry_delay:
                entry_price = entry_bars.iloc[entry_delay]['close']
                entry_time = entry_bars.iloc[entry_delay].name
            else:
                continue
        
        # Find exit index with delay
        exit_mask = market_df.index >= trade['exit_time']
        exit_bars = market_df[exit_mask]
        
        if len(exit_bars) <= abs(exit_delay):
            continue
        
        # Adjust exit (negative delay = exit earlier, positive = exit later)
        if exit_delay < 0:  # Exit earlier
            pre_exit_mask = (market_df.index < trade['exit_time']) & \
                           (market_df.index >= trade['exit_time'] - pd.Timedelta(minutes=-exit_delay))
            pre_bars = market_df[pre_exit_mask]
            if len(pre_bars) >= -exit_delay:
                exit_price = pre_bars.iloc[exit_delay]['close']  # negative index
                exit_time = pre_bars.iloc[exit_delay].name
            else:
                continue
        else:  # Exit later
            if len(exit_bars) > exit_delay:
                exit_price = exit_bars.iloc[exit_delay]['close']
                exit_time = exit_bars.iloc[exit_delay].name
            else:
                continue
        
        # Calculate P&L
        if trade['direction'] == 'long':
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl_pct = (entry_price / exit_price - 1) * 100
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': trade['direction'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct
        })
    
    return pd.DataFrame(trades)

# Test various delay combinations
print("=== Impact of Entry Delay (Exit Unchanged) ===")
print(f"{'Entry Delay':<15} {'Trades':<10} {'Win Rate':<12} {'Avg Return':<12} {'Change':<12}")
print("-" * 60)

for delay in [-5, -3, -2, -1, 0, 1, 2, 3, 5, 10]:
    delayed_df = calculate_delayed_performance(entry_delay=delay, exit_delay=0)
    if len(delayed_df) > 0:
        win_rate = (delayed_df['pnl_pct'] > 0).mean() * 100
        avg_return = delayed_df['pnl_pct'].mean()
        change = avg_return - original_df['pnl_pct'].mean()
        
        delay_str = f"{delay:+d} bars" if delay != 0 else "Original"
        print(f"{delay_str:<15} {len(delayed_df):<10} {win_rate:>10.1f}%  {avg_return:>10.3f}%  {change:>+10.3f}%")

print("\n=== Impact of Exit Delay (Entry Unchanged) ===")
print(f"{'Exit Delay':<15} {'Trades':<10} {'Win Rate':<12} {'Avg Return':<12} {'Change':<12}")
print("-" * 60)

for delay in [-5, -3, -2, -1, 0, 1, 2, 3, 5, 10]:
    delayed_df = calculate_delayed_performance(entry_delay=0, exit_delay=delay)
    if len(delayed_df) > 0:
        win_rate = (delayed_df['pnl_pct'] > 0).mean() * 100
        avg_return = delayed_df['pnl_pct'].mean()
        change = avg_return - original_df['pnl_pct'].mean()
        
        delay_str = f"{delay:+d} bars" if delay != 0 else "Original"
        print(f"{delay_str:<15} {len(delayed_df):<10} {win_rate:>10.1f}%  {avg_return:>10.3f}%  {change:>+10.3f}%")

print("\n=== Combined Entry and Exit Delays ===")
print(f"{'Entry/Exit':<15} {'Trades':<10} {'Win Rate':<12} {'Avg Return':<12} {'Annual Net':<12}")
print("-" * 70)

# Test promising combinations
test_combinations = [
    (0, 0),    # Original
    (1, 0),    # Delay entry by 1
    (0, 1),    # Delay exit by 1
    (1, 1),    # Delay both by 1
    (2, 0),    # Delay entry by 2
    (0, 2),    # Delay exit by 2
    (2, 2),    # Delay both by 2
    (-1, 0),   # Enter 1 bar early
    (0, -1),   # Exit 1 bar early
    (-1, 1),   # Enter early, exit late
    (1, -1),   # Enter late, exit early
]

# Calculate annualization factor
date_range_days = (original_df['exit_time'].max() - original_df['entry_time'].min()).days
original_tpd = len(original_df) / date_range_days
exec_cost = 0.0001

best_config = {'annual_net': -999}

for entry_d, exit_d in test_combinations:
    delayed_df = calculate_delayed_performance(entry_delay=entry_d, exit_delay=exit_d)
    
    if len(delayed_df) > 0:
        win_rate = (delayed_df['pnl_pct'] > 0).mean() * 100
        avg_return = delayed_df['pnl_pct'].mean()
        
        # Estimate annual return
        trade_ratio = len(delayed_df) / len(original_df)
        tpd = original_tpd * trade_ratio
        tpy = tpd * 252
        
        net_per_trade = (avg_return / 100) - (2 * exec_cost)
        if net_per_trade > -1 and tpy > 0:
            annual_net = (1 + net_per_trade) ** tpy - 1
            annual_net_pct = annual_net * 100
            annual_str = f"{annual_net_pct:.1f}%"
            
            if annual_net_pct > best_config['annual_net']:
                best_config = {
                    'entry_delay': entry_d,
                    'exit_delay': exit_d,
                    'trades': len(delayed_df),
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'annual_net': annual_net_pct
                }
        else:
            annual_str = "LOSS"
        
        delay_str = f"({entry_d:+d}, {exit_d:+d})"
        print(f"{delay_str:<15} {len(delayed_df):<10} {win_rate:>10.1f}%  {avg_return:>10.3f}%  {annual_str:>11}")

if best_config['annual_net'] > -999:
    print(f"\n*** Best Timing Configuration ***")
    print(f"Entry Delay: {best_config['entry_delay']:+d} bars")
    print(f"Exit Delay: {best_config['exit_delay']:+d} bars")
    print(f"Win Rate: {best_config['win_rate']:.1f}%")
    print(f"Avg Return: {best_config['avg_return']:.3f}%")
    print(f"Annual Return: {best_config['annual_net']:.1f}%")

# Analyze what happens with delays
print("\n=== Why Do Delays Impact Performance? ===")

# Look at immediate price action after signals
print("\nPrice Action After Entry Signals:")
for bars_after in [1, 2, 3, 5]:
    returns_after_long = []
    returns_after_short = []
    
    for _, trade in original_df.iterrows():
        mask = market_df.index >= trade['entry_time']
        after_bars = market_df[mask]
        
        if len(after_bars) > bars_after:
            entry_px = after_bars.iloc[0]['close']
            future_px = after_bars.iloc[bars_after]['close']
            ret = (future_px / entry_px - 1) * 100
            
            if trade['direction'] == 'long':
                returns_after_long.append(ret)
            else:
                returns_after_short.append(-ret)  # Invert for shorts
    
    if returns_after_long:
        avg_long = np.mean(returns_after_long)
        avg_short = np.mean(returns_after_short) if returns_after_short else 0
        print(f"  After {bars_after} bars - Long: {avg_long:+.3f}%, Short: {avg_short:+.3f}%")

print("\nPrice Action Before Exit Signals:")
for bars_before in [1, 2, 3, 5]:
    returns_before_exit = []
    
    for _, trade in original_df.iterrows():
        mask = market_df.index <= trade['exit_time']
        before_bars = market_df[mask]
        
        if len(before_bars) > bars_before:
            exit_px = before_bars.iloc[-1]['close']
            past_px = before_bars.iloc[-1-bars_before]['close']
            ret = (exit_px / past_px - 1) * 100
            
            if trade['direction'] == 'long':
                returns_before_exit.append(ret)
            else:
                returns_before_exit.append(-ret)
    
    if returns_before_exit:
        avg_ret = np.mean(returns_before_exit)
        print(f"  {bars_before} bars before exit: {avg_ret:+.3f}%")