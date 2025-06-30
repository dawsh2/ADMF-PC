"""Analyze profit target exit strategy for win rate improvement"""
import pandas as pd
import numpy as np
from pathlib import Path

# Workspace path
workspace = Path("workspaces/signal_generation_7ecda4b8")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"

# Read the sparse signal data
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

# Load market data for accurate profit target simulation
market_df = pd.read_parquet('./data/SPY_1m.parquet')
market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
market_df = market_df.set_index('timestamp').sort_index()

print("=== Profit Target Exit Strategy Analysis ===")
print(f"Testing profit targets to improve win rate while maintaining profitability\n")

# Convert sparse signals to trades
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
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'direction': 'long' if current_position > 0 else 'short',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'bars_held': bars_held
            })
    
    # Update position
    if new_signal != 0 and new_signal != current_position:
        entry_time = row['ts']
        entry_price = row['px']
        entry_idx = row['idx']
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)
print(f"Analyzing {len(trades_df)} trades")

# Test profit target strategies
profit_targets = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
stop_loss = 0.005  # Use optimal stop loss from previous analysis

date_range_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
base_trades_per_day = len(trades_df) / date_range_days
exec_cost = 0.0001

print(f"\n{'Target':<8} {'Win Rate':<10} {'Avg Return':<12} {'Trades/Day':<12} {'Annual Net':<12} {'Sharpe':<8}")
print("-" * 75)

best_config = {'sharpe': -999}

for target_pct in profit_targets:
    trades_modified = trades_df.copy()
    
    # Apply stop loss
    trades_modified.loc[trades_modified['pnl_pct'] < -stop_loss, 'pnl_pct'] = -stop_loss
    
    # Apply profit target
    trades_modified.loc[trades_modified['pnl_pct'] > target_pct, 'pnl_pct'] = target_pct
    
    # Calculate metrics
    avg_return = trades_modified['pnl_pct'].mean()
    win_rate = (trades_modified['pnl_pct'] > 0).mean()
    
    # Estimate increased trading frequency due to early exits
    avg_bars_original = trades_df['bars_held'].mean()
    # Trades hitting profit target would exit earlier
    profit_hit = trades_df['pnl_pct'] > target_pct
    if profit_hit.any():
        # Assume trades hitting target exit 50% earlier on average
        avg_bars_modified = (
            trades_df.loc[~profit_hit, 'bars_held'].mean() * (~profit_hit).sum() +
            trades_df.loc[profit_hit, 'bars_held'].mean() * 0.5 * profit_hit.sum()
        ) / len(trades_df)
    else:
        avg_bars_modified = avg_bars_original
    
    frequency_multiplier = avg_bars_original / avg_bars_modified if avg_bars_modified > 0 else 1
    est_trades_per_day = base_trades_per_day * frequency_multiplier
    trades_per_year = est_trades_per_day * 252
    
    # Annual return
    net_return_per_trade = (avg_return / 100) - (2 * exec_cost)
    if net_return_per_trade > -1:
        annual_net = (1 + net_return_per_trade) ** trades_per_year - 1
        annual_net_str = f"{annual_net*100:.1f}%"
    else:
        annual_net_str = "LOSS"
    
    # Sharpe calculation
    trades_modified['date'] = trades_modified['exit_time'].dt.date
    trades_modified['pnl_decimal_net'] = (trades_modified['pnl_pct'] / 100) - (2 * exec_cost)
    
    # Adjust for increased frequency
    daily_returns = trades_modified.groupby('date')['pnl_decimal_net'].sum() * frequency_multiplier
    date_range = pd.date_range(start=daily_returns.index.min(), 
                              end=daily_returns.index.max(), 
                              freq='D')
    daily_returns = daily_returns.reindex(date_range.date, fill_value=0)
    
    daily_mean = daily_returns.mean()
    daily_std = daily_returns.std()
    
    if daily_std > 0:
        sharpe = (daily_mean * 252) / (daily_std * np.sqrt(252))
    else:
        sharpe = 0
    
    if sharpe > best_config['sharpe']:
        best_config = {
            'target': target_pct,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'trades_per_day': est_trades_per_day,
            'annual_net': annual_net * 100 if net_return_per_trade > -1 else -999,
            'sharpe': sharpe
        }
    
    print(f"{target_pct:.3f}%   {win_rate:>8.1%}   {avg_return:>10.3f}%  {est_trades_per_day:>10.2f}  {annual_net_str:>11}  {sharpe:>6.2f}")

print(f"\n*** Best Configuration ***")
print(f"Profit Target: {best_config['target']:.3f}%")
print(f"Stop Loss: {stop_loss:.3f}%")
print(f"Win Rate: {best_config['win_rate']:.1%}")
print(f"Average Return per Trade: {best_config['avg_return']:.3f}%")
print(f"Trades per Day: {best_config['trades_per_day']:.2f}")
print(f"Annual Return (net): {best_config['annual_net']:.1f}%")
print(f"Sharpe Ratio: {best_config['sharpe']:.2f}")

# Compare with stop-loss only strategy
print(f"\n=== Comparison with Stop-Loss Only ===")
trades_stop_only = trades_df.copy()
trades_stop_only.loc[trades_stop_only['pnl_pct'] < -stop_loss, 'pnl_pct'] = -stop_loss
stop_only_avg = trades_stop_only['pnl_pct'].mean()
stop_only_wr = (trades_stop_only['pnl_pct'] > 0).mean()

print(f"\nStop-loss only ({stop_loss:.3f}%):")
print(f"  Win Rate: {stop_only_wr:.1%}")
print(f"  Avg Return: {stop_only_avg:.3f}%")

print(f"\nProfit target ({best_config['target']:.3f}%) + Stop-loss ({stop_loss:.3f}%):")
print(f"  Win Rate: {best_config['win_rate']:.1%} (+{(best_config['win_rate'] - stop_only_wr)*100:.1f}pp)")
print(f"  Avg Return: {best_config['avg_return']:.3f}% ({best_config['avg_return'] - stop_only_avg:+.3f}%)")

# Distribution analysis
print(f"\n=== Trade Distribution Analysis ===")
trades_capped = trades_df.copy()
trades_capped.loc[trades_capped['pnl_pct'] < -stop_loss, 'pnl_pct'] = -stop_loss
trades_capped.loc[trades_capped['pnl_pct'] > best_config['target'], 'pnl_pct'] = best_config['target']

print(f"Trades hitting stop loss: {(trades_df['pnl_pct'] < -stop_loss).sum()} ({(trades_df['pnl_pct'] < -stop_loss).sum()/len(trades_df)*100:.1f}%)")
print(f"Trades hitting profit target: {(trades_df['pnl_pct'] > best_config['target']).sum()} ({(trades_df['pnl_pct'] > best_config['target']).sum()/len(trades_df)*100:.1f}%)")
print(f"Trades between limits: {((trades_df['pnl_pct'] >= -stop_loss) & (trades_df['pnl_pct'] <= best_config['target'])).sum()} ({((trades_df['pnl_pct'] >= -stop_loss) & (trades_df['pnl_pct'] <= best_config['target'])).sum()/len(trades_df)*100:.1f}%)")