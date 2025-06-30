"""Analyze trading performance for workspace signal_generation_7ecda4b8"""
import pandas as pd
import numpy as np
from pathlib import Path

# Workspace path
workspace = Path("workspaces/signal_generation_7ecda4b8")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"

# Read the sparse signal data
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

print("=== Bollinger RSI Simple Signals - Performance Analysis ===")
print(f"Workspace: signal_generation_7ecda4b8")
print(f"Date range: {signals_df['ts'].min()} to {signals_df['ts'].max()}")
print(f"Total signal changes: {len(signals_df)}")

# Convert sparse signals to trades
trades = []
current_position = 0

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    new_signal = row['val']
    
    # Close existing position if changing
    if current_position != 0 and new_signal != current_position:
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
            'direction': 'long' if current_position > 0 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'bars_held': bars_held,
            'pnl_decimal': pnl_pct / 100  # For calculations
        })
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    # Basic metrics
    print(f"\n=== PERFORMANCE METRICS ===")
    avg_return = trades_df['pnl_pct'].mean()
    win_rate = (trades_df['pnl_pct'] > 0).mean()
    total_trades = len(trades_df)
    
    print(f"Total completed trades: {total_trades}")
    print(f"Average return per trade: {avg_return:.3f}%")
    print(f"Win rate: {win_rate:.1%}")
    
    # Calculate annualized metrics
    date_range_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
    trades_per_day = total_trades / date_range_days if date_range_days > 0 else 0
    trades_per_year = trades_per_day * 252  # Trading days
    
    # Annualized return (compounded)
    avg_return_decimal = avg_return / 100
    if avg_return_decimal > -1:  # Avoid division by zero
        annual_return = (1 + avg_return_decimal) ** trades_per_year - 1
        print(f"\n=== ANNUALIZED METRICS ===")
        print(f"Trading period: {date_range_days} days")
        print(f"Trades per year: {trades_per_year:.0f}")
        print(f"Annualized return (gross): {annual_return*100:.1f}%")
        
        # With execution costs
        exec_cost = 0.0001  # 1 basis point
        net_return_per_trade = avg_return_decimal - (2 * exec_cost)  # Entry + exit
        if net_return_per_trade > -1:
            annual_return_net = (1 + net_return_per_trade) ** trades_per_year - 1
            print(f"Annualized return (1bp cost): {annual_return_net*100:.1f}%")
    
    # Sharpe ratio calculation
    returns_series = trades_df['pnl_decimal'].values
    
    # Daily returns (approximate by grouping trades by day)
    trades_df['date'] = trades_df['exit_time'].dt.date
    daily_returns = trades_df.groupby('date')['pnl_decimal'].sum()
    
    # Fill missing days with zeros
    date_range = pd.date_range(start=daily_returns.index.min(), 
                              end=daily_returns.index.max(), 
                              freq='D')
    daily_returns = daily_returns.reindex(date_range.date, fill_value=0)
    
    # Calculate Sharpe
    daily_mean = daily_returns.mean()
    daily_std = daily_returns.std()
    
    if daily_std > 0:
        # Annualize
        annual_mean = daily_mean * 252
        annual_std = daily_std * np.sqrt(252)
        sharpe_ratio = annual_mean / annual_std
        
        print(f"\n=== RISK-ADJUSTED METRICS ===")
        print(f"Daily return mean: {daily_mean*100:.3f}%")
        print(f"Daily return std: {daily_std*100:.3f}%")
        print(f"Sharpe ratio (annualized): {sharpe_ratio:.2f}")
        print(f"Sharpe ratio (with 1bp cost): {((daily_mean - trades_per_day/252 * 2 * exec_cost) * 252) / annual_std:.2f}")
    
    # Additional metrics
    print(f"\n=== TRADE STATISTICS ===")
    print(f"Best trade: {trades_df['pnl_pct'].max():.3f}%")
    print(f"Worst trade: {trades_df['pnl_pct'].min():.3f}%")
    print(f"Std dev of returns: {trades_df['pnl_pct'].std():.3f}%")
    
    # By direction
    longs = trades_df[trades_df['direction'] == 'long']
    shorts = trades_df[trades_df['direction'] == 'short']
    
    print(f"\nLongs: {len(longs)} trades, avg: {longs['pnl_pct'].mean():.3f}%, win rate: {(longs['pnl_pct'] > 0).mean():.1%}")
    print(f"Shorts: {len(shorts)} trades, avg: {shorts['pnl_pct'].mean():.3f}%, win rate: {(shorts['pnl_pct'] > 0).mean():.1%}")
    
else:
    print("\nNo completed trades found in the signal data.")