"""Analyze trading performance with stop losses for workspace signal_generation_7ecda4b8"""
import pandas as pd
import numpy as np
from pathlib import Path

# Workspace path
workspace = Path("workspaces/signal_generation_7ecda4b8")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"

# Read the sparse signal data
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

print("=== Bollinger RSI Simple Signals - Performance Analysis with Stop Losses ===")
print(f"Workspace: signal_generation_7ecda4b8")
print(f"Date range: {signals_df['ts'].min()} to {signals_df['ts'].max()}")

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
            'bars_held': bars_held
        })
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    # Test different stop loss levels
    stop_levels = [None, 0.005, 0.01, 0.02, 0.05]
    
    print(f"\nTotal trades: {len(trades_df)}")
    print(f"\n{'Stop Loss':<12} {'Avg Return':<12} {'Win Rate':<10} {'Annual Gross':<15} {'Annual Net':<15} {'Sharpe':<10}")
    print("-" * 85)
    
    date_range_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
    trades_per_day = len(trades_df) / date_range_days
    trades_per_year = trades_per_day * 252
    exec_cost = 0.0001  # 1 basis point
    
    for stop_pct in stop_levels:
        # Apply stop loss
        trades_with_stop = trades_df.copy()
        if stop_pct is not None:
            trades_with_stop.loc[trades_with_stop['pnl_pct'] < -stop_pct, 'pnl_pct'] = -stop_pct
            stop_label = f"{stop_pct:.3f}%"
        else:
            stop_label = "None"
        
        # Calculate metrics
        avg_return = trades_with_stop['pnl_pct'].mean()
        win_rate = (trades_with_stop['pnl_pct'] > 0).mean()
        
        # Annualized returns
        avg_return_decimal = avg_return / 100
        net_return_per_trade = avg_return_decimal - (2 * exec_cost)
        
        # Gross annual
        if avg_return_decimal > -1:
            annual_gross = (1 + avg_return_decimal) ** trades_per_year - 1
            annual_gross_str = f"{annual_gross*100:.1f}%"
        else:
            annual_gross_str = "N/A"
        
        # Net annual
        if net_return_per_trade > -1:
            annual_net = (1 + net_return_per_trade) ** trades_per_year - 1
            annual_net_str = f"{annual_net*100:.1f}%"
        else:
            annual_net_str = "N/A"
        
        # Calculate Sharpe ratio
        trades_with_stop['date'] = trades_with_stop['exit_time'].dt.date
        trades_with_stop['pnl_decimal'] = trades_with_stop['pnl_pct'] / 100
        
        # Apply execution costs to each trade
        trades_with_stop['pnl_decimal_net'] = trades_with_stop['pnl_decimal'] - (2 * exec_cost)
        
        # Daily returns
        daily_returns = trades_with_stop.groupby('date')['pnl_decimal_net'].sum()
        date_range = pd.date_range(start=daily_returns.index.min(), 
                                  end=daily_returns.index.max(), 
                                  freq='D')
        daily_returns = daily_returns.reindex(date_range.date, fill_value=0)
        
        # Sharpe calculation
        daily_mean = daily_returns.mean()
        daily_std = daily_returns.std()
        
        if daily_std > 0:
            sharpe = (daily_mean * 252) / (daily_std * np.sqrt(252))
            sharpe_str = f"{sharpe:.2f}"
        else:
            sharpe_str = "N/A"
        
        print(f"{stop_label:<12} {avg_return:>8.3f}%    {win_rate:>7.1%}    {annual_gross_str:>14} {annual_net_str:>14} {sharpe_str:>9}")
    
    # Detailed analysis for 0.01% stop
    print(f"\n=== DETAILED ANALYSIS: 0.01% Stop Loss ===")
    trades_001_stop = trades_df.copy()
    trades_001_stop.loc[trades_001_stop['pnl_pct'] < -0.01, 'pnl_pct'] = -0.01
    
    trades_hit_stop = len(trades_df[trades_df['pnl_pct'] < -0.01])
    print(f"Trades hitting 0.01% stop: {trades_hit_stop} ({trades_hit_stop/len(trades_df)*100:.1f}%)")
    
    # By direction
    for direction in ['long', 'short']:
        dir_trades = trades_df[trades_df['direction'] == direction]
        dir_trades_stop = dir_trades.copy()
        dir_trades_stop.loc[dir_trades_stop['pnl_pct'] < -0.01, 'pnl_pct'] = -0.01
        
        print(f"\n{direction.capitalize()}s:")
        print(f"  Original avg: {dir_trades['pnl_pct'].mean():.3f}%")
        print(f"  With 0.01% stop: {dir_trades_stop['pnl_pct'].mean():.3f}%")
        print(f"  Trades hit stop: {len(dir_trades[dir_trades['pnl_pct'] < -0.01])}")
        print(f"  Win rate: {(dir_trades['pnl_pct'] > 0).mean():.1%} → {(dir_trades_stop['pnl_pct'] > 0).mean():.1%}")
    
    # Return distribution
    print(f"\n=== Return Distribution (No Stop) ===")
    print(f"5th percentile: {trades_df['pnl_pct'].quantile(0.05):.3f}%")
    print(f"25th percentile: {trades_df['pnl_pct'].quantile(0.25):.3f}%")
    print(f"Median: {trades_df['pnl_pct'].median():.3f}%")
    print(f"75th percentile: {trades_df['pnl_pct'].quantile(0.75):.3f}%")
    print(f"95th percentile: {trades_df['pnl_pct'].quantile(0.95):.3f}%")
    
else:
    print("\nNo completed trades found in the signal data.")