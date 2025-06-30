"""Analyze trading performance with progressively tighter stop losses"""
import pandas as pd
import numpy as np
from pathlib import Path

# Workspace path
workspace = Path("workspaces/signal_generation_7ecda4b8")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"

# Read the sparse signal data
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

print("=== Bollinger RSI Simple Signals - Progressive Stop Loss Analysis ===")
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
    # Test progressively tighter stop losses
    stop_levels = [None, 0.10, 0.05, 0.02, 0.01, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
    
    # Calculate base metrics
    date_range_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
    trades_per_day = len(trades_df) / date_range_days
    trades_per_year = trades_per_day * 252
    exec_cost = 0.0001  # 1 basis point
    
    print(f"\nTotal trades: {len(trades_df)}")
    print(f"Trades per day: {trades_per_day:.2f}")
    print(f"Trades per year: {trades_per_year:.0f}")
    
    print(f"\n{'Stop Loss':<10} {'Avg/Trade':<10} {'Trades Hit':<11} {'Win Rate':<10} {'Annual Gross':<13} {'Annual Net':<13} {'Sharpe':<8}")
    print("-" * 95)
    
    best_sharpe = -999
    best_stop = None
    
    for stop_pct in stop_levels:
        # Apply stop loss
        trades_with_stop = trades_df.copy()
        if stop_pct is not None:
            trades_hit = len(trades_df[trades_df['pnl_pct'] < -stop_pct])
            trades_with_stop.loc[trades_with_stop['pnl_pct'] < -stop_pct, 'pnl_pct'] = -stop_pct
            stop_label = f"{stop_pct:.3f}%"
        else:
            trades_hit = 0
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
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_stop = stop_pct
        else:
            sharpe_str = "N/A"
            sharpe = -999
        
        print(f"{stop_label:<10} {avg_return:>7.3f}%   {trades_hit:>10} {win_rate:>8.1%}   {annual_gross_str:>12} {annual_net_str:>12} {sharpe_str:>7}")
    
    # Highlight best Sharpe
    if best_stop is not None:
        print(f"\n*** Best Sharpe ratio of {best_sharpe:.2f} achieved with {best_stop:.3f}% stop loss ***")
    
    # Detailed analysis of key stop levels
    print(f"\n=== KEY METRICS SUMMARY ===")
    
    for stop_pct in [0.005, 0.004, 0.003]:
        trades_with_stop = trades_df.copy()
        trades_hit = len(trades_df[trades_df['pnl_pct'] < -stop_pct])
        trades_with_stop.loc[trades_with_stop['pnl_pct'] < -stop_pct, 'pnl_pct'] = -stop_pct
        
        avg_return = trades_with_stop['pnl_pct'].mean()
        net_return_per_trade = (avg_return / 100) - (2 * exec_cost)
        annual_net = (1 + net_return_per_trade) ** trades_per_year - 1
        
        print(f"\n{stop_pct:.3f}% Stop Loss:")
        print(f"  - Average return per trade: {avg_return:.3f}%")
        print(f"  - Trades per day: {trades_per_day:.2f}")
        print(f"  - Win rate: {(trades_with_stop['pnl_pct'] > 0).mean():.1%}")
        print(f"  - Annual return (net): {annual_net*100:.1f}%")
        print(f"  - Trades hitting stop: {trades_hit} ({trades_hit/len(trades_df)*100:.1f}%)")
    
    # What percentage of losses are below each threshold?
    print(f"\n=== Loss Distribution Analysis ===")
    losing_trades = trades_df[trades_df['pnl_pct'] < 0]
    print(f"Total losing trades: {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.1f}%)")
    
    for threshold in [-0.001, -0.002, -0.003, -0.004, -0.005, -0.01, -0.02, -0.05]:
        count = len(trades_df[trades_df['pnl_pct'] < threshold])
        print(f"Losses worse than {threshold:.3f}%: {count} trades ({count/len(trades_df)*100:.1f}%)")
    
else:
    print("\nNo completed trades found in the signal data.")