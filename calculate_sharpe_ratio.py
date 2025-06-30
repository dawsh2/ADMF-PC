#!/usr/bin/env python3
"""Calculate accurate Sharpe ratio for the strategy."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time

# Best performing strategies
GOOD_STRATEGY_IDS = [17, 18, 19]  # Multipliers 1.35, 1.40, 1.45

def get_strategy_returns(signals_file: str, transaction_cost_bps: float = 1.0):
    """Get returns series for Sharpe calculation."""
    
    signals_df = pd.read_parquet(signals_file)
    if signals_df.empty:
        return pd.Series()
    
    signals_df['datetime'] = pd.to_datetime(signals_df['ts'])
    signals_df['date'] = signals_df['datetime'].dt.date
    signals_df['time'] = signals_df['datetime'].dt.time
    
    # Standard EOD settings
    no_new_trades_time = time(15, 45)
    force_exit_time = time(15, 59)
    stop_pct = 0.003
    
    trades = []
    entry_price = None
    entry_signal = None
    entry_time = None
    entry_date = None
    
    for i in range(len(signals_df)):
        signal = signals_df.iloc[i]['val']
        price = signals_df.iloc[i]['px']
        current_time = signals_df.iloc[i]['datetime']
        current_date = current_time.date()
        current_tod = current_time.time()
        
        if entry_price is not None:
            exit_reason = None
            exit_price = price
            
            # EOD check
            if current_date != entry_date or current_tod >= force_exit_time:
                exit_reason = 'eod'
            
            # Stop check
            elif stop_pct:
                if entry_signal > 0:
                    if (entry_price - price) / entry_price > stop_pct:
                        exit_reason = 'stop'
                        exit_price = entry_price * (1 - stop_pct)
                else:
                    if (price - entry_price) / entry_price > stop_pct:
                        exit_reason = 'stop'
                        exit_price = entry_price * (1 + stop_pct)
            
            # Signal exit
            if not exit_reason and (signal == 0 or signal == -entry_signal):
                exit_reason = 'signal'
            
            if exit_reason:
                gross_return = (exit_price / entry_price - 1) * entry_signal
                # Apply transaction costs
                net_return = gross_return - (transaction_cost_bps / 10000)
                
                trades.append({
                    'date': current_date,
                    'return': net_return
                })
                
                entry_price = None
                
                # Re-enter on reversal
                if signal != 0 and exit_reason == 'signal' and current_tod < no_new_trades_time:
                    entry_price = price
                    entry_signal = signal
                    entry_time = current_time
                    entry_date = current_date
        
        elif signal != 0 and entry_price is None and current_tod < no_new_trades_time:
            entry_price = price
            entry_signal = signal
            entry_time = current_time
            entry_date = current_date
    
    if not trades:
        return pd.Series()
    
    # Convert to daily returns
    df = pd.DataFrame(trades)
    daily_returns = df.groupby('date')['return'].sum()
    
    return daily_returns

# Calculate for all three strategies combined
workspace = "workspaces/signal_generation_5433aa9b"
all_daily_returns = []

for strat_id in GOOD_STRATEGY_IDS:
    signal_file = f"{workspace}/traces/SPY_1m/signals/keltner_bands/SPY_compiled_strategy_{strat_id}.parquet"
    daily_returns = get_strategy_returns(signal_file, transaction_cost_bps=1.0)
    all_daily_returns.append(daily_returns)

# Combine strategies (equal weight)
combined_returns = pd.concat(all_daily_returns, axis=1).fillna(0).mean(axis=1)

print("=== SHARPE RATIO CALCULATION ===\n")

# Daily statistics
daily_mean = combined_returns.mean()
daily_std = combined_returns.std()
daily_sharpe = daily_mean / daily_std if daily_std > 0 else 0

print(f"DAILY STATISTICS:")
print(f"Mean daily return: {daily_mean*100:.4f}%")
print(f"Daily volatility: {daily_std*100:.4f}%")
print(f"Daily Sharpe: {daily_sharpe:.4f}")

# Annualized statistics
trading_days = 252
annual_return = daily_mean * trading_days
annual_volatility = daily_std * np.sqrt(trading_days)
annual_sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0

# Alternative calculation using sqrt(252)
annual_sharpe_alt = daily_sharpe * np.sqrt(trading_days)

print(f"\nANNUALIZED STATISTICS:")
print(f"Expected annual return: {annual_return*100:.2f}%")
print(f"Annual volatility: {annual_volatility*100:.2f}%")
print(f"Annual Sharpe ratio: {annual_sharpe:.2f}")
print(f"Annual Sharpe (alt): {annual_sharpe_alt:.2f}")

# Additional risk metrics
print(f"\nRISK METRICS:")
print(f"Best day: {combined_returns.max()*100:.2f}%")
print(f"Worst day: {combined_returns.min()*100:.2f}%")
print(f"% Positive days: {(combined_returns > 0).mean()*100:.1f}%")

# Calculate max drawdown
cumulative = (1 + combined_returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()

print(f"Max drawdown: {max_drawdown*100:.2f}%")

# Calmar ratio (return / max drawdown)
calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
print(f"Calmar ratio: {calmar:.2f}")

# Information about sample
print(f"\nSAMPLE INFO:")
print(f"Trading days in sample: {len(combined_returns)}")
print(f"Total calendar days: {(combined_returns.index[-1] - combined_returns.index[0]).days}")

# Risk-adjusted return comparison
print(f"\nCOMPARISON:")
print(f"Strategy Sharpe: {annual_sharpe:.2f}")
print(f"Typical equity Sharpe: 0.5-1.0")
print(f"Good quant strategy: 1.0-2.0")
print(f"Excellent quant strategy: >2.0")

# Calculate required return for different Sharpe targets
for target_sharpe in [0.5, 1.0, 1.5, 2.0]:
    required_return = target_sharpe * annual_volatility
    print(f"\nFor Sharpe {target_sharpe}: need {required_return*100:.1f}% annual return")
    print(f"  = {required_return/trading_days*10000:.2f} bps per day")
    print(f"  = {required_return/trading_days/6.3*10000:.2f} bps per trade (at 6.3 trades/day)")