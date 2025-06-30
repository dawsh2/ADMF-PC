#!/usr/bin/env python3
"""Calculate annualized returns for 15m strategies with execution costs."""

import pandas as pd
import numpy as np
from datetime import datetime

# Load signal data
basic_signals = pd.read_parquet('/Users/daws/ADMF-PC/workspaces/signal_generation_bc947151/traces/SPY_15m_1m/signals/bollinger_rsi_simple_signals/SPY_15m_compiled_strategy_0.parquet')
optimized_signals = pd.read_parquet('/Users/daws/ADMF-PC/workspaces/signal_generation_5d710d47/traces/SPY_15m_1m/signals/bollinger_rsi_simple_signals/SPY_15m_compiled_strategy_0.parquet')

print('=== 15-MINUTE STRATEGY RETURNS ANALYSIS ===')
print('Execution cost: 0.5 bps (0.005%) per trade\n')

def calculate_returns(df, name):
    """Calculate returns from sparse signal data."""
    print(f'\n{name}:')
    print('-' * 60)
    
    # Track trades and returns
    trades = []
    current_position = None
    total_log_return = 0
    
    # Process each signal change
    for i in range(len(df)):
        row = df.iloc[i]
        signal = row['val']
        price = row['px']
        bar_idx = row['idx']
        timestamp = row['ts']
        
        # Handle position changes
        if current_position is None and signal != 0:
            # Opening new position
            current_position = {
                'entry_price': price,
                'entry_bar': bar_idx,
                'entry_time': timestamp,
                'direction': signal,
                'type': 'LONG' if signal > 0 else 'SHORT'
            }
        
        elif current_position is not None and signal == 0:
            # Closing position
            exit_price = price
            entry_price = current_position['entry_price']
            
            # Calculate log return
            if current_position['direction'] > 0:  # Long
                gross_log_return = np.log(exit_price / entry_price)
            else:  # Short
                gross_log_return = np.log(entry_price / exit_price)
            
            # Apply execution costs (0.5 bps each way = 1 bp round trip)
            # Convert to multiplicative factor
            cost_multiplier = 1 - 0.0001  # 1 bp total cost
            net_log_return = gross_log_return + np.log(cost_multiplier)
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'gross_return': np.exp(gross_log_return) - 1,
                'net_return': np.exp(net_log_return) - 1,
                'bars_held': bar_idx - current_position['entry_bar'],
                'type': current_position['type']
            })
            
            total_log_return += net_log_return
            current_position = None
        
        elif current_position is not None and signal != 0 and signal != current_position['direction']:
            # Closing and reversing position
            exit_price = price
            entry_price = current_position['entry_price']
            
            # Close current position
            if current_position['direction'] > 0:  # Long
                gross_log_return = np.log(exit_price / entry_price)
            else:  # Short
                gross_log_return = np.log(entry_price / exit_price)
            
            cost_multiplier = 1 - 0.0001
            net_log_return = gross_log_return + np.log(cost_multiplier)
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'gross_return': np.exp(gross_log_return) - 1,
                'net_return': np.exp(net_log_return) - 1,
                'bars_held': bar_idx - current_position['entry_bar'],
                'type': current_position['type']
            })
            
            total_log_return += net_log_return
            
            # Open new position
            current_position = {
                'entry_price': price,
                'entry_bar': bar_idx,
                'entry_time': timestamp,
                'direction': signal,
                'type': 'LONG' if signal > 0 else 'SHORT'
            }
    
    # Calculate statistics
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Basic stats
        num_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_return'] > 0])
        losing_trades = len(trades_df[trades_df['net_return'] < 0])
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        # Returns
        avg_win = trades_df[trades_df['net_return'] > 0]['net_return'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_return'] < 0]['net_return'].mean() if losing_trades > 0 else 0
        
        # Total return
        total_return = np.exp(total_log_return) - 1
        
        # Calculate time period for annualization
        first_timestamp = df.iloc[0]['ts']
        last_timestamp = df.iloc[-1]['ts']
        
        # Parse timestamps and calculate days
        if isinstance(first_timestamp, str):
            first_date = pd.to_datetime(first_timestamp)
            last_date = pd.to_datetime(last_timestamp)
        else:
            first_date = first_timestamp
            last_date = last_timestamp
        
        days_traded = (last_date - first_date).days
        years_traded = days_traded / 365.25
        
        # Annualized return
        if years_traded > 0:
            annualized_return = (1 + total_return) ** (1 / years_traded) - 1
        else:
            annualized_return = 0
        
        # Print results
        print(f'Total trades: {num_trades}')
        print(f'Winners: {winning_trades} ({win_rate:.1%})')
        print(f'Losers: {losing_trades} ({(1-win_rate):.1%})')
        print(f'Average win: {avg_win:.2%}')
        print(f'Average loss: {avg_loss:.2%}')
        print(f'Win/Loss ratio: {abs(avg_win/avg_loss):.2f}' if avg_loss != 0 else 'N/A')
        
        print(f'\nPeriod: {first_date.date()} to {last_date.date()} ({days_traded} days)')
        print(f'Total return: {total_return:.2%}')
        print(f'Annualized return: {annualized_return:.2%}')
        
        # Calculate Sharpe ratio (simplified - using daily returns)
        if len(trades_df) > 1:
            returns_series = trades_df['net_return']
            sharpe = np.sqrt(252) * returns_series.mean() / returns_series.std() if returns_series.std() > 0 else 0
            print(f'Sharpe ratio (simplified): {sharpe:.2f}')
        
        # Show sample trades
        print(f'\nSample trades:')
        for i, trade in trades_df.head(3).iterrows():
            print(f"  {trade['type']}: {trade['gross_return']:.2%} gross, {trade['net_return']:.2%} net ({trade['bars_held']} bars)")
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'sharpe': sharpe if 'sharpe' in locals() else None
        }

# Calculate returns for both strategies
basic_results = calculate_returns(basic_signals, 'Basic 15m Strategy')
optimized_results = calculate_returns(optimized_signals, 'Optimized 15m Strategy')

print('\n\n=== COMPARISON SUMMARY ===')
print(f'\nAfter 0.5 bps (0.005%) execution cost per trade:')
print(f'\nBasic 15m:')
print(f'  Annualized return: {basic_results["annualized_return"]:.2%}')
print(f'  Total trades: {basic_results["num_trades"]}')
print(f'  Win rate: {basic_results["win_rate"]:.1%}')

print(f'\nOptimized 15m:')
print(f'  Annualized return: {optimized_results["annualized_return"]:.2%}')
print(f'  Total trades: {optimized_results["num_trades"]}')
print(f'  Win rate: {optimized_results["win_rate"]:.1%}')

print(f'\nThe optimized strategy is {"BETTER" if optimized_results["annualized_return"] > basic_results["annualized_return"] else "WORSE"} after costs!')