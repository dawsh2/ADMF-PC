#!/usr/bin/env python3
"""
Analyze cost-optimized ensemble strategy performance using entry/exit prices.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load the signal data
file_path = '/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_cost_optimized_v1_014a539f/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_cost_optimized.parquet'
df = pd.read_parquet(file_path)

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 10 rows:")
print(df.head(10))
print("\nData types:")
print(df.dtypes)

# Check for entry/exit price columns
print("\nFirst few rows:")
print(df.head())

# Ensure timestamp is datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Calculate basic statistics
total_bars = len(df)
signal_bars = df['signal'].sum()
signal_frequency = signal_bars / total_bars

print(f"\nTotal bars: {total_bars:,}")
print(f"Signal bars: {signal_bars:,}")
print(f"Signal frequency: {signal_frequency:.1%}")

# Define the three periods
last_22k_start = len(df) - 22000
last_12k_start = len(df) - 12000

periods = {
    'Full Period': (0, len(df)),
    'Last 22k bars': (last_22k_start, len(df)),
    'Last 12k bars': (last_12k_start, len(df))
}

def calculate_performance(df_period, period_name):
    """Calculate performance metrics for a given period."""
    print(f"\n{'='*60}")
    print(f"Performance Analysis: {period_name}")
    print(f"{'='*60}")
    
    # Basic info
    start_date = df_period['timestamp'].iloc[0]
    end_date = df_period['timestamp'].iloc[-1]
    n_bars = len(df_period)
    
    print(f"Period: {start_date} to {end_date}")
    print(f"Number of bars: {n_bars:,}")
    
    # Check if we have entry/exit prices
    has_entry_exit = 'entry_price' in df_period.columns and 'exit_price' in df_period.columns
    
    if has_entry_exit:
        print("\nUsing entry/exit prices for calculations")
        
        # Find trade entries and exits
        df_period = df_period.copy()
        df_period['position'] = df_period['signal'].astype(int)
        df_period['position_change'] = df_period['position'].diff()
        
        # Entries are when position changes from 0 to 1
        entries = df_period[df_period['position_change'] == 1].copy()
        # Exits are when position changes from 1 to 0
        exits = df_period[df_period['position_change'] == -1].copy()
        
        # Calculate trades
        trades = []
        for i, entry in entries.iterrows():
            # Find the next exit after this entry
            next_exits = exits[exits.index > i]
            if len(next_exits) > 0:
                exit_row = next_exits.iloc[0]
                
                # Use entry_price from entry bar and exit_price from exit bar
                entry_price = entry['entry_price'] if pd.notna(entry['entry_price']) else entry['close']
                exit_price = exit_row['exit_price'] if pd.notna(exit_row['exit_price']) else exit_row['close']
                
                trade = {
                    'entry_time': entry['timestamp'],
                    'exit_time': exit_row['timestamp'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': (exit_price / entry_price) - 1,
                    'bars_held': exit_row.name - i
                }
                trades.append(trade)
        
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) > 0:
            # Calculate cumulative returns
            trades_df['cum_return'] = (1 + trades_df['return']).cumprod()
            
            # Performance metrics
            total_return = trades_df['cum_return'].iloc[-1] - 1
            n_trades = len(trades_df)
            win_rate = (trades_df['return'] > 0).mean()
            avg_return = trades_df['return'].mean()
            avg_win = trades_df[trades_df['return'] > 0]['return'].mean() if win_rate > 0 else 0
            avg_loss = trades_df[trades_df['return'] < 0]['return'].mean() if win_rate < 1 else 0
            
            # Calculate annualized metrics
            days = (end_date - start_date).days
            years = days / 365.25
            annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
            
            # Sharpe ratio (assuming 390 bars per day)
            daily_returns = []
            current_date = start_date.date()
            for _, trade in trades_df.iterrows():
                if trade['exit_time'].date() > current_date:
                    daily_returns.append(trade['return'])
                    current_date = trade['exit_time'].date()
            
            if len(daily_returns) > 1:
                sharpe = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
            else:
                sharpe = 0
            
            # Maximum drawdown
            drawdowns = (trades_df['cum_return'] / trades_df['cum_return'].cummax() - 1)
            max_drawdown = drawdowns.min()
            
            # Average trade duration
            avg_bars_held = trades_df['bars_held'].mean()
            avg_minutes_held = avg_bars_held  # Since 1-minute bars
            
            print(f"\nTrade Statistics:")
            print(f"Number of trades: {n_trades}")
            print(f"Win rate: {win_rate:.1%}")
            print(f"Average return per trade: {avg_return:.2%}")
            print(f"Average winning trade: {avg_win:.2%}")
            print(f"Average losing trade: {avg_loss:.2%}")
            print(f"Average trade duration: {avg_minutes_held:.0f} minutes ({avg_minutes_held/60:.1f} hours)")
            
            print(f"\nPerformance Metrics:")
            print(f"Total return: {total_return:.2%}")
            print(f"Annualized return: {annualized_return:.2%}")
            print(f"Sharpe ratio: {sharpe:.2f}")
            print(f"Maximum drawdown: {max_drawdown:.2%}")
            
            # Compare to buy-and-hold
            bh_return = (df_period['close'].iloc[-1] / df_period['close'].iloc[0]) - 1
            bh_annualized = (1 + bh_return) ** (1/years) - 1 if years > 0 else 0
            
            print(f"\nBuy-and-Hold Comparison:")
            print(f"Buy-and-hold return: {bh_return:.2%}")
            print(f"Buy-and-hold annualized: {bh_annualized:.2%}")
            print(f"Strategy vs B&H: {total_return - bh_return:.2%}")
            
            # Signal efficiency
            signal_bars = df_period['signal'].sum()
            signal_freq = signal_bars / n_bars
            trades_per_1000_bars = n_trades / (n_bars / 1000)
            
            print(f"\nSignal Efficiency:")
            print(f"Signal frequency: {signal_freq:.1%}")
            print(f"Trades per 1,000 bars: {trades_per_1000_bars:.1f}")
            print(f"Signals per trade: {signal_bars / n_trades:.0f}")
            
        else:
            print("\nNo completed trades in this period")
    
    else:
        print("\nWARNING: No entry_price/exit_price columns found. Using close prices.")
        # Fallback to close prices
        # Calculate returns when in position
        df_period = df_period.copy()
        df_period['returns'] = df_period['close'].pct_change()
        df_period['strategy_returns'] = df_period['returns'] * df_period['signal'].shift(1)
        
        # Calculate metrics
        total_return = (1 + df_period['strategy_returns']).prod() - 1
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Buy and hold
        bh_return = (df_period['close'].iloc[-1] / df_period['close'].iloc[0]) - 1
        
        print(f"\nStrategy return: {total_return:.2%}")
        print(f"Buy-and-hold return: {bh_return:.2%}")
        print(f"Strategy vs B&H: {total_return - bh_return:.2%}")

# Analyze each period
for period_name, (start_idx, end_idx) in periods.items():
    df_period = df.iloc[start_idx:end_idx].copy()
    calculate_performance(df_period, period_name)

print("\n" + "="*60)
print("Cost Optimization Impact Analysis")
print("="*60)

# Analyze the impact of cost optimization
print(f"\nOverall signal frequency: {signal_frequency:.1%}")
print("This 28.8% signal frequency suggests the cost optimization is working to:")
print("1. Reduce trading frequency to minimize transaction costs")
print("2. Focus on higher-confidence signals")
print("3. Hold positions longer to amortize entry/exit costs")

# Check if we have entry/exit prices to analyze spread impact
if 'entry_price' in df.columns and 'exit_price' in df.columns:
    # Calculate average spread impact
    entry_spread = (df['entry_price'] / df['close'] - 1).abs().mean()
    exit_spread = (df['exit_price'] / df['close'] - 1).abs().mean()
    
    print(f"\nSpread Impact:")
    print(f"Average entry price vs close: {entry_spread:.3%}")
    print(f"Average exit price vs close: {exit_spread:.3%}")
    print(f"Total round-trip impact: {(entry_spread + exit_spread):.3%}")