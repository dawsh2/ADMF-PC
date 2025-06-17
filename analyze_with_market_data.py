#!/usr/bin/env python3
"""
Analyze cost-optimized ensemble strategy with actual market data.
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("Cost-Optimized Ensemble Strategy Analysis")
print("="*60)

# Load market data
print("\nLoading SPY 1-minute data...")
market_df = pd.read_csv('/Users/daws/ADMF-PC/data/SPY_1m.csv')
market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
market_df = market_df.sort_values('timestamp').reset_index(drop=True)
market_df['idx'] = range(len(market_df))

print(f"Market data shape: {market_df.shape}")
print(f"Date range: {market_df['timestamp'].min()} to {market_df['timestamp'].max()}")

# Load signal data
print("\nLoading signal trace data...")
signal_file = '/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_cost_optimized_v1_014a539f/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_cost_optimized.parquet'
df_signals = pd.read_parquet(signal_file)

print(f"Compressed signal records: {len(df_signals):,}")

# Parse signal data
df_signals['timestamp'] = pd.to_datetime(df_signals['ts'])
df_signals = df_signals.sort_values('idx').reset_index(drop=True)

# Reconstruct full signal series
min_idx = df_signals['idx'].min()
max_idx = df_signals['idx'].max()
print(f"\nSignal index range: {min_idx:,} to {max_idx:,}")

# Create full signal series
full_idx = pd.DataFrame({'idx': range(min_idx, min(max_idx + 1, len(market_df)))})
df_full = full_idx.merge(df_signals[['idx', 'val']], on='idx', how='left')
df_full['signal'] = df_full['val'].ffill().fillna(0).astype(int)

# Merge with market data
print("\nMerging signals with market data...")
df_merged = market_df.merge(df_full[['idx', 'signal']], on='idx', how='left')
df_merged['signal'] = df_merged['signal'].fillna(0).astype(int)

# Filter to signal data range
df_merged = df_merged[df_merged['idx'] >= min_idx].reset_index(drop=True)

print(f"Merged data shape: {df_merged.shape}")
print(f"Date range: {df_merged['timestamp'].min()} to {df_merged['timestamp'].max()}")

# Signal statistics
total_bars = len(df_merged)
signal_bars = df_merged['signal'].sum()
signal_freq = signal_bars / total_bars

print(f"\nSignal Statistics:")
print(f"Total bars: {total_bars:,}")
print(f"Signal ON bars: {signal_bars:,}")
print(f"Signal frequency: {signal_freq:.1%}")

# Define analysis periods
periods = {
    'Full Period': (0, len(df_merged)),
    'Last 22k bars': (max(0, len(df_merged) - 22000), len(df_merged)),
    'Last 12k bars': (max(0, len(df_merged) - 12000), len(df_merged))
}

print("\n" + "="*60)
print("PERFORMANCE ANALYSIS - COST OPTIMIZED ENSEMBLE")
print("="*60)

for period_name, (start_idx, end_idx) in periods.items():
    print(f"\n{'='*50}")
    print(f"{period_name}")
    print(f"{'='*50}")
    
    df_period = df_merged.iloc[start_idx:end_idx].copy()
    
    # Basic info
    start_date = df_period['timestamp'].iloc[0]
    end_date = df_period['timestamp'].iloc[-1]
    n_bars = len(df_period)
    
    print(f"Period: {start_date} to {end_date}")
    print(f"Bars: {n_bars:,}")
    
    # Calculate returns
    df_period['returns'] = df_period['close'].pct_change()
    
    # Strategy returns - enter at next bar after signal
    df_period['strategy_returns'] = df_period['returns'] * df_period['signal'].shift(1)
    
    # Find trades
    df_period['position'] = df_period['signal']
    df_period['position_change'] = df_period['position'].diff()
    
    entries = df_period[df_period['position_change'] == 1]
    exits = df_period[df_period['position_change'] == -1]
    
    # Calculate completed trades
    trades = []
    for _, entry in entries.iterrows():
        # Find next exit
        next_exits = exits[exits.index > entry.name]
        if len(next_exits) > 0:
            exit_row = next_exits.iloc[0]
            
            # Use close prices (in reality would have slippage)
            entry_price = entry['close']
            exit_price = exit_row['close']
            
            trade = {
                'entry_time': entry['timestamp'],
                'exit_time': exit_row['timestamp'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': (exit_price / entry_price) - 1,
                'bars_held': exit_row.name - entry.name
            }
            trades.append(trade)
    
    if trades:
        trades_df = pd.DataFrame(trades)
        n_trades = len(trades_df)
        win_rate = (trades_df['return'] > 0).mean()
        avg_return = trades_df['return'].mean()
        avg_win = trades_df[trades_df['return'] > 0]['return'].mean() if win_rate > 0 else 0
        avg_loss = trades_df[trades_df['return'] < 0]['return'].mean() if win_rate < 1 else 0
        avg_bars = trades_df['bars_held'].mean()
        
        print(f"\nTrade Statistics:")
        print(f"  Completed trades: {n_trades}")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Avg return per trade: {avg_return:.3%}")
        print(f"  Avg winning trade: {avg_win:.3%}")
        print(f"  Avg losing trade: {avg_loss:.3%}")
        print(f"  Avg bars per trade: {avg_bars:.0f} ({avg_bars/60:.1f} hours)")
    else:
        n_trades = 0
    
    # Calculate cumulative returns
    strategy_returns = df_period['strategy_returns'].fillna(0)
    cum_returns = (1 + strategy_returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    
    # Buy and hold
    bh_return = (df_period['close'].iloc[-1] / df_period['close'].iloc[0]) - 1
    
    # Annualized returns
    days = (end_date - start_date).days
    years = days / 365.25
    
    if years > 0:
        annualized_return = (1 + total_return) ** (1/years) - 1
        bh_annualized = (1 + bh_return) ** (1/years) - 1
    else:
        annualized_return = 0
        bh_annualized = 0
    
    # Sharpe ratio (annualized)
    if strategy_returns.std() > 0 and len(strategy_returns) > 1:
        # Convert to daily returns for Sharpe calculation
        # Group by date and sum returns
        df_period['date'] = df_period['timestamp'].dt.date
        daily_returns = df_period.groupby('date')['strategy_returns'].sum()
        
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    # Maximum drawdown
    drawdowns = cum_returns / cum_returns.cummax() - 1
    max_drawdown = drawdowns.min()
    
    # Signal frequency in period
    period_signal_freq = df_period['signal'].mean()
    
    print(f"\nPerformance Metrics:")
    print(f"  Strategy Return: {total_return:.2%} ({annualized_return:.2%} annualized)")
    print(f"  Buy & Hold: {bh_return:.2%} ({bh_annualized:.2%} annualized)")
    print(f"  Excess Return: {total_return - bh_return:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    
    print(f"\nActivity Metrics:")
    print(f"  Signal frequency: {period_signal_freq:.1%}")
    print(f"  Trades per 1000 bars: {n_trades / (n_bars / 1000):.1f}")
    
    # Estimate transaction costs
    # Assume 0.01% per trade (1 basis point for entry + exit)
    est_transaction_cost = n_trades * 0.0001
    net_return = total_return - est_transaction_cost
    net_annualized = (1 + net_return) ** (1/years) - 1 if years > 0 else 0
    
    print(f"\nCost-Adjusted Performance (0.01% per trade):")
    print(f"  Estimated transaction cost: {est_transaction_cost:.2%}")
    print(f"  Net return: {net_return:.2%} ({net_annualized:.2%} annualized)")
    print(f"  Net excess vs B&H: {net_return - bh_return:.2%}")

print("\n" + "="*60)
print("COST OPTIMIZATION IMPACT ANALYSIS")
print("="*60)

# Overall statistics
total_entries = len(df_merged[df_merged['position'].diff() == 1])
avg_signal_freq = df_merged['signal'].mean()

print(f"\nOverall Trading Characteristics:")
print(f"- Signal frequency: {avg_signal_freq:.1%}")
print(f"- Total trade entries: {total_entries}")
print(f"- This represents a highly selective strategy")
print(f"- The 28.8% signal frequency in compressed data translates to {avg_signal_freq:.1%} actual time in market")

print(f"\nCost Optimization Benefits:")
print("1. Reduced trading frequency minimizes transaction costs")
print("2. Selective entry/exit based on ensemble agreement")
print("3. Longer holding periods to amortize bid-ask spreads")
print("4. Focus on higher-confidence signals from multiple strategies")

print("\nNote: This analysis uses close-to-close prices. In practice:")
print("- Entry would be at ask price (higher)")
print("- Exit would be at bid price (lower)")
print("- Additional slippage during fast markets")
print("- Real costs likely 2-3x the estimates shown above")