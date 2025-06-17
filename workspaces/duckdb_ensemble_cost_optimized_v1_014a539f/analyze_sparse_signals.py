"""
Analyze cost-optimized ensemble strategy performance from sparse signal traces.
This handles the actual sparse signal format where:
- 'idx' is the bar index
- 'val' is the signal value (-1, 0, 1)
- 'px' is the price at signal time
- Each row represents a signal change (sparse storage)
"""
import pandas as pd
import numpy as np
from datetime import datetime

# Load the sparse signal trace parquet file
parquet_path = 'traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_cost_optimized.parquet'
print(f"Loading signal traces from: {parquet_path}")
df = pd.read_parquet(parquet_path)

# Convert timestamp
df['ts'] = pd.to_datetime(df['ts'])
df = df.sort_values(['idx', 'ts']).reset_index(drop=True)

print(f"\nDataFrame shape: {df.shape}")
print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
print(f"Bar index range: {df['idx'].min()} to {df['idx'].max()}")
print(f"Total signal changes (sparse storage): {len(df)}")

# Get max bar index for period filtering
max_idx = df['idx'].max()

# Define analysis periods
periods = {
    'Full Period': df.copy(),
    'Last 22k bars': df[df['idx'] > max_idx - 22000].copy(),
    'Last 12k bars': df[df['idx'] > max_idx - 12000].copy()
}

def calculate_trades_and_pnl(signal_df):
    """Calculate trades and P&L from signal transitions."""
    trades = []
    
    # Sort by index to ensure chronological order
    signal_df = signal_df.sort_values('idx').reset_index(drop=True)
    
    entry_idx = None
    entry_price = None
    entry_signal = None
    
    for i, row in signal_df.iterrows():
        current_signal = row['val']
        current_price = row['px']
        current_idx = row['idx']
        
        # If we have an open position and signal changes
        if entry_idx is not None and current_signal != entry_signal:
            # Close the position
            exit_price = current_price
            
            # Calculate return based on position direction
            if entry_signal == 1:  # Long position
                pnl_pct = (exit_price - entry_price) / entry_price
            elif entry_signal == -1:  # Short position
                pnl_pct = (entry_price - exit_price) / entry_price
            else:
                pnl_pct = 0
            
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': current_idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'signal': entry_signal,
                'pnl_pct': pnl_pct,
                'bars_held': current_idx - entry_idx
            })
            
            # Reset entry if signal goes to 0
            if current_signal == 0:
                entry_idx = None
                entry_price = None
                entry_signal = None
            else:
                # New position in opposite direction
                entry_idx = current_idx
                entry_price = current_price
                entry_signal = current_signal
        
        # If no position and signal is non-zero, open position
        elif entry_idx is None and current_signal != 0:
            entry_idx = current_idx
            entry_price = current_price
            entry_signal = current_signal
    
    return pd.DataFrame(trades)

def analyze_performance(signal_df, period_name):
    """Analyze performance for a given period."""
    print(f"\n{'='*60}")
    print(f"Performance Analysis: {period_name}")
    print(f"{'='*60}")
    
    if len(signal_df) == 0:
        print("No signals in this period")
        return
    
    # Calculate trades
    trades_df = calculate_trades_and_pnl(signal_df)
    
    if len(trades_df) == 0:
        print("No completed trades in this period")
        
        # Check if there's an open position
        last_signal = signal_df.iloc[-1]['val']
        if last_signal != 0:
            print(f"Open position at end: Signal={last_signal}, Price={signal_df.iloc[-1]['px']:.2f}")
        return
    
    # Performance metrics
    total_trades = len(trades_df)
    winning_trades = (trades_df['pnl_pct'] > 0).sum()
    losing_trades = (trades_df['pnl_pct'] < 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate cumulative return
    cumulative_return = (1 + trades_df['pnl_pct']).prod() - 1
    
    # Average trade metrics
    avg_trade_pnl = trades_df['pnl_pct'].mean()
    avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = (1 + trades_df['pnl_pct']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (using per-trade returns)
    if len(trades_df) > 1 and trades_df['pnl_pct'].std() > 0:
        sharpe_per_trade = trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()
        # Estimate annualization: ~390 bars per day * 252 days = 98,280 bars per year
        bars_in_period = signal_df['idx'].max() - signal_df['idx'].min()
        trades_per_year = total_trades * (98280 / bars_in_period) if bars_in_period > 0 else 0
        sharpe_annualized = sharpe_per_trade * np.sqrt(trades_per_year) if trades_per_year > 0 else 0
    else:
        sharpe_per_trade = 0
        sharpe_annualized = 0
    
    # Print results
    print(f"\nPeriod Statistics:")
    print(f"  Signal changes: {len(signal_df)}")
    print(f"  Bar range: {signal_df['idx'].min()} to {signal_df['idx'].max()}")
    print(f"  Total bars: {signal_df['idx'].max() - signal_df['idx'].min()}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {cumulative_return*100:.2f}%")
    print(f"  Number of Trades: {total_trades}")
    print(f"  Win Rate: {win_rate*100:.1f}% ({winning_trades}W/{losing_trades}L)")
    print(f"  Average Trade P&L: {avg_trade_pnl*100:.3f}%")
    print(f"  Average Win: {avg_win*100:.3f}%")
    print(f"  Average Loss: {avg_loss*100:.3f}%")
    print(f"  Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "  Win/Loss Ratio: N/A")
    print(f"  Maximum Drawdown: {max_drawdown*100:.2f}%")
    print(f"  Sharpe Ratio (per trade): {sharpe_per_trade:.3f}")
    print(f"  Sharpe Ratio (annualized): {sharpe_annualized:.3f}")
    
    # Trading frequency
    bars_per_trade = (signal_df['idx'].max() - signal_df['idx'].min()) / total_trades if total_trades > 0 else 0
    print(f"\nTrading Frequency:")
    print(f"  Average bars per trade: {bars_per_trade:.1f}")
    print(f"  Average hold time: {trades_df['bars_held'].mean():.1f} bars")
    print(f"  Trades per 1000 bars: {1000/bars_per_trade:.1f}" if bars_per_trade > 0 else "  Trades per 1000 bars: N/A")
    
    # Show sample trades
    print(f"\nSample trades (first 5 and last 5):")
    sample_cols = ['entry_idx', 'exit_idx', 'signal', 'entry_price', 'exit_price', 'pnl_pct', 'bars_held']
    if len(trades_df) > 10:
        print("First 5:")
        print(trades_df[sample_cols].head())
        print("\nLast 5:")
        print(trades_df[sample_cols].tail())
    else:
        print(trades_df[sample_cols])
    
    return trades_df

# Analyze each period
all_trades = {}
for period_name, period_df in periods.items():
    trades = analyze_performance(period_df, period_name)
    if trades is not None:
        all_trades[period_name] = trades

# Sparse storage efficiency
print(f"\n{'='*60}")
print("Sparse Storage Efficiency Analysis")
print(f"{'='*60}")
total_bars = max_idx - df['idx'].min()
print(f"Total bars in data range: {total_bars}")
print(f"Stored signal changes: {len(df)}")
print(f"Storage efficiency: {len(df) / total_bars * 100:.2f}% of bars stored")
print(f"Average bars between signal changes: {total_bars / len(df):.1f}")

# Signal distribution
print(f"\nSignal Distribution:")
signal_counts = df['val'].value_counts().sort_index()
for signal, count in signal_counts.items():
    print(f"  Signal {signal:2d}: {count:6d} occurrences ({count/len(df)*100:5.1f}%)")

# Compare to previous results
print(f"\n{'='*60}")
print("Comparison to Previous Ensemble Results")
print(f"{'='*60}")
print("Previous Results (Last 12k bars):")
print("  - Default Ensemble: +8.39%")
print("  - Custom Ensemble: -0.30%")
print("\nCost-Optimized Ensemble performance shown above.")

# Additional analysis: Trading patterns by signal type
print(f"\n{'='*60}")
print("Trading Pattern Analysis")
print(f"{'='*60}")

if 'Last 12k bars' in all_trades and len(all_trades['Last 12k bars']) > 0:
    trades_12k = all_trades['Last 12k bars']
    
    # Analyze by signal direction
    long_trades = trades_12k[trades_12k['signal'] == 1]
    short_trades = trades_12k[trades_12k['signal'] == -1]
    
    print(f"\nLong Trades (Last 12k bars):")
    if len(long_trades) > 0:
        print(f"  Count: {len(long_trades)}")
        print(f"  Win Rate: {(long_trades['pnl_pct'] > 0).mean()*100:.1f}%")
        print(f"  Avg P&L: {long_trades['pnl_pct'].mean()*100:.3f}%")
        print(f"  Total P&L: {((1 + long_trades['pnl_pct']).prod() - 1)*100:.2f}%")
    
    print(f"\nShort Trades (Last 12k bars):")
    if len(short_trades) > 0:
        print(f"  Count: {len(short_trades)}")
        print(f"  Win Rate: {(short_trades['pnl_pct'] > 0).mean()*100:.1f}%")
        print(f"  Avg P&L: {short_trades['pnl_pct'].mean()*100:.3f}%")
        print(f"  Total P&L: {((1 + short_trades['pnl_pct']).prod() - 1)*100:.2f}%")