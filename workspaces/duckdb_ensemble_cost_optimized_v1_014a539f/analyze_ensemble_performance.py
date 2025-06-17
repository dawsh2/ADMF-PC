"""
Analyze cost-optimized ensemble strategy performance from sparse signal traces.
"""
import pandas as pd
import numpy as np
from datetime import datetime

# Load the sparse signal trace parquet file
parquet_path = 'traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_cost_optimized.parquet'
print(f"Loading signal traces from: {parquet_path}")
df = pd.read_parquet(parquet_path)

print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast 5 rows:")
print(df.tail())

# Check for required columns
required_cols = ['bar_idx', 'entry_price', 'exit_price', 'signal']
for col in required_cols:
    if col not in df.columns:
        print(f"\nWARNING: Missing required column: {col}")

# Get max bar index for period filtering
max_bar_idx = df['bar_idx'].max()
print(f"\nMax bar_idx: {max_bar_idx}")
print(f"Total signal changes (sparse storage): {len(df)}")

# Define analysis periods
periods = {
    'Full Period': df,
    'Last 22k bars': df[df['bar_idx'] > max_bar_idx - 22000],
    'Last 12k bars': df[df['bar_idx'] > max_bar_idx - 12000]
}

def calculate_performance(trades_df, period_name):
    """Calculate performance metrics for a given period."""
    print(f"\n{'='*60}")
    print(f"Performance Analysis: {period_name}")
    print(f"{'='*60}")
    
    if len(trades_df) == 0:
        print("No trades in this period")
        return
    
    # Filter for actual trades (signal changes from 0 to non-zero)
    # In sparse format, each row represents a signal change
    trades = trades_df[trades_df['signal'] != 0].copy()
    
    if len(trades) == 0:
        print("No position entries in this period")
        return
    
    print(f"Signal changes in period: {len(trades_df)}")
    print(f"Position entries (non-zero signals): {len(trades)}")
    
    # Calculate returns using entry and exit prices
    # For sparse format, entry_price is when position is opened, exit_price is when closed
    trades['return'] = (trades['exit_price'] - trades['entry_price']) / trades['entry_price']
    
    # Handle NaN exit prices (positions still open)
    open_positions = trades['exit_price'].isna().sum()
    if open_positions > 0:
        print(f"Open positions (no exit price): {open_positions}")
        # For open positions, we can't calculate return
        trades = trades.dropna(subset=['exit_price'])
    
    if len(trades) == 0:
        print("No closed trades to analyze")
        return
    
    # Calculate metrics
    total_return = (1 + trades['return']).prod() - 1
    num_trades = len(trades)
    
    # Adjust returns for signal direction
    trades['adj_return'] = trades['return'] * trades['signal']
    
    # Win rate
    wins = (trades['adj_return'] > 0).sum()
    win_rate = wins / num_trades if num_trades > 0 else 0
    
    # Average trade P&L
    avg_trade_pnl = trades['adj_return'].mean()
    
    # Maximum drawdown calculation
    cumulative_returns = (1 + trades['adj_return']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (assuming 0 risk-free rate and using per-trade returns)
    if len(trades) > 1:
        sharpe = trades['adj_return'].mean() / trades['adj_return'].std() if trades['adj_return'].std() > 0 else 0
        # Annualize assuming ~6.5 hours of trading per day, 252 days per year
        # With 1-minute bars, that's approximately 390 * 252 = 98,280 bars per year
        bars_in_period = trades_df['bar_idx'].max() - trades_df['bar_idx'].min()
        annualization_factor = np.sqrt(98280 / bars_in_period) if bars_in_period > 0 else 1
        sharpe_annualized = sharpe * annualization_factor
    else:
        sharpe = 0
        sharpe_annualized = 0
    
    # Print results
    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {total_return*100:.2f}%")
    print(f"  Number of Trades: {num_trades}")
    print(f"  Win Rate: {win_rate*100:.1f}%")
    print(f"  Average Trade P&L: {avg_trade_pnl*100:.3f}%")
    print(f"  Maximum Drawdown: {max_drawdown*100:.2f}%")
    print(f"  Sharpe Ratio (per trade): {sharpe:.3f}")
    print(f"  Sharpe Ratio (annualized): {sharpe_annualized:.3f}")
    
    # Additional analysis for sparse storage
    print(f"\nSparse Storage Analysis:")
    print(f"  Average bars between signal changes: {bars_in_period / len(trades_df):.1f}" if len(trades_df) > 0 else "  No signal changes")
    
    # Show sample trades
    print(f"\nSample trades (first 5):")
    sample = trades[['bar_idx', 'signal', 'entry_price', 'exit_price', 'return', 'adj_return']].head()
    print(sample)

# Analyze each period
for period_name, period_df in periods.items():
    calculate_performance(period_df, period_name)

# Additional analysis on sparse storage efficiency
print(f"\n{'='*60}")
print("Sparse Storage Efficiency Analysis")
print(f"{'='*60}")
print(f"Total bars in data range: {max_bar_idx - df['bar_idx'].min()}")
print(f"Stored signal changes: {len(df)}")
print(f"Storage efficiency: {len(df) / (max_bar_idx - df['bar_idx'].min()) * 100:.2f}% of bars stored")

# Check signal distribution
print(f"\nSignal Distribution:")
signal_counts = df['signal'].value_counts().sort_index()
for signal, count in signal_counts.items():
    print(f"  Signal {signal:2.0f}: {count:6d} occurrences ({count/len(df)*100:5.1f}%)")

# Compare to previous results
print(f"\n{'='*60}")
print("Comparison to Previous Ensemble Results")
print(f"{'='*60}")
print("Previous Results (Last 12k bars):")
print("  - Default Ensemble: +8.39%")
print("  - Custom Ensemble: -0.30%")
print("\nCost-Optimized Ensemble performance shown above.")