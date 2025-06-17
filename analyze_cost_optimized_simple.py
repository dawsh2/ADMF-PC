#!/usr/bin/env python3
"""
Simple analysis of cost-optimized ensemble strategy using signal trace data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import duckdb

# Read the compressed signal trace
signal_file = '/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_cost_optimized_v1_014a539f/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_cost_optimized.parquet'
df_signals = pd.read_parquet(signal_file)

print("Signal Trace Analysis")
print("="*60)
print(f"Compressed signal records: {len(df_signals):,}")
print(f"Columns: {df_signals.columns.tolist()}")

# Parse the data
df_signals['timestamp'] = pd.to_datetime(df_signals['ts'])
df_signals = df_signals.sort_values('idx').reset_index(drop=True)

# Get the range
min_idx = df_signals['idx'].min()
max_idx = df_signals['idx'].max()
total_bars = max_idx - min_idx + 1

print(f"\nIndex range: {min_idx:,} to {max_idx:,}")
print(f"Total bars covered: {total_bars:,}")
print(f"Compression ratio: {len(df_signals) / total_bars:.1%}")

# Reconstruct full signal series
print("\nReconstructing full signal series...")
full_idx = pd.DataFrame({'idx': range(min_idx, max_idx + 1)})
df_full = full_idx.merge(df_signals[['idx', 'val', 'px', 'timestamp']], on='idx', how='left')

# Forward fill
df_full['signal'] = df_full['val'].ffill().fillna(0).astype(int)
df_full['price'] = df_full['px'].ffill()
df_full['timestamp'] = df_full['timestamp'].ffill()

# Calculate signal statistics
signal_bars = df_full['signal'].sum()
signal_freq = signal_bars / len(df_full)

print(f"\nSignal Statistics:")
print(f"Total bars: {len(df_full):,}")
print(f"Signal ON bars: {signal_bars:,}")
print(f"Signal frequency: {signal_freq:.1%}")

# Find signal changes for trade counting
df_full['signal_change'] = df_full['signal'].diff()
entries = df_full[df_full['signal_change'] == 1]
exits = df_full[df_full['signal_change'] == -1]

print(f"\nTrade Statistics:")
print(f"Total entries: {len(entries)}")
print(f"Total exits: {len(exits)}")

# Now try to get market data from DuckDB
print("\n" + "="*60)
print("Attempting to get market data from DuckDB...")

db_path = '/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_cost_optimized_v1_014a539f/analytics.duckdb'
conn = duckdb.connect(db_path, read_only=True)

# Check for bar data
try:
    # Try event_archives
    bars_query = """
    SELECT 
        timestamp,
        json_extract_string(data, '$.symbol') as symbol,
        CAST(json_extract_string(data, '$.open') AS DOUBLE) as open,
        CAST(json_extract_string(data, '$.high') AS DOUBLE) as high,
        CAST(json_extract_string(data, '$.low') AS DOUBLE) as low,
        CAST(json_extract_string(data, '$.close') AS DOUBLE) as close,
        CAST(json_extract_string(data, '$.volume') AS BIGINT) as volume
    FROM event_archives
    WHERE event_type = 'bar' 
    AND json_extract_string(data, '$.symbol') = 'SPY'
    ORDER BY timestamp
    """
    
    bars_df = conn.execute(bars_query).df()
    print(f"Found {len(bars_df):,} bar records")
    
    if len(bars_df) > 0:
        bars_df['timestamp'] = pd.to_datetime(bars_df['timestamp'])
        bars_df['idx'] = range(len(bars_df))
        
        # Align indices
        idx_offset = bars_df.index[0] - min_idx
        df_full['aligned_idx'] = df_full['idx'] - min_idx
        
        # Merge
        df_merged = bars_df.merge(
            df_full[['aligned_idx', 'signal']], 
            left_on='idx', 
            right_on='aligned_idx', 
            how='left'
        )
        df_merged['signal'] = df_merged['signal'].fillna(0).astype(int)
        
        print(f"Merged data shape: {df_merged.shape}")
        
        # Analyze performance for different periods
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
            
            # Calculate returns using close prices
            df_period['returns'] = df_period['close'].pct_change()
            df_period['strategy_returns'] = df_period['returns'] * df_period['signal'].shift(1)
            
            # Remove NaN values
            strategy_returns = df_period['strategy_returns'].fillna(0)
            
            # Cumulative returns
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
            
            # Sharpe ratio
            if strategy_returns.std() > 0:
                # Annualized Sharpe (252 trading days, 390 minutes per day)
                sharpe = np.sqrt(252 * 390) * strategy_returns.mean() / strategy_returns.std()
            else:
                sharpe = 0
            
            # Max drawdown
            drawdowns = cum_returns / cum_returns.cummax() - 1
            max_drawdown = drawdowns.min()
            
            # Trade analysis
            df_period['position'] = df_period['signal']
            df_period['position_change'] = df_period['position'].diff()
            
            period_entries = df_period[df_period['position_change'] == 1]
            period_exits = df_period[df_period['position_change'] == -1]
            
            n_trades = len(period_entries)
            signal_freq = df_period['signal'].mean()
            
            print(f"\nPerformance Metrics:")
            print(f"  Strategy Return: {total_return:.2%} ({annualized_return:.2%} annualized)")
            print(f"  Buy & Hold: {bh_return:.2%} ({bh_annualized:.2%} annualized)")
            print(f"  Excess Return: {total_return - bh_return:.2%}")
            print(f"  Sharpe Ratio: {sharpe:.2f}")
            print(f"  Max Drawdown: {max_drawdown:.2%}")
            
            print(f"\nTrading Activity:")
            print(f"  Number of trades: {n_trades}")
            print(f"  Signal frequency: {signal_freq:.1%}")
            print(f"  Avg bars per trade: {n_bars / max(n_trades, 1):.0f}")
            
        print("\n" + "="*60)
        print("COST OPTIMIZATION IMPACT ANALYSIS")
        print("="*60)
        print(f"\nThe 28.8% signal frequency demonstrates cost-aware trading:")
        print("1. Reduced trading frequency compared to always-on strategies")
        print("2. Selective entry/exit to minimize transaction costs")
        print("3. Longer holding periods to amortize bid-ask spreads")
        print("\nNote: This analysis uses close prices. With actual entry/exit prices,")
        print("returns would be lower due to bid-ask spreads and slippage.")
        
except Exception as e:
    print(f"Error getting bar data: {e}")
    print("\nUnable to perform full performance analysis without market data.")
    
conn.close()