#!/usr/bin/env python3
"""Analyze the signal values from bollinger strategy #10"""

import pandas as pd
import numpy as np

# Read the strategy #10 trace file
trace_file = "config/bollinger/results/20250623_062931/traces/bollinger_bands/SPY_5m_compiled_strategy_10.parquet"

df = pd.read_parquet(trace_file)

print("=== Bollinger Strategy #10 Signal Analysis ===")
print(f"Total records: {len(df):,}")
print()

# Analyze the 'val' column which contains signal values
print("Signal value distribution:")
signal_counts = df['val'].value_counts().sort_index()
print(signal_counts)
print()

# Count trades (non-zero signals)
non_zero_signals = df[df['val'] != 0]
print(f"Non-zero signals: {len(non_zero_signals):,} ({len(non_zero_signals)/len(df)*100:.1f}%)")
print(f"Buy signals (val > 0): {len(df[df['val'] > 0]):,}")
print(f"Sell signals (val < 0): {len(df[df['val'] < 0]):,}")
print()

# Show some examples of non-zero signals
if len(non_zero_signals) > 0:
    print("First 10 non-zero signals:")
    for idx, row in non_zero_signals.head(10).iterrows():
        print(f"  {row['ts']}: signal={row['val']:.4f}, price={row['px']:.2f}")
    
    # Check date range of signals
    print(f"\nDate range of signals: {non_zero_signals['ts'].min()} to {non_zero_signals['ts'].max()}")
    
    # Calculate approximate returns if we traded on these signals
    print("\n=== Simple Performance Estimate ===")
    # Convert signals to positions (1 for buy, -1 for sell)
    df_sorted = df.sort_values('ts').copy()
    df_sorted['position'] = np.where(df_sorted['val'] > 0, 1, 
                                    np.where(df_sorted['val'] < 0, -1, 0))
    
    # Forward fill positions (hold until next signal)
    df_sorted['position'] = df_sorted['position'].replace(0, np.nan).ffill().fillna(0)
    
    # Calculate returns
    df_sorted['returns'] = df_sorted['px'].pct_change()
    df_sorted['strategy_returns'] = df_sorted['position'].shift(1) * df_sorted['returns']
    
    total_return = (1 + df_sorted['strategy_returns']).prod() - 1
    avg_return = df_sorted['strategy_returns'].mean()
    
    print(f"Total return (no costs): {total_return*100:.2f}%")
    print(f"Average return per period: {avg_return*10000:.2f} bps")
    
    # With transaction costs
    position_changes = df_sorted['position'].diff().fillna(0) != 0
    num_trades = position_changes.sum()
    print(f"\nNumber of trades: {num_trades}")
    
    # Assuming 10 bps round-trip cost
    transaction_cost = 0.001  # 10 bps
    total_cost = num_trades * transaction_cost
    net_return = total_return - total_cost
    
    print(f"Transaction costs (10 bps per trade): {total_cost*100:.2f}%")
    print(f"Net return: {net_return*100:.2f}%")
    
else:
    print("No non-zero signals found!")

# Check the configuration used
print("\n=== Configuration Notes ===")
print("Source file indicates 1-minute data was used: ./data/SPY_5m_1m.csv")
print("This might explain why the isolated bollinger_10 test (using 5m data) generated no signals")