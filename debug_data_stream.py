#!/usr/bin/env python3
"""Debug what data is being streamed."""

import pandas as pd

# Load the SPY_5m data
print("=== Loading SPY_5m Data ===")
try:
    df = pd.read_csv('data/SPY_5m.csv')
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Check data range
    if 'close' in df.columns:
        print(f"\nPrice range: {df['close'].min():.2f} - {df['close'].max():.2f}")
        print(f"Mean price: {df['close'].mean():.2f}")
        
        # Calculate a simple 20-period bollinger bands for the last 30 rows
        last_30 = df.tail(30).copy()
        last_30['sma20'] = last_30['close'].rolling(20).mean()
        last_30['std20'] = last_30['close'].rolling(20).std()
        last_30['upper_band'] = last_30['sma20'] + 2 * last_30['std20']
        last_30['lower_band'] = last_30['sma20'] - 2 * last_30['std20']
        
        print("\nLast 10 rows with Bollinger Bands:")
        print(last_30[['close', 'sma20', 'upper_band', 'lower_band']].tail(10))
        
        # Check if price ever touches bands
        touches_upper = (last_30['close'] >= last_30['upper_band']).sum()
        touches_lower = (last_30['close'] <= last_30['lower_band']).sum()
        print(f"\nTouches upper band: {touches_upper}")
        print(f"Touches lower band: {touches_lower}")
        
except Exception as e:
    print(f"Error loading data: {e}")
    import traceback
    traceback.print_exc()