#!/usr/bin/env python3
"""
Check MA Crossover Frequency

Analyzes the actual 5/20 MA crossover frequency in the first 2000 bars
to understand why our analysis only found 3 trades.
"""
import duckdb
import pandas as pd

def check_ma_crossover_frequency():
    """Check actual MA crossover frequency in the data."""
    con = duckdb.connect()
    
    # Load first 2000 bars and calculate SMAs manually
    data = con.execute("""
    SELECT bar_index, close, timestamp
    FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    WHERE bar_index BETWEEN 0 AND 2000
    ORDER BY bar_index
    """).df()
    
    print(f"=== MA CROSSOVER FREQUENCY ANALYSIS ===")
    print(f"Data shape: {data.shape}")
    print(f"Price range: ${data.close.min():.2f} - ${data.close.max():.2f}")
    print(f"Total price change: {(data.close.iloc[-1] - data.close.iloc[0]) / data.close.iloc[0] * 100:.2f}%")
    print(f"Date range: {data.timestamp.iloc[0]} to {data.timestamp.iloc[-1]}")
    
    # Calculate SMAs
    data['sma_5'] = data['close'].rolling(window=5).mean()
    data['sma_20'] = data['close'].rolling(window=20).mean()
    
    # Find crossovers
    data['fast_above_slow'] = data['sma_5'] > data['sma_20']
    data['prev_fast_above_slow'] = data['fast_above_slow'].shift(1)
    data['crossover'] = (data['fast_above_slow'] != data['prev_fast_above_slow']) & data['prev_fast_above_slow'].notna()
    
    # Count crossovers after SMA warmup
    valid_data = data[data['sma_20'].notna()].copy()
    crossovers = valid_data[valid_data['crossover'] == True].copy()
    
    print(f"\nTotal bars with valid SMAs: {len(valid_data)}")
    print(f"Total crossovers detected: {len(crossovers)}")
    
    if len(crossovers) > 0:
        print(f"Crossover frequency: 1 every {len(valid_data) / len(crossovers):.1f} bars")
        print("\nCrossover events:")
        for i, row in crossovers.iterrows():
            direction = "Bullish" if row['fast_above_slow'] else "Bearish"
            print(f"  Bar {row['bar_index']:4d}: {direction} crossover - SMA5: {row['sma_5']:.3f}, SMA20: {row['sma_20']:.3f}")
    else:
        print("No crossovers detected in this period")
        
        # Check if the market was trending in one direction
        first_signal = valid_data.iloc[0]['fast_above_slow']
        last_signal = valid_data.iloc[-1]['fast_above_slow']
        trend = "bullish" if first_signal else "bearish"
        print(f"Market appears to be in a {trend} trend throughout the period")
        print(f"Fast SMA above slow: {first_signal} -> {last_signal}")
    
    # Show some sample SMA values
    print(f"\nSample SMA values:")
    sample_data = valid_data.iloc[::200]  # Every 200 bars
    for i, row in sample_data.iterrows():
        print(f"  Bar {row['bar_index']:4d}: Close: ${row['close']:.2f}, SMA5: {row['sma_5']:.3f}, SMA20: {row['sma_20']:.3f}, Fast>Slow: {row['fast_above_slow']}")

if __name__ == "__main__":
    check_ma_crossover_frequency()