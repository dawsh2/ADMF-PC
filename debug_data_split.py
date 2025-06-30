#!/usr/bin/env python3
"""Debug data splitting for train/test."""

from src.data.handlers import SimpleHistoricalDataHandler
import pandas as pd

# Load and check the data
print("=== Testing Data Splits ===")
df = pd.read_csv('data/SPY_5m.csv')
print(f"Total data rows: {len(df)}")
print(f"Split at 80%: {int(len(df) * 0.8)}")
print(f"Train data would be: rows 0 to {int(len(df) * 0.8)-1}")

# Check price volatility in different segments
segments = {
    'First 1000': df.head(1000),
    'Middle 1000': df.iloc[10000:11000],
    'Last 1000': df.tail(1000),
    'Train start (first 100)': df.head(100),
    'Train end (around 80% mark)': df.iloc[int(len(df)*0.8)-50:int(len(df)*0.8)+50]
}

for name, segment in segments.items():
    if 'close' in segment.columns and len(segment) >= 20:
        # Calculate simple metrics
        sma20 = segment['close'].rolling(20).mean()
        std20 = segment['close'].rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        
        # Count band touches
        touches_upper = (segment['close'] >= upper).sum()
        touches_lower = (segment['close'] <= lower).sum()
        
        print(f"\n{name}:")
        print(f"  Price range: {segment['close'].min():.2f} - {segment['close'].max():.2f}")
        print(f"  Volatility (std): {segment['close'].std():.2f}")
        print(f"  Upper band touches: {touches_upper}")
        print(f"  Lower band touches: {touches_lower}")
        
        # Show a few examples where price is near bands
        close_to_bands = segment.copy()
        close_to_bands['sma20'] = sma20
        close_to_bands['upper'] = upper
        close_to_bands['lower'] = lower
        close_to_bands['dist_upper'] = (close_to_bands['close'] - close_to_bands['upper']).abs()
        close_to_bands['dist_lower'] = (close_to_bands['close'] - close_to_bands['lower']).abs()
        
        # Find closest approaches
        closest_upper = close_to_bands.nsmallest(3, 'dist_upper')[['close', 'upper', 'dist_upper']]
        closest_lower = close_to_bands.nsmallest(3, 'dist_lower')[['close', 'lower', 'dist_lower']]
        
        if not closest_upper.empty:
            print(f"  Closest to upper band:")
            print(closest_upper)
        if not closest_lower.empty:
            print(f"  Closest to lower band:")
            print(closest_lower)