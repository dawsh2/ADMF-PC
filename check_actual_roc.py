#!/usr/bin/env python3
"""Check actual ROC values in the real data."""

import sys
sys.path.insert(0, '/Users/daws/ADMF-PC')
import pandas as pd
from src.data.loaders.csv_loader import CSVDataLoader

# Load actual data
loader = CSVDataLoader()
data = loader.load_data(
    symbols=['SPY'],
    data_dir='data/1m',
    start_date=None,
    end_date=None,
    limit=500  # Load 500 bars
)

if 'SPY' not in data or data['SPY'].empty:
    print("No SPY data found")
    exit(1)

df = data['SPY']
print(f"Loaded {len(df)} bars of SPY data")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

# Calculate ROC for different periods
periods = [5, 10, 20]
thresholds = [0.1, 0.2, 0.3]

for period in periods:
    print(f"\n{period}-bar ROC analysis:")
    print("-" * 50)
    
    # Calculate ROC
    df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100)
    
    # Get valid ROC values (after warmup)
    roc_values = df[f'roc_{period}'].dropna()
    
    if len(roc_values) == 0:
        print("No valid ROC values")
        continue
        
    # Statistics
    print(f"Min ROC: {roc_values.min():.3f}%")
    print(f"Max ROC: {roc_values.max():.3f}%")
    print(f"Mean ROC: {roc_values.mean():.3f}%")
    print(f"Std ROC: {roc_values.std():.3f}%")
    
    # Check signals for each threshold
    for threshold in thresholds:
        buy_count = (roc_values > threshold).sum()
        sell_count = (roc_values < -threshold).sum()
        flat_count = ((roc_values >= -threshold) & (roc_values <= threshold)).sum()
        
        print(f"\nThreshold {threshold}%:")
        print(f"  BUY signals: {buy_count} ({buy_count/len(roc_values)*100:.1f}%)")
        print(f"  SELL signals: {sell_count} ({sell_count/len(roc_values)*100:.1f}%)")
        print(f"  FLAT signals: {flat_count} ({flat_count/len(roc_values)*100:.1f}%)")

# Show some specific examples
print("\n" + "=" * 60)
print("Sample ROC values and expected signals (20-bar ROC, 0.1% threshold):")
print("=" * 60)
df['expected_signal'] = df['roc_20'].apply(
    lambda x: 'BUY' if pd.notna(x) and x > 0.1 
    else 'SELL' if pd.notna(x) and x < -0.1 
    else 'FLAT' if pd.notna(x) 
    else None
)

sample = df[['close', 'roc_20', 'expected_signal']].dropna().tail(20)
for idx, row in sample.iterrows():
    print(f"{idx}: Close=${row['close']:.2f}, ROC={row['roc_20']:6.3f}%, Signal={row['expected_signal']}")