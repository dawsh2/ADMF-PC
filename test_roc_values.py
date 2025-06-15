#!/usr/bin/env python3
"""Test actual ROC values with minute data."""

import sys
sys.path.insert(0, '/Users/daws/ADMF-PC')
import numpy as np

# Simulate typical minute-by-minute price changes
# Most stocks move less than 0.1% per minute on average
print("Simulating ROC values for minute-by-minute data:")
print("=" * 60)

# Starting price
prices = [100.0]

# Generate 100 minutes of price data
np.random.seed(42)
for i in range(100):
    # Typical minute change is very small (0.05% std dev)
    minute_change = np.random.normal(0, 0.0005)  # 0.05% std dev
    new_price = prices[-1] * (1 + minute_change)
    prices.append(new_price)

# Calculate ROC for different periods
periods = [5, 10, 20]
thresholds = [1.0, 2.0, 3.0]

print(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")
print(f"Total price change over 100 minutes: {((prices[-1] - prices[0]) / prices[0] * 100):.2f}%")
print()

for period in periods:
    print(f"\n{period}-minute ROC values:")
    print("-" * 40)
    
    roc_values = []
    for i in range(period, len(prices)):
        current_price = prices[i]
        past_price = prices[i-period]
        roc = ((current_price - past_price) / past_price) * 100
        roc_values.append(roc)
    
    # Statistics
    min_roc = min(roc_values)
    max_roc = max(roc_values)
    avg_roc = np.mean(roc_values)
    std_roc = np.std(roc_values)
    
    print(f"Min ROC: {min_roc:.3f}%")
    print(f"Max ROC: {max_roc:.3f}%")
    print(f"Avg ROC: {avg_roc:.3f}%")
    print(f"Std ROC: {std_roc:.3f}%")
    
    # Check how many would trigger signals
    for threshold in thresholds:
        buy_signals = sum(1 for r in roc_values if r > threshold)
        sell_signals = sum(1 for r in roc_values if r < -threshold)
        flat_signals = sum(1 for r in roc_values if -threshold <= r <= threshold)
        
        print(f"\nThreshold {threshold}%:")
        print(f"  BUY signals (ROC > {threshold}%): {buy_signals} ({buy_signals/len(roc_values)*100:.1f}%)")
        print(f"  SELL signals (ROC < -{threshold}%): {sell_signals} ({sell_signals/len(roc_values)*100:.1f}%)")
        print(f"  FLAT signals: {flat_signals} ({flat_signals/len(roc_values)*100:.1f}%)")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("With minute-by-minute data, ROC values are typically < 0.5%")
print("Thresholds of 1%, 2%, 3% are too high for 1m timeframe")
print("This explains why all signals are FLAT (0)")