#!/usr/bin/env python3
"""Test new ROC thresholds with minute data."""

import sys
sys.path.insert(0, '/Users/daws/ADMF-PC')
import numpy as np

# Simulate typical minute-by-minute price changes
print("Testing new ROC thresholds (0.1%, 0.2%, 0.3%) with minute data:")
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
new_thresholds = [0.1, 0.2, 0.3]  # New thresholds

for period in periods:
    print(f"\n{period}-minute ROC analysis:")
    print("-" * 40)
    
    roc_values = []
    for i in range(period, len(prices)):
        current_price = prices[i]
        past_price = prices[i-period]
        roc = ((current_price - past_price) / past_price) * 100
        roc_values.append(roc)
    
    # Check how many would trigger signals with new thresholds
    for threshold in new_thresholds:
        buy_signals = sum(1 for r in roc_values if r > threshold)
        sell_signals = sum(1 for r in roc_values if r < -threshold)
        flat_signals = sum(1 for r in roc_values if -threshold <= r <= threshold)
        
        print(f"\nThreshold {threshold}%:")
        print(f"  BUY signals (ROC > {threshold}%): {buy_signals} ({buy_signals/len(roc_values)*100:.1f}%)")
        print(f"  SELL signals (ROC < -{threshold}%): {sell_signals} ({sell_signals/len(roc_values)*100:.1f}%)")
        print(f"  FLAT signals: {flat_signals} ({flat_signals/len(roc_values)*100:.1f}%)")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("With thresholds of 0.1%, 0.2%, 0.3%, we should see a good mix of BUY/SELL/FLAT signals")
print("This should work much better for 1m timeframe data")