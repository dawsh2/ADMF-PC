#!/usr/bin/env python3
"""Debug why ROC strategies only generate FLAT signals."""

import sys
sys.path.insert(0, '/Users/daws/ADMF-PC')
import pandas as pd
import numpy as np

# Test the ROC calculation and threshold logic
print("Testing ROC threshold strategy logic:")
print("-" * 50)

# The default threshold is 2.0 (2%)
threshold = 2.0

# Test various ROC values
test_roc_values = [-5.0, -3.0, -2.0, -1.5, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

for roc in test_roc_values:
    if roc > threshold:
        signal_value = 1   # Bullish momentum
    elif roc < -threshold:
        signal_value = -1  # Bearish momentum
    else:
        signal_value = 0   # Neutral
    
    print(f"ROC: {roc:6.1f}% => Signal: {signal_value:2d} {'(BUY)' if signal_value == 1 else '(SELL)' if signal_value == -1 else '(FLAT)'}")

print("\nConclusion:")
print("- ROC must be > 2.0% for BUY signal")
print("- ROC must be < -2.0% for SELL signal")
print("- ROC between -2.0% and 2.0% generates FLAT (0) signal")

# Now let's check what the actual ROC values might be
print("\n" + "=" * 50)
print("Simulating ROC calculation for typical price movements:")
print("=" * 50)

# Simulate some price data with typical daily movements
prices = [100.0]  # Starting price
np.random.seed(42)

# Generate 20 days of price data with small random movements
for i in range(20):
    # Typical daily price change is often < 2%
    daily_change = np.random.normal(0, 0.01)  # 0% mean, 1% std dev
    new_price = prices[-1] * (1 + daily_change)
    prices.append(new_price)

# Calculate 10-period ROC for each point
for i in range(10, len(prices)):
    current_price = prices[i]
    past_price = prices[i-10]
    roc = ((current_price - past_price) / past_price) * 100
    
    # Apply threshold logic
    if roc > threshold:
        signal_value = 1
    elif roc < -threshold:
        signal_value = -1
    else:
        signal_value = 0
    
    print(f"Day {i:2d}: Price={current_price:6.2f}, 10-day ROC={roc:6.2f}% => Signal: {signal_value} {'(BUY)' if signal_value == 1 else '(SELL)' if signal_value == -1 else '(FLAT)'}")

print("\nObservation:")
print("With typical daily price movements (1% std dev), 10-period ROC rarely exceeds ±2%")
print("This explains why most signals are FLAT (0)")