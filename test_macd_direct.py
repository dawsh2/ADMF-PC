#!/usr/bin/env python3
"""Test MACD feature computation directly."""

import sys
sys.path.append('.')

from src.strategy.components.features.indicators.momentum import MACD

# Test MACD computation
macd = MACD(fast_period=5, slow_period=20, signal_period=7, name="macd_5_20_7")

print(f"MACD created: {macd.name}")
print(f"Is ready: {macd.is_ready}")

# Simulate some price data
prices = [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5,
          107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0,
          114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0]

print(f"\nUpdating MACD with {len(prices)} prices...")

for i, price in enumerate(prices):
    result = macd.update(price)
    if i % 5 == 0:  # Print every 5th update
        print(f"  Bar {i+1}: price={price}, result={result}, ready={macd.is_ready}")

print(f"\nFinal MACD result: {macd.value}")
print(f"Is ready: {macd.is_ready}")

# Test with bar data format
print(f"\nTesting with bar format...")
bar = {
    'open': 120.0,
    'high': 121.0,
    'low': 119.5,
    'close': 120.5,
    'volume': 100000
}

result = macd.update(bar.get('close', 0))
print(f"Result with bar data: {result}")
print(f"MACD value: {macd.value}")