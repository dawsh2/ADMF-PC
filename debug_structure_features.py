#!/usr/bin/env python3

"""
Debug structure features to understand the key format issues.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('src'))

from strategy.components.features.indicators.structure import (
    PivotPoints, SupportResistance, TrendLines
)

def test_pivot_points():
    print("=== PIVOT POINTS TEST ===")
    pp = PivotPoints("pivot_points_standard")
    
    # Simulate some OHLC data
    test_data = [
        {"price": 100, "high": 102, "low": 98},
        {"price": 101, "high": 103, "low": 99},
        {"price": 102, "high": 104, "low": 100},
    ]
    
    for i, data in enumerate(test_data):
        result = pp.update(data["price"], data["high"], data["low"])
        print(f"Update {i+1}: {result}")
    
    print(f"Final result: {pp.value}")
    if pp.value:
        print("Keys available:", list(pp.value.keys()))

def test_support_resistance():
    print("\n=== SUPPORT RESISTANCE TEST ===")
    sr = SupportResistance(lookback=20, name="support_resistance_20")
    
    # Simulate more data for SR to work
    import random
    for i in range(30):
        price = 100 + random.uniform(-5, 5)
        high = price + random.uniform(0, 2)
        low = price - random.uniform(0, 2)
        result = sr.update(price, high, low)
        if i > 25:  # Only print last few
            print(f"Update {i+1}: {result}")
    
    print(f"Final result: {sr.value}")
    if sr.value:
        print("Keys available:", list(sr.value.keys()))

def test_trendlines():
    print("\n=== TRENDLINES TEST ===")
    tl = TrendLines(pivot_lookback=20, min_touches=2, tolerance=0.002, name="trendlines_20_2_0.002")
    
    # Simulate trend data
    import random
    for i in range(50):
        price = 100 + i * 0.5 + random.uniform(-2, 2)  # Uptrend with noise
        high = price + random.uniform(0, 1)
        low = price - random.uniform(0, 1)
        result = tl.update(price, high, low)
        if i > 45:  # Only print last few
            print(f"Update {i+1}: {result}")
    
    print(f"Final result: {tl.value}")
    if tl.value:
        print("Keys available:", list(tl.value.keys()))

if __name__ == "__main__":
    test_pivot_points()
    test_support_resistance()
    test_trendlines()