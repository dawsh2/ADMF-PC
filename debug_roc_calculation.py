#!/usr/bin/env python3
"""Debug ROC calculation in the system."""

import sys
sys.path.insert(0, '/Users/daws/ADMF-PC')

# Test the ROC feature calculation
test_prices = [100.0, 100.1, 100.2, 100.15, 100.25, 100.35, 100.3, 100.4, 100.5, 100.45, 100.55]

print("Test price series:")
for i, price in enumerate(test_prices):
    print(f"  Bar {i}: ${price}")

print("\nManual ROC calculation (10-period):")
for i in range(10, len(test_prices)):
    current = test_prices[i]
    past = test_prices[i-10]
    roc = ((current - past) / past) * 100
    print(f"  Bar {i}: ROC = ((${current} - ${past}) / ${past}) * 100 = {roc:.3f}%")
    
    # Check signals with different thresholds
    for threshold in [0.05, 0.1, 0.2]:
        if roc > threshold:
            signal = "BUY"
        elif roc < -threshold:
            signal = "SELL"
        else:
            signal = "FLAT"
        print(f"    Threshold {threshold}%: {signal}")

# Now let's create a minimal test of the actual strategy
print("\n" + "=" * 60)
print("Testing ROC threshold strategy directly:")
print("=" * 60)

from src.strategy.strategies.indicators.oscillators import roc_threshold

# Test with specific ROC values
test_cases = [
    {"roc": 0.15, "threshold": 0.1, "expected": "BUY"},
    {"roc": -0.15, "threshold": 0.1, "expected": "SELL"},
    {"roc": 0.05, "threshold": 0.1, "expected": "FLAT"},
    {"roc": 0.25, "threshold": 0.2, "expected": "BUY"},
    {"roc": -0.25, "threshold": 0.2, "expected": "SELL"},
]

for test in test_cases:
    features = {f"roc_10": test["roc"]}
    bar = {"symbol": "SPY", "timeframe": "1m", "timestamp": "2023-01-01", "close": 100}
    params = {"roc_period": 10, "threshold": test["threshold"]}
    
    result = roc_threshold(features, bar, params)
    signal_value = result['signal_value'] if result else None
    signal_type = "BUY" if signal_value == 1 else "SELL" if signal_value == -1 else "FLAT" if signal_value == 0 else "NONE"
    
    print(f"ROC={test['roc']:.2f}%, Threshold={test['threshold']}%: Signal={signal_type} (expected {test['expected']})")
    if signal_type != test['expected']:
        print("  ERROR: Signal mismatch!")