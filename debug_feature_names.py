#!/usr/bin/env python3
"""Debug feature name generation."""

import sys
sys.path.insert(0, '/Users/daws/ADMF-PC')

# Test stochastic feature directly
from src.strategy.components.features.oscillators import stochastic_feature
import pandas as pd
import numpy as np

# Create test data
n_bars = 50
high = pd.Series(np.random.uniform(100, 102, n_bars))
low = pd.Series(np.random.uniform(98, 100, n_bars)) 
close = pd.Series(np.random.uniform(99, 101, n_bars))

print("Testing stochastic feature function directly:")
print("=" * 50)

# Test stochastic calculation
result = stochastic_feature(high, low, close, k_period=5, d_period=3)
print(f"Stochastic result type: {type(result)}")
print(f"Stochastic result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

if isinstance(result, dict):
    for key, series in result.items():
        last_value = series.iloc[-1] if len(series) > 0 and not pd.isna(series.iloc[-1]) else "NaN"
        print(f"  {key}: {last_value}")

print("\n" + "=" * 50)
print("Testing how FeatureHub would name these:")

feature_name = "stochastic_5_3"  # This is what topology should generate
if isinstance(result, dict):
    for sub_name, series in result.items():
        final_name = f"{feature_name}_{sub_name}"
        print(f"  Feature name: {final_name}")

print("\n" + "=" * 50)
print("What the strategy expects:")
k_period = 5
d_period = 3
expected_k = f'stochastic_{k_period}_{d_period}_k'
expected_d = f'stochastic_{k_period}_{d_period}_d'
print(f"  Expected K: {expected_k}")
print(f"  Expected D: {expected_d}")

print("\n" + "=" * 50)
print("Do they match?")
feature_name = "stochastic_5_3"
actual_k = f"{feature_name}_k"
actual_d = f"{feature_name}_d"
print(f"  Actual K: {actual_k} == Expected K: {expected_k} ? {actual_k == expected_k}")
print(f"  Actual D: {actual_d} == Expected D: {expected_d} ? {actual_d == expected_d}")