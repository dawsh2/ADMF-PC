#!/usr/bin/env python3

import pandas as pd

prices = pd.read_csv("./data/SPY_1m.csv")

# Check VWAP
print("VWAP Analysis:")
print(f"Total NaN in vwap: {prices['vwap'].isna().sum()}")
print(f"First valid vwap index: {prices['vwap'].first_valid_index()}")

# Check where we have valid close prices
print(f"\nFirst 5 vwap values: {list(prices['vwap'].head())}")
print(f"VWAP values at indices 100-105: {list(prices['vwap'].iloc[100:106])}")

# Let's calculate our own simple VWAP
prices['my_vwap'] = (prices['Close'] * prices['Volume']).cumsum() / prices['Volume'].cumsum()
print(f"\nOur calculated VWAP at index 24: {prices['my_vwap'].iloc[24]}")
print(f"Original VWAP at index 24: {prices['vwap'].iloc[24]}")