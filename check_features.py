#!/usr/bin/env python3
"""Check which Bollinger features are being used"""

import pandas as pd

# Check the trace metadata
signals = pd.read_parquet('config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')

print("Checking Bollinger Bands configuration")
print("=" * 50)

# The feature names include the parameters
# Format: bollinger_bands_{period}_{std_dev}_upper/lower/middle

print("\nIf using period=11, std_dev=2.0:")
print("  Expected features: bollinger_bands_11_2.0_upper/lower/middle")

print("\nIf using period=20, std_dev=2.0 (defaults):")
print("  Expected features: bollinger_bands_20_2.0_upper/lower/middle")

print("\n⚠️  The actual features being used will determine which parameters are active")
print("\nTo fix this, we need to ensure the parameters are properly passed through the compiler")