#!/usr/bin/env python3
"""Check if OHLC implementation is correct."""

import importlib.util
import sys

# Load the strategy module directly
spec = importlib.util.spec_from_file_location("volatility", "src/strategy/strategies/indicators/volatility.py")
volatility = importlib.util.module_from_spec(spec)
sys.modules["volatility"] = volatility
spec.loader.exec_module(volatility)

# Check if bollinger_bands function has OHLC in metadata
import inspect

print("=== Checking Bollinger Bands Implementation ===")

# Get the source code
source = inspect.getsource(volatility.bollinger_bands)

# Check for OHLC fields
if "'open': bar.get('open', 0)" in source:
    print("✓ 'open' field found in metadata")
else:
    print("❌ 'open' field NOT found in metadata")

if "'high': bar.get('high', 0)" in source:
    print("✓ 'high' field found in metadata")
else:
    print("❌ 'high' field NOT found in metadata")

if "'low': bar.get('low', 0)" in source:
    print("✓ 'low' field found in metadata")
else:
    print("❌ 'low' field NOT found in metadata")

if "'close': bar.get('close', 0)" in source:
    print("✓ 'close' field found in metadata")
else:
    print("❌ 'close' field NOT found in metadata")

# Show the actual metadata section
print("\n\nActual metadata section:")
lines = source.split('\n')
in_metadata = False
for line in lines:
    if "'metadata': {" in line:
        in_metadata = True
    if in_metadata:
        print(line)
        if "}" in line and not "{" in line:
            break

# Also check risk manager
print("\n\n=== Checking Risk Manager Implementation ===")
spec2 = importlib.util.spec_from_file_location("strategy_risk_manager", "src/risk/strategy_risk_manager.py")
risk_manager = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(risk_manager)

source2 = inspect.getsource(risk_manager.StrategyRiskManager.evaluate_signal)

if "bar_data = {" in source2:
    print("✓ bar_data extraction found in risk manager")
else:
    print("❌ bar_data extraction NOT found in risk manager")

if "position_bar_data['low']" in source2:
    print("✓ Using low price for stop loss checks")
else:
    print("❌ NOT using low price for stop loss checks")

if "position_bar_data['high']" in source2:
    print("✓ Using high price for take profit checks")
else:
    print("❌ NOT using high price for take profit checks")