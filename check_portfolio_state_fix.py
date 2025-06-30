#!/usr/bin/env python3
"""Check if the portfolio state fix is actually in the code."""

import inspect
from src.portfolio.state import PortfolioState

print("=== Checking Portfolio State Fix ===")

# Get the update_position method signature
sig = inspect.signature(PortfolioState.update_position)
print(f"\nupdate_position signature: {sig}")

params = list(sig.parameters.keys())
print(f"\nParameters: {params}")

if 'metadata' in params:
    print("\n✓ SUCCESS: 'metadata' parameter is present in update_position!")
    print("The fix has been applied to the code.")
    
    # Check default value
    metadata_param = sig.parameters['metadata']
    print(f"Metadata parameter details: {metadata_param}")
    print(f"Default value: {metadata_param.default}")
else:
    print("\n❌ ERROR: 'metadata' parameter NOT found in update_position!")
    print("The fix has not been applied or module needs reloading.")

# Also check the on_fill method to see if it's using metadata
print("\n\nChecking on_fill method implementation...")
source = inspect.getsource(PortfolioState.on_fill)

# Look for key lines that indicate the fix
if "metadata=position_metadata" in source:
    print("✓ on_fill is passing metadata to update_position")
else:
    print("❌ on_fill is NOT passing metadata to update_position")

if "position_metadata = {}" in source:
    print("✓ on_fill creates position_metadata dict")
else:
    print("❌ on_fill does NOT create position_metadata dict")

if "position_metadata['strategy_id'] = strategy_id" in source:
    print("✓ on_fill sets strategy_id in position_metadata")
else:
    print("❌ on_fill does NOT set strategy_id in position_metadata")

print("\n=== Action Required ===")
print("If the fix is not detected:")
print("1. Make sure you saved the changes to src/portfolio/state.py")
print("2. Restart your Python session or reload the module")
print("3. Run your backtest again")
print("\nIf the fix IS detected but you still see 453 trades:")
print("1. Make sure you're running the backtest with the updated code")
print("2. Check that the config has exit_memory_enabled: true")