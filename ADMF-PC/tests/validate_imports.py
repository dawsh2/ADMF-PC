"""
Validate that all test modules can be imported successfully.
"""

import sys
import os
import importlib
import traceback

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

test_modules = [
    "tests.test_config",
    "tests.test_data",
    "tests.test_execution.test_market_simulation",
    "tests.test_execution.test_order_flow",
    "tests.test_execution.test_unified_backtest",
    "tests.test_integration.test_full_backtest_flow",
    "tests.test_integration.test_risk_execution_integration",
    "tests.test_integration.test_signal_to_fill_flow",
    "tests.test_risk.test_signal_advanced",
    "tests.test_risk.test_signal_flow",
    "tests.test_strategies.test_example_strategies",
]

print("Validating test module imports...\n")

success_count = 0
failed_modules = []

for module_name in test_modules:
    try:
        print(f"Importing {module_name}...", end=" ")
        importlib.import_module(module_name)
        print("✓")
        success_count += 1
    except Exception as e:
        print(f"✗ - {type(e).__name__}: {str(e)}")
        failed_modules.append((module_name, str(e)))
        if "--verbose" in sys.argv:
            traceback.print_exc()

print(f"\nSummary: {success_count}/{len(test_modules)} modules imported successfully")

if failed_modules:
    print("\nFailed imports:")
    for module, error in failed_modules:
        print(f"  - {module}: {error}")
else:
    print("\nAll test modules imported successfully!")