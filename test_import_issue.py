#!/usr/bin/env python3
"""
Test to find the duckdb import issue.
"""

import traceback

print("Testing imports...")

try:
    print("1. Importing SimpleHierarchicalStorage...")
    from src.core.events.storage.hierarchical_parquet import SimpleHierarchicalStorage
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    traceback.print_exc()

try:
    print("\n2. Importing HierarchicalPortfolioTracer...")
    from src.core.events.observers.hierarchical_portfolio_tracer import HierarchicalPortfolioTracer
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    traceback.print_exc()

try:
    print("\n3. Importing sparse_storage...")
    from src.analytics.storage.sparse_storage import SparseSignalStorage
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    traceback.print_exc()

try:
    print("\n4. Importing analytics module...")
    import src.analytics
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    traceback.print_exc()

print("\nDone")