#!/usr/bin/env python3
"""
Runner for isolated tests.

These tests validate core logic without importing the main codebase,
so they should always work even when there are import issues.
"""

import unittest
import sys
import os

def run_isolated_tests():
    """Run all isolated tests."""
    print("=" * 60)
    print("RUNNING ISOLATED TESTS")
    print("=" * 60)
    print("These tests validate core logic without imports")
    print("They should always pass even if main code has issues")
    print()
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    result = runner.run(suite)
    
    print()
    print("=" * 60)
    print("ISOLATED TESTS SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_isolated_tests()
    sys.exit(0 if success else 1)