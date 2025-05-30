#!/usr/bin/env python3
"""
Test runner for ADMF-PC test suite.

Runs all tests with proper configuration and reporting.
"""

import sys
import os
import unittest
import argparse
from datetime import datetime
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def configure_logging(verbosity):
    """Configure logging for tests."""
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def discover_tests(test_dir, pattern='test_*.py'):
    """Discover all tests in the given directory."""
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern, top_level_dir=project_root)
    return suite


def run_test_suite(suite, verbosity=1):
    """Run a test suite and return results."""
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    print(f"\n{'='*70}")
    print(f"ADMF-PC Test Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    result = runner.run(suite)
    
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    return result


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description='Run ADMF-PC test suite')
    
    parser.add_argument(
        'test_path',
        nargs='?',
        default='tests',
        help='Path to test file or directory (default: tests)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=1,
        help='Increase verbosity (can be repeated: -v, -vv, -vvv)'
    )
    
    parser.add_argument(
        '-p', '--pattern',
        default='test_*.py',
        help='Test file pattern (default: test_*.py)'
    )
    
    parser.add_argument(
        '-f', '--failfast',
        action='store_true',
        help='Stop on first failure'
    )
    
    parser.add_argument(
        '--execution',
        action='store_true',
        help='Run only execution module tests'
    )
    
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run only integration tests'
    )
    
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run only unit tests'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(args.verbose)
    
    # Determine test path
    test_path = args.test_path
    if not os.path.isabs(test_path):
        test_path = os.path.join(project_root, test_path)
    
    # Handle specific test categories
    if args.execution:
        test_path = os.path.join(project_root, 'tests', 'test_execution')
    elif args.integration:
        test_path = os.path.join(project_root, 'tests', 'test_integration')
    elif args.unit:
        # Run all non-integration tests
        suite = unittest.TestSuite()
        for subdir in ['test_execution', 'test_strategies']:
            subpath = os.path.join(project_root, 'tests', subdir)
            if os.path.exists(subpath):
                suite.addTests(discover_tests(subpath, args.pattern))
        result = run_test_suite(suite, verbosity=args.verbose)
        return 0 if result.wasSuccessful() else 1
    
    # Discover and run tests
    if os.path.isfile(test_path):
        # Run specific test file
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(
            test_path.replace(project_root + '/', '').replace('/', '.').replace('.py', '')
        )
    else:
        # Discover tests in directory
        suite = discover_tests(test_path, args.pattern)
    
    # Run tests
    result = run_test_suite(suite, verbosity=args.verbose)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())