#!/usr/bin/env python3
"""
Test runner for all indicator strategies.

This script discovers and runs tests for all strategies in src/strategy/strategies/indicators/
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def discover_indicator_strategies():
    """Discover all strategy functions in the indicators module."""
    strategies_by_module = {}
    indicators_path = PROJECT_ROOT / 'src' / 'strategy' / 'strategies' / 'indicators'
    
    # Sort to ensure consistent order
    py_files = sorted(indicators_path.glob('*.py'))
    
    for py_file in py_files:
        if py_file.name in ['__init__.py', 'crossovers_migrated.py']:
            continue
        # Skip temporary/hidden files
        if py_file.name.startswith('.'):
            continue
            
        module_name = py_file.stem
        strategies = []
        
        # Parse file to find @strategy decorators
        try:
            with open(py_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if line.strip().startswith('@strategy('):
                        # Look for the function name in the next few lines
                        # The decorator can span many lines, so look further ahead
                        for j in range(i+1, min(i+20, len(lines))):
                            if lines[j].strip().startswith('def '):
                                func_name = lines[j].strip().split('(')[0].replace('def ', '')
                                strategies.append(func_name)
                                break
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
            continue
        
        if strategies:
            strategies_by_module[module_name] = strategies
    
    return strategies_by_module

def run_tests_for_module(module_name, strategies):
    """Run tests for a specific module's strategies."""
    test_file = PROJECT_ROOT / 'tests' / 'unit' / 'strategy' / 'indicators' / f'test_{module_name}.py'
    
    if not test_file.exists():
        return {
            'module': module_name,
            'status': 'NO_TESTS',
            'strategies': strategies,
            'message': f'Test file not found: {test_file}'
        }
    
    # First check if pytest is available
    pytest_check = subprocess.run([sys.executable, '-m', 'pytest', '--version'], capture_output=True)
    if pytest_check.returncode != 0:
        # Fallback to running directly
        cmd = [sys.executable, str(test_file)]
    else:
        # Run pytest for this specific test file
        cmd = [sys.executable, '-m', 'pytest', str(test_file), '-v', '--tb=short']
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        'module': module_name,
        'status': 'PASSED' if result.returncode == 0 else 'FAILED',
        'strategies': strategies,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'returncode': result.returncode
    }

def create_test_report(results):
    """Create a comprehensive test report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_modules': len(results),
            'passed': sum(1 for r in results if r['status'] == 'PASSED'),
            'failed': sum(1 for r in results if r['status'] == 'FAILED'),
            'missing': sum(1 for r in results if r['status'] == 'NO_TESTS'),
        },
        'details': results
    }
    
    # Save JSON report
    report_path = PROJECT_ROOT / 'indicator_test_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create human-readable summary
    print("\n" + "="*80)
    print("INDICATOR STRATEGY TEST REPORT")
    print("="*80)
    print(f"Timestamp: {report['timestamp']}")
    print(f"\nSummary:")
    print(f"  Total Modules: {report['summary']['total_modules']}")
    print(f"  Passed: {report['summary']['passed']}")
    print(f"  Failed: {report['summary']['failed']}")
    print(f"  Missing Tests: {report['summary']['missing']}")
    
    print("\n" + "-"*80)
    print("MODULE DETAILS:")
    print("-"*80)
    
    for result in results:
        status_symbol = {
            'PASSED': '✓',
            'FAILED': '✗',
            'NO_TESTS': '?'
        }[result['status']]
        
        print(f"\n{status_symbol} {result['module']}.py ({len(result['strategies'])} strategies)")
        print(f"  Status: {result['status']}")
        print(f"  Strategies: {', '.join(result['strategies'])}")
        
        if result['status'] == 'NO_TESTS':
            print(f"  Message: {result['message']}")
        elif result['status'] == 'FAILED':
            print(f"  Error: See detailed output in indicator_test_report.json")
    
    return report

def main():
    """Main test runner."""
    print("Discovering indicator strategies...")
    strategies_by_module = discover_indicator_strategies()
    
    print(f"Found {len(strategies_by_module)} modules with strategies:")
    for module, strategies in strategies_by_module.items():
        print(f"  - {module}: {len(strategies)} strategies")
    
    # Create test directory if it doesn't exist
    test_dir = PROJECT_ROOT / 'tests' / 'unit' / 'strategy' / 'indicators'
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning tests...")
    results = []
    
    for module_name, strategies in strategies_by_module.items():
        print(f"\nTesting {module_name}...")
        result = run_tests_for_module(module_name, strategies)
        results.append(result)
    
    # Create report
    report = create_test_report(results)
    
    # Return appropriate exit code
    if report['summary']['failed'] > 0 or report['summary']['missing'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()