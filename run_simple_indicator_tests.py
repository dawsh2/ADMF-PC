#!/usr/bin/env python3
"""
Simple test runner for indicator strategies that doesn't require pandas.

This validates that all strategies have corresponding test files.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

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

def check_test_exists(module_name):
    """Check if test file exists for a module."""
    test_file = PROJECT_ROOT / 'tests' / 'unit' / 'strategy' / 'indicators' / f'test_{module_name}.py'
    return test_file.exists(), test_file

def validate_test_content(test_file, strategies):
    """Validate that test file contains tests for all strategies."""
    try:
        with open(test_file, 'r') as f:
            content = f.read()
        
        missing_tests = []
        for strategy in strategies:
            # Check if there's a test method for this strategy
            if f'test_{strategy}' not in content and strategy not in content:
                missing_tests.append(strategy)
        
        return len(missing_tests) == 0, missing_tests
    except Exception as e:
        return False, [str(e)]

def main():
    """Main test validation."""
    print("Discovering indicator strategies...")
    strategies_by_module = discover_indicator_strategies()
    
    total_strategies = sum(len(s) for s in strategies_by_module.values())
    print(f"\nFound {len(strategies_by_module)} modules with {total_strategies} strategies total:")
    
    results = []
    
    for module_name, strategies in sorted(strategies_by_module.items()):
        print(f"\n{module_name}.py ({len(strategies)} strategies):")
        print(f"  Strategies: {', '.join(strategies)}")
        
        # Check if test file exists
        exists, test_file = check_test_exists(module_name)
        
        if not exists:
            print(f"  ❌ Test file missing: {test_file}")
            results.append({
                'module': module_name,
                'strategies': strategies,
                'status': 'MISSING',
                'test_file': str(test_file)
            })
        else:
            # Validate test content
            valid, missing = validate_test_content(test_file, strategies)
            
            if valid:
                print(f"  ✅ Test file exists with all strategy tests")
                results.append({
                    'module': module_name,
                    'strategies': strategies,
                    'status': 'VALID',
                    'test_file': str(test_file)
                })
            else:
                print(f"  ⚠️  Test file exists but missing tests for: {', '.join(missing)}")
                results.append({
                    'module': module_name,
                    'strategies': strategies,
                    'status': 'INCOMPLETE',
                    'test_file': str(test_file),
                    'missing_tests': missing
                })
    
    # Create summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_modules': len(results),
        'total_strategies': total_strategies,
        'valid': sum(1 for r in results if r['status'] == 'VALID'),
        'incomplete': sum(1 for r in results if r['status'] == 'INCOMPLETE'),
        'missing': sum(1 for r in results if r['status'] == 'MISSING'),
        'details': results
    }
    
    # Save report
    report_path = PROJECT_ROOT / 'indicator_test_validation.json'
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("TEST VALIDATION SUMMARY")
    print("="*80)
    print(f"Total Modules: {summary['total_modules']}")
    print(f"Total Strategies: {summary['total_strategies']}")
    print(f"Valid Test Files: {summary['valid']}")
    print(f"Incomplete Test Files: {summary['incomplete']}")
    print(f"Missing Test Files: {summary['missing']}")
    print(f"\nDetailed report saved to: {report_path}")
    
    # Return appropriate exit code
    if summary['missing'] > 0 or summary['incomplete'] > 0:
        print("\n⚠️  Some strategies are missing tests!")
        print("Run: python generate_indicator_tests.py")
        return 1
    else:
        print("\n✅ All strategies have test files!")
        return 0

if __name__ == '__main__':
    sys.exit(main())