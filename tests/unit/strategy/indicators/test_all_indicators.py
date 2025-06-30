#!/usr/bin/env python3
"""
Run all indicator strategy tests.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run all indicator tests."""
    test_dir = Path(__file__).parent
    test_files = sorted(test_dir.glob('test_*.py'))
    
    print(f"Running {len(test_files)} test files...")
    
    failed = []
    for test_file in test_files:
        if test_file.name == 'test_all_indicators.py':
            continue
            
        print(f"\nRunning {test_file.name}...")
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            failed.append(test_file.name)
            print(f"  FAILED")
            print(result.stdout)
            print(result.stderr)
        else:
            print(f"  PASSED")
    
    if failed:
        print(f"\n{len(failed)} tests failed:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"\nAll tests passed!")
        sys.exit(0)

if __name__ == '__main__':
    main()
