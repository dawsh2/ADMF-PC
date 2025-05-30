#!/usr/bin/env python3
"""Check Python environment and available packages."""

import sys
import os
import subprocess

print("Python Environment Check")
print("=" * 50)

print(f"\nPython Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print(f"Python Path: {sys.path[0]}")

print("\nEnvironment Variables:")
print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'Not set')}")
print(f"PATH: {os.environ.get('PATH', 'Not set')[:100]}...")

print("\nChecking for packages:")
packages = ['numpy', 'pandas', 'matplotlib', 'scipy']

for package in packages:
    try:
        __import__(package)
        print(f"✓ {package} is installed")
    except ImportError:
        print(f"✗ {package} is NOT installed")

print("\nPip list (first 20 packages):")
try:
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                          capture_output=True, text=True)
    lines = result.stdout.split('\n')[:22]  # Header + 20 packages
    for line in lines:
        print(line)
except Exception as e:
    print(f"Error running pip: {e}")

print("\nTo activate your virtual environment, run:")
print("source /path/to/your/venv/bin/activate")
print("\nThen run your tests with:")
print("python test_basic_backtest_simple.py")