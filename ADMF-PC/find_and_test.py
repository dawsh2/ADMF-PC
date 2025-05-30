#!/usr/bin/env python3
"""Find virtual environment and show how to run tests."""

import os
import sys
import subprocess
from pathlib import Path

print("Virtual Environment Finder")
print("=" * 50)

# Common venv names and locations
venv_names = ['venv', 'env', '.venv', '.env', 'virtualenv']
search_paths = [
    Path.cwd(),  # Current directory
    Path.cwd().parent,  # Parent directory
    Path.home() / 'envs',  # Common home location
    Path.home() / '.virtualenvs',  # virtualenvwrapper default
    Path.home() / '.local' / 'share' / 'virtualenvs',  # pipenv default
]

found_venvs = []

print("\nSearching for virtual environments...")
for base_path in search_paths:
    if not base_path.exists():
        continue
    
    for venv_name in venv_names:
        venv_path = base_path / venv_name
        activate_path = venv_path / 'bin' / 'activate'
        
        if activate_path.exists():
            found_venvs.append(venv_path)
            print(f"✓ Found: {venv_path}")

# Also check for conda environments
conda_envs_path = Path.home() / 'miniconda3' / 'envs'
if not conda_envs_path.exists():
    conda_envs_path = Path.home() / 'anaconda3' / 'envs'

if conda_envs_path.exists():
    print(f"\nChecking conda environments in {conda_envs_path}...")
    for env_dir in conda_envs_path.iterdir():
        if env_dir.is_dir() and (env_dir / 'bin' / 'python').exists():
            found_venvs.append(env_dir)
            print(f"✓ Found conda env: {env_dir.name}")

print("\n" + "=" * 50)

if found_venvs:
    print("\nFound virtual environments:")
    for i, venv in enumerate(found_venvs, 1):
        print(f"{i}. {venv}")
        
        # Check if it has required packages
        python_path = venv / 'bin' / 'python'
        if python_path.exists():
            try:
                result = subprocess.run(
                    [str(python_path), '-c', 'import numpy, pandas; print("✓ Has numpy & pandas")'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"   {result.stdout.strip()}")
                else:
                    print("   ✗ Missing numpy/pandas")
            except:
                pass
    
    print("\nTo run tests with a virtual environment:")
    print("1. Direct execution:")
    for venv in found_venvs[:2]:  # Show first 2
        print(f"   {venv}/bin/python test_basic_backtest_simple.py")
    
    print("\n2. Or activate and run:")
    if found_venvs:
        print(f"   source {found_venvs[0]}/bin/activate")
        print("   python test_basic_backtest_simple.py")
else:
    print("\nNo virtual environments found in common locations.")
    print("\nTo create a new virtual environment with required packages:")
    print("   python3 -m venv venv")
    print("   source venv/bin/activate")
    print("   pip install numpy pandas matplotlib scipy")

print("\n" + "=" * 50)
print("Testing with system Python...")
print(f"System Python: {sys.executable}")
print(f"Version: {sys.version.split()[0]}")

# Check system packages
packages = ['numpy', 'pandas', 'matplotlib', 'scipy']
print("\nSystem packages:")
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg}")
    except ImportError:
        print(f"✗ {pkg}")

print("\n" + "=" * 50)
print("\nIf you know your venv path, you can:")
print("1. Update run_tests.sh with the correct VENV_PATH")
print("2. Or run directly: /your/venv/path/bin/python test_basic_backtest_simple.py")