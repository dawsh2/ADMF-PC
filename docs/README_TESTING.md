# Running ADMF-PC Tests

## Quick Start (With Your Virtual Environment)

Since you have numpy and pandas in your venv, here's how to run the full test suite:

### Step 1: Restore the Monitoring Imports
First, undo our temporary fixes:
```bash
cd /Users/daws/ADMF-PC/ADMF-PC
git checkout src/core/infrastructure/capabilities.py src/core/infrastructure/__init__.py
```

### Step 2: Activate Your Virtual Environment
```bash
# Replace with your actual venv activation command
source /path/to/your/venv/bin/activate

# Verify you're in the venv
which python  # Should show your venv path
python --version
```

### Step 3: Run the Tests
```bash
# Full integration test with pandas/numpy
python test_basic_backtest_simple.py

# Minimal integration test
python test_minimal_integration.py

# Ultra minimal test (no dependencies)
python test_ultra_minimal.py
```

## Test Files Overview

1. **`test_ultra_minimal.py`** - No external dependencies, pure Python
   - Tests basic portfolio math
   - Signal → Order flow
   - P&L calculation

2. **`test_direct_imports.py`** - Minimal imports, works without numpy
   - Tests actual ADMF-PC modules
   - Risk limits
   - Signal aggregation

3. **`test_minimal_integration.py`** - Attempts full module imports
   - Currently blocked by numpy dependency

4. **`test_basic_backtest_simple.py`** - Full integration test
   - Uses pandas for data generation
   - Complete backtest simulation
   - Requires your venv

## Without Virtual Environment

If you want to test without setting up dependencies:

```bash
# This works with system Python (no numpy/pandas needed)
python3 test_ultra_minimal.py
python3 test_direct_imports.py
```

## Finding Your Virtual Environment

Common locations to check:
- `~/envs/`
- `~/.virtualenvs/`
- `~/miniconda3/envs/`
- Project-specific: `./venv`, `./.venv`

To find all Python environments:
```bash
# Find all activate scripts
find ~ -name "activate" -path "*/bin/activate" 2>/dev/null | grep -v ".Trash"

# Check conda environments
conda env list

# Check pipenv
pipenv --venv
```

## Creating a New Virtual Environment

If you need to create a fresh venv:
```bash
cd /Users/daws/ADMF-PC/ADMF-PC
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas matplotlib scipy pyyaml
```

## Alternative: Direct Python Path

If you know your venv's Python path, you can run directly:
```bash
# Example (replace with your actual path)
~/envs/admf/bin/python test_basic_backtest_simple.py
```