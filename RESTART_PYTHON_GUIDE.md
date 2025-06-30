# How to Restart Python and Clear Bytecode

## Step 1: Clear Python Bytecode Cache

Run the provided script:
```bash
./clear_python_cache.sh
```

Or manually:
```bash
# Remove all __pycache__ directories
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Remove all .pyc files
find . -name "*.pyc" -delete
```

## Step 2: Restart Python

### Option A: Command Line Python
```bash
# Exit current session
exit()
# or Ctrl+D

# Start fresh
python main.py --config config/bollinger/test.yaml
```

### Option B: Jupyter Notebook
1. Click "Kernel" menu
2. Select "Restart & Clear Output"
3. Re-run all cells

### Option C: IPython
```python
# Exit IPython
exit
# or Ctrl+D

# Restart
ipython
```

### Option D: VS Code / PyCharm
1. Close the Python terminal/console
2. Open a new terminal
3. Run your script again

## Step 3: Verify the Fix is Loaded

After restarting, run:
```bash
python check_portfolio_state_fix.py
```

This should show that the metadata parameter is present.

## About the 0.075% Gain Issue

If you're seeing many trades exiting at exactly +0.075% (which is your stop loss value), this could indicate:

1. **Stop loss calculation might be inverted** - but the code looks correct
2. **Price data might have the high/low reversed** 
3. **The exit price calculation might be using the wrong price**

Run this after restarting Python:
```bash
python analyze_suspicious_exits.py
```

This will show:
- How many trades exit at exactly +0.075%
- Whether they're marked as stop_loss or take_profit
- Whether they're long or short trades

If long trades are exiting at +0.075% with exit_type='stop_loss', there's definitely a bug in the price calculation or data.